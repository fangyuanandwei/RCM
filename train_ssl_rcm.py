import argparse
import os
import datetime
import logging
import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU,get_color_pallete
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

def strip_prefix_if_present(state_dict, prefix):
    from collections import OrderedDict
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix + 'layer5'):
            continue
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict



def Mask4Gen(src_label, rcm_label):
    h, w  = rcm_label.shape
    mask_fusingEasy = torch.zeros(h, w)
    src_label = src_label.int()
    rcm_label = rcm_label.int()
    'classIndex'
    src_label_flatten = src_label.flatten()
    rcm_label_flatten = rcm_label.flatten()
    src_label_bincount    = torch.bincount(src_label_flatten.int(),minlength = 256)[:19]
    rcm_label_bincount    = torch.bincount(rcm_label_flatten.int(),minlength = 256)[:19]

    try:
        class_sampler = (src_label_bincount < 100) & (rcm_label_bincount > 100)
    except RuntimeError:
        print(src_label_bincount)
        print(rcm_label_bincount)

    classIndex    = torch.nonzero(class_sampler).flatten()

    '''0为取CRM像素'''
    Mask1 = (src_label == 255) & (rcm_label != 255)
    mask_fusingEasy[Mask1] = 0

    '''1为取原像素'''
    Mask2 = (src_label == 255) & (rcm_label == 255)
    mask_fusingEasy[Mask2] = 1

    '''1为取原像素'''
    Mask3 = (src_label != 255) & (rcm_label == 255)
    mask_fusingEasy[Mask3] = 1

    '''首先让mask_fusingEasy都取原像素'''
    Mask4 = (src_label != 255) & (rcm_label != 255)
    mask_fusingEasy[Mask4] = 1

    '''
    首先查找ImageCMPred中的ClassIndex位置 Mask_classIndex
    选择既在Mask4中，也在Mask_classIndex中的位置
    '''
    Mask_class = torch.zeros(h, w).byte()
    for classvalue in classIndex:
        classMask  = (rcm_label == classvalue)
        Mask_class = Mask_class | classMask
    Mask_classIndex = Mask4 & Mask_class
    mask_fusingEasy[Mask_classIndex] = 0

    return mask_fusingEasy


def train(cfg, local_rank, distributed, logger):
    logger.info("Start training")

    feature_extractor = build_feature_extractor(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor.to(device)

    classifier = build_classifier(cfg)
    classifier.to(device)

    batch_size = cfg.SOLVER.BATCH_SIZE
    if distributed:
        pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        batch_size = int(cfg.SOLVER.BATCH_SIZE / torch.distributed.get_world_size())
        if not cfg.MODEL.FREEZE_BN:
            feature_extractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_extractor)
        feature_extractor = torch.nn.parallel.DistributedDataParallel(
            feature_extractor, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg1
        )
        pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg2
        )
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()
    
    if local_rank == 0:
        print(feature_extractor)
        print(classifier)

    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()

    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR * 10, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = local_rank == 0

    iteration = 0

    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        model_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(model_weights)
        classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)

    RCM_distance = np.load(os.path.join(cfg.PREPARE_DIR,'Target_classSM.npy'))
    src_train_data = build_dataset(cfg, mode='train', is_source=True, RCM_distance=RCM_distance)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(src_train_data)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        src_train_data,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )

    ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    max_iters = cfg.SOLVER.MAX_ITER
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    feature_extractor.train()
    classifier.train()
    start_training_time = time.time()
    end = time.time()
    best_mIoU = 0
    best_iteration = 0

    for i, (src_input, src_label, src_name,
            rcm_input, rcm_label, rcm_name) in enumerate(train_loader):
        data_time  = time.time() - end
        current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters,
                                          power=cfg.SOLVER.LR_POWER)
        for index in range(len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[index]['lr'] = current_lr * 10

        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()

        mix_image = []
        mix_label = []
        masks     = []
        for bi in range(batch_size):
            src_image_bi = src_input[bi]
            rcm_image_bi = rcm_input[bi]
            src_label_bi = src_label[bi]
            rcm_label_bi = rcm_label[bi]

            mask = Mask4Gen(src_label_bi,rcm_label_bi)
            masks.append(mask)
            'mask中1为取src像素,0为取rcm像素'
            mix_image_bi = src_image_bi * mask + rcm_image_bi * (1-mask)
            mix_label_bi = src_label_bi * mask + rcm_label_bi * (1-mask)
            mix_image.append(mix_image_bi)
            mix_label.append(mix_label_bi)

        mix_input = torch.stack(mix_image, 0)
        mix_label = torch.stack(mix_label, 0)
        masks     = torch.stack(masks, 0)

        if i % 1000 == 0:
            save_image(i,
                       src_input[0],src_label[0],src_name[0],
                       rcm_input[0],rcm_label[0],rcm_name[0],
                       mix_input[0],mix_label[0],masks[0],
                       output_dir)


        mix_input = mix_input.cuda(non_blocking=True)
        mix_label = mix_label.cuda(non_blocking=True).long()
        size = mix_label.shape[-2:]
        mix_pred = classifier(feature_extractor(mix_input), size)
        loss_mix = ce_criterion(mix_pred, mix_label)


        loss = loss_mix
        loss.backward()

        optimizer_fea.step()
        optimizer_cls.step()
        meters.update(loss_seg=loss.item())
        iteration += 1

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
        eta_string  = str(datetime.timedelta(seconds=int(eta_seconds)))
        if iteration % 20 == 0 or iteration == max_iters:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.2f} GB",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer_fea.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                )
            )

        if (iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or iteration == cfg.SOLVER.STOP_ITER):
        # if iteration % 10 ==0:
            current_mIoU, current_mAcc, current_allAcc = run_test(cfg, feature_extractor, classifier, local_rank, distributed, logger)
            feature_extractor.train()
            classifier.train()
            if save_to_disk:
                # update best model
                if current_mIoU > best_mIoU:
                    filename = os.path.join(output_dir, "model_best_{}.pth".format(current_mIoU))
                    torch.save({'iteration': iteration, 'feature_extractor': feature_extractor.state_dict(),
                        'classifier': classifier.state_dict(), 'optimizer_fea': optimizer_fea.state_dict(),
                        'optimizer_cls': optimizer_cls.state_dict()}, filename)
                    best_mIoU = current_mIoU
                    best_iteration = iteration
                else:
                    filename = os.path.join(output_dir, "model_current.pth")
                    torch.save({'iteration': iteration, 'feature_extractor': feature_extractor.state_dict(),
                        'classifier': classifier.state_dict(), 'optimizer_fea': optimizer_fea.state_dict(),
                        'optimizer_cls': optimizer_cls.state_dict()}, filename)
                logger.info(f"-------- Best mIoU {best_mIoU} at iteration {best_iteration} --------")
            torch.cuda.empty_cache()

        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / cfg.SOLVER.STOP_ITER
        )
    )

def save_image(i,
               src_input,src_label, src_name,
               rcm_input,rcm_label, rcm_name,
               mix_input, mix_label, masks,
               output_folder):


    save_path = os.path.join(output_folder,'VisualResults',str(i))
    mkdir(save_path)

    'label'
    src_label = get_color_pallete(src_label.cpu().numpy(), "city")
    rcm_label = get_color_pallete(rcm_label.cpu().numpy(), "city")
    mix_label = get_color_pallete(mix_label.cpu().numpy(), "city")

    width1,height1 = src_label.size
    new_label = Image.new('RGB', (width1 * 3 + 20 * (3- 1), height1), (255, 255, 255))
    new_label.paste(src_label, (0, 0))
    new_label.paste(rcm_label, ((width1+20)*1, 0))
    new_label.paste(mix_label, ((width1+20)*2, 0))
    new_label_name = src_name.split('/')[1].split('.')[0]+'_'+rcm_name.split('/')[1].split('.')[0]+'.png'
    new_label.save(os.path.join(save_path,new_label_name))

    'image'
    data_root = '/media/lxj/d3c963a4-6608-4513-b391-9f9eb8c10a7a/DA/Data/CITY/leftImg8bit/train/'
    src_image_path = os.path.join(data_root,src_name)
    rcm_image_path = os.path.join(data_root,rcm_name)

    src_image = Image.open(src_image_path).convert('RGB')
    rcm_image = Image.open(rcm_image_path).convert('RGB')
    src_image = src_image.resize((width1, height1))
    rcm_image = rcm_image.resize((width1, height1))

    masks     = masks.numpy().astype(int)
    rev_maks  = 1-masks
    src_image = np.array(src_image)
    rcm_image = np.array(rcm_image)
    masks = np.broadcast_to(masks[..., None], (512, 1024, 3))
    rev_maks = np.broadcast_to(rev_maks[..., None], (512, 1024, 3))
    src_image_maks = src_image * masks
    rcm_image_maks = rcm_image * rev_maks
    mix_image = src_image_maks + rcm_image_maks

    src_image = Image.fromarray(np.uint8(src_image))
    rcm_image = Image.fromarray(np.uint8(rcm_image))
    mix_image = Image.fromarray(np.uint8(mix_image))

    new_input  = Image.new('RGB', (width1 * 3 + 20 * (3- 1), height1), (255, 255, 255))
    new_input.paste(src_image, (0, 0))
    new_input.paste(rcm_image, ((width1+20)*1, 0))
    new_input.paste(mix_image, ((width1+20)*2, 0))
    new_input.save(os.path.join(save_path,'Image.png'))



def run_test(cfg, feature_extractor, classifier, local_rank, distributed, logger):
    if local_rank == 0:
        logger.info('>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    if distributed:
        feature_extractor, classifier = feature_extractor.module, classifier.module
    torch.cuda.empty_cache()
    dataset_name = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)

    test_data = build_dataset(cfg, mode='test', is_source=False)
    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4,
                                              pin_memory=True, sampler=test_sampler)
    feature_extractor.eval()
    classifier.eval()
    end = time.time()
    with torch.no_grad():
        for i, (x, y, _) in enumerate(test_loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()

            size = y.shape[-2:]
            output = classifier(feature_extractor(x))
            output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
            output = output.max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES,
                                                                  cfg.INPUT.IGNORE_LABEL)
            if distributed:
                torch.distributed.all_reduce(intersection), torch.distributed.all_reduce(
                    union), torch.distributed.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)
            end = time.time()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if local_rank == 0:
        logger.info("Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(mIoU, mAcc, allAcc))
        for i in range(cfg.MODEL.NUM_CLASSES):
            logger.info(
                "Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.".format(i, test_data.trainid2name[i],
                                                                         iou_class[i], accuracy_class[i])
            )
    return mIoU, mAcc, allAcc

def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("SelfSupervised", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, args.local_rank, args.distributed, logger)


if __name__ == "__main__":
    main()
