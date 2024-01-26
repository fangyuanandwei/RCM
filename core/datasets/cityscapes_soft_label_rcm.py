import os
import numpy as np
from torch.utils import data
from PIL import Image
import random


class CRM_cityscapesSoftLabelDataSet(data.Dataset):
    """
            original softlabel resolution at 2048x1024
    """
    def __init__(
            self,
            data_root,
            data_list,
            label_dir,
            max_iters=None,
            num_classes=19,
            split="train",
            transform=None,
            ignore_label=255,
            debug=False,
            RCM_distance = None

    ):
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.label_dir = label_dir
        self.data_list = []
        self.RCM_distance = RCM_distance
        self.data_root = '/media/lxj/d3c963a4-6608-4513-b391-9f9eb8c10a7a/DA/Data/CITY/'

        with open(data_list, "r") as handle:
            content = handle.readlines()

        for fname in content:
            name = fname.strip()
            self.data_list.append(
                {
                    "img": os.path.join(
                        self.data_root, "leftImg8bit/%s/%s" % (self.split, name)
                    ),
                    "label": os.path.join(
                        self.label_dir,
                        name.split('/')[1],
                    ),
                    "name": name,
                }
            )

        if max_iters is not None:
            self.target_number = len(self.data_list)
            self.epochs    = int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list * self.epochs

        # GTA5, Synscapes, cross-city
        self.id_to_trainid = {
            7: 0,
            8: 1,
            11: 2,
            12: 3,
            13: 4,
            17: 5,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            31: 16,
            32: 17,
            33: 18,
        }
        self.trainid2name = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "light",
            7: "sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motocycle",
            18: "bicycle",
        }

        if self.NUM_CLASS == 16:  # SYNTHIA
            self.id_to_trainid = {
                7: 0,
                8: 1,
                11: 2,
                12: 3,
                13: 4,
                17: 5,
                19: 6,
                20: 7,
                21: 8,
                23: 9,
                24: 10,
                25: 11,
                26: 12,
                28: 13,
                32: 14,
                33: 15,
            }
            self.trainid2name = {
                0: "road",
                1: "sidewalk",
                2: "building",
                3: "wall",
                4: "fence",
                5: "pole",
                6: "light",
                7: "sign",
                8: "vegetation",
                9: "sky",
                10: "person",
                11: "rider",
                12: "car",
                13: "bus",
                14: "motocycle",
                15: "bicycle",
            }

        self.transform = transform

        self.ignore_label = ignore_label

        self.debug = debug



    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        image, label, name = self.processImage(index)
        ori_index   = np.mod(index,self.target_number)
        classMatrix = self.RCM_distance[ori_index]
        sortIndex   = random.choice(np.argsort(classMatrix)[-5:])
        # sortIndex = random.randint(0, len(self.data_list)-1)
        image_crm, label_crm, name_crm = self.processImage(sortIndex)

        return image, label, name,image_crm, label_crm, name_crm


    def processImage(self,index):
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = np.array(Image.open(datafiles["label"]), dtype=np.float32)
        name = datafiles["name"]

        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k in self.trainid2name.keys():
            label_copy[label == k] = k
        label = Image.fromarray(label_copy)

        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label, name
