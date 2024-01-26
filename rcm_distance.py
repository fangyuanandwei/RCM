import numpy as np
import os
from tqdm import tqdm


if __name__ == '__main__':
    image_path = './results/synthia/r101_g2c_ours_proca/soft_labels/cityscapes_train_npy'
    image_list = np.load('./results/gta5/r101_g2c_ours_proca/name_list.npy').tolist()

    class_number = 19
    Target_number = len(image_list)

    Target_maskBincount  = []
    Target_maskSM = np.zeros([Target_number,Target_number])


    Target_classBincount = []
    Target_classSM = np.zeros([Target_number,Target_number])

    for index_i in tqdm(range(Target_number)):
        image_name = image_list[index_i]
        image_name = image_name.split('.')[0] + '.npy'
        image_file = np.load(os.path.join(image_path,image_name))

        '''mask'''
        image_mask         = (image_file < 250).astype(int)
        Target_maskBincount.append(image_mask)

        '''class'''
        image_flatten      = image_file.flatten()
        class_bincount     = np.bincount(image_flatten)
        class_bincount1    = class_bincount[:19]
        class_bincount2    = class_bincount[255]
        class_bincount_int = (class_bincount1>1).astype(int)
        Target_classBincount.append(class_bincount_int)

    print('Target_classSM')
    for i in tqdm(range(Target_number)):
        for j in range(i,Target_number):
            S = Target_classBincount[i]
            T = Target_classBincount[j]

            'Class haming Distance'
            class_smstr = np.nonzero(S - T)
            class_sm    = np.shape(class_smstr[0])[0]/class_number
            Target_classSM[i,j] = class_sm
            Target_classSM[j,i] = class_sm
            
    np.save('./results/synthia/r101_g2c_ours_proca/Target_classSM.npy', Target_classSM)
    print('Target_maskSM')
    for i in range(Target_number):
        print('Process Target_maskSM:{}'.format(i))
        for j in tqdm(range(i,Target_number)):
            if i==j:
                Target_maskSM[i, j] = 0
            else:
                S = Target_maskBincount[i].flatten()
                T = Target_maskBincount[j].flatten()

                'Mask haming Distance'
                Mask_smstr = np.nonzero(S - T)
                Mask_sm    = np.shape(Mask_smstr[0])[0]/S.size
                Target_maskSM[i,j] = Mask_sm
                Target_maskSM[j,i] = Mask_sm

    np.save('./results/synthia/r101_g2c_ours_proca/Target_maskSM.npy', Target_maskSM)
    Target_SM = Target_classSM+Target_maskSM
    np.save('./results/synthia/r101_g2c_ours_proca/Target_SM.npy', Target_SM)



