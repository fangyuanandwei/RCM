import os
if __name__ == '__main__':
    lsit = os.listdir('/media/lxj/d3c963a4-6608-4513-b391-9f9eb8c10a7a/DA/Other/proda/results/gta5/2023_08_16/image/')
    with open('rcm_train_list.txt','w') as f:
        for name in lsit:
            f.write(name+'\n')
