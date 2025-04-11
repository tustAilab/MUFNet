import os
import numpy as np
import pandas as pd
import torch



path_csv = '/media/lab210/D/EchoNet-Dynamic/dynamic/echonet/utils/output/segmentation/deeplabv3_resnet50_random/'
with open(os.path.join(path_csv, "size_all.csv"), "r") as g:
    text = pd.read_csv(g)
    listType = text['Filename'].unique()
    ef_all = []
    fname = []
    for i in range(len(listType)):
#        print(i)
        text_i = text[text['Filename'].isin([listType[i]])]
        fname.append(text_i['Filename'].iloc[0])
        diastole = text_i[text_i['ComputerLarge']==1][['Frame','Size']]
        systole = text_i[text_i['ComputerSmall']==1][['Frame','Size']]
        edv = np.array(diastole['Size'])
        esv = np.array(systole['Size'])
        ef=[]
        for j in range(min(len(edv),len(esv))):
            ef_j = (edv[j] - esv[j]) / edv[j] * 100
            ef.append(ef_j)
        ef_mean = np.mean(ef)
#        ef_std = np.std(ef)
        ef_all.append(ef_mean)

with open(os.path.join(path_csv, "FileList.csv"), "r") as f:
    test_lb = pd.read_csv(f)
    test_vd = test_lb[test_lb['Split']=='TEST']
    ef_lb = list(test_vd['EF'])
    filename = list(test_vd['FileName'])

diff = list(map(lambda x: x[0]-x[1], zip(ef_all, ef_lb[0:1275])))


with open(os.path.join(path_csv, "diff.csv"), "w") as h:
    h.write("Filename,ComputerEF,HumanEF,Difference\n")
    for m in range(1276):
        print(m)
        h.write("{},{},{},{}\n".format(filename[m], ef_all[m], ef_lb[m], diff[m]))