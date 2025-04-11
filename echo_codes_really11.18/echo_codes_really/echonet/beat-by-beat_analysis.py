import torch
import csv
import pandas as pd

pred_path = '/media/lab210/D/EchoNet-Dynamic/dynamic/echonet/utils/output/video/r2plus1d_18_32_2_pretrained/test_predictions.csv'
#with open(pred_path, 'r') as f:
#    data = csv.reader(f)
data = pd.read_csv(pred_path)
size = pd.read_csv('/media/lab210/D/EchoNet-Dynamic/dynamic/echonet/utils/output/segmentation/deeplabv3_resnet50_random/size.csv')
print(size[0])