#可视化结果
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from medpy import metric
from scipy import ndimage
from medpy.metric.binary import hd

from models_flow_unet import Unsupervised
from echo_codes_really import echonet
from scipy.spatial.distance import directed_hausdorff
from train_new import run_epoch
from utils_new import unsup_loss


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
mean, std =0,1
tasks_2 = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
kwargs_2 = {"target_type": tasks_2,
                "mean": mean,
                "std": std
                }
path = '/media/ubuntu/e/mengjinpeng/echo_codes_really11.18/echo_codes_really/Checkpoints/funet_try/Unsupervised/FlowNetS/best_92.69.pt'
mymodel = Unsupervised().to(device)
mymodel.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])
mymodel = mymodel.predictor_s
mymodel.eval()
dataset_test = echonet.datasets.Echo(split='test', **kwargs_2)
test2 = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=2, shuffle=False,pin_memory=(device.type == "cuda"))
loss, large_inter, large_union, small_inter, small_union=run_epoch(mymodel,test2,False,None,device)
overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
print("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format("test", *echonet.utils.bootstrap(
                np.concatenate((large_inter, small_inter)), np.concatenate((large_union, small_union)),
                echonet.utils.dice_similarity_coefficient)))
print("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format("test", *echonet.utils.bootstrap(large_inter,
                                                                                                      large_union,
                                                                        echonet.utils.dice_similarity_coefficient)))
print("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format("test", *echonet.utils.bootstrap(small_inter,
                                                                                                      small_union,
                                                                        echonet.utils.dice_similarity_coefficient)))
#print(overall_dice)
# sun_hd=0
# sum1=0
# for (_, (large_frame, small_frame, large_trace, small_trace)) in test2:
#     large_frame = large_frame.to(device)
#     small_frame = small_frame.to(device)
#     y_large = mymodel(large_frame).squeeze(0).detach().cpu().numpy()>0
#     y_small =mymodel(small_frame).squeeze(0).detach().cpu().numpy()>0
#     large_trace=large_trace.detach().cpu().numpy()
#     small_trace=small_trace.detach().cpu().numpy()
#     # hd1=metric.binary.hd(y_large,large_trace)
#     # hd2=metric.binary.hd(y_small,small_trace)
#     # sum1+=2
#     # sun_hd+=hd1
#     # sun_hd+=hd2
#     #print(hd1,hd2)
# print(sun_hd/sum1)

def load_model(image,coo2,i,j):
    image_o = image.transpose((1, 2, 0))
    img_min2 = image_o.min()
    img_max2=image_o.max()
    img_show2=((image_o-img_min2)/(img_max2-img_min2))*255
    img_show2=img_show2.astype(np.uint8)
    image=torch.from_numpy(image).float()
    image=image.unsqueeze(0)
    path='/media/ubuntu/e/mengjinpeng/echo_codes_really11.18/echo_codes_really/Checkpoints/funet_4/Unsupervised/FlowNetS/best_92.69.pt'
    mymodel = Unsupervised()
    mymodel.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])
    mymodel = mymodel.predictor_s
    mymodel.eval()
    b=mymodel.forward(image)
    resu=b.squeeze().detach().numpy()
    norm=(resu-np.min(resu))/(np.max(resu)-np.min(resu))
    b_arr=np.where(norm>0.5,1,0)
    e = ndimage.binary_erosion(b_arr)
    d = ndimage.binary_dilation(b_arr)
    edge = np.logical_xor(d, e).astype(np.uint8) * 255
    co = np.column_stack(np.where(edge == 255))
    for coord in co:
        img_show2[coord[0], coord[1]] = [0, 0, 255]
    for coord in coo2:
        img_show2[coord[0], coord[1]] = [255, 0, 0]
    image_save=Image.fromarray(img_show2)
    image_save.save(f'/media/ubuntu/e/mengjinpeng/echo_codes_really11.18/echo_codes_really/result_image/imgage{i}——{j}.png')
    # plt.imshow(img_show2)
    # plt.show()

def unlabel():
    c = dataset_test.__getitem__(8)
    frame=c[0]
    for j in range(16):
        i=frame[:,j,:,:]
        # image_o=i.transpose((1, 2, 0))
        # img_min2 = image_o.min()
        # img_max2 = image_o.max()
        # img_show2 = ((image_o - img_min2) / (img_max2 - img_min2)) * 255
        # img_show2 = img_show2.astype(np.uint8)
        # img_show_save=Image.fromarray(img_show2)
        # img_show_save.save(f'/media/ubuntu/e/mengjinpeng/echo_codes_really11.18/echo_codes_really/result_image/unlabel-8-o/img{j}.png')
        img_show2=Image.open( f'/media/ubuntu/e/mengjinpeng/echo_codes_really11.18/echo_codes_really/result_image/unlabel-8/img{j}.png')
        img_show2 = np.array(img_show2)
        image = torch.from_numpy(i).float()
        image = image.unsqueeze(0)
        path = '/media/ubuntu/e/mengjinpeng/echo_codes_really11.18/echo_codes_really/Checkpoints/funet_4/Unsupervised/FlowNetS/best_92.69.pt'
        mymodel = Unsupervised()
        mymodel.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])
        mymodel = mymodel.predictor_s
        mymodel.eval()
        b = mymodel.forward(image)
        resu = b.squeeze().detach().numpy()
        norm = (resu - np.min(resu)) / (np.max(resu) - np.min(resu))
        b_arr = np.where(norm > 0.5, 1, 0)
        e = ndimage.binary_erosion(b_arr)
        d = ndimage.binary_dilation(b_arr)
        edge = np.logical_xor(d, e).astype(np.uint8) * 255
        co = np.column_stack(np.where(edge == 255))
        for coord in co:
            img_show2[coord[0], coord[1]] = [0,0, 255]
        image_save = Image.fromarray(img_show2)
        image_save.save(
            f'/media/ubuntu/e/mengjinpeng/echo_codes_really11.18/echo_codes_really/result_image/unlabel-8/img{j}.png')
#unlabel()

#print('')
def com_img(image,coo2,i,j):
    # image_o = image.transpose((1, 2, 0))
    # img_min2 = image_o.min()
    # img_max2=image_o.max()
    # img_show2=((image_o-img_min2)/(img_max2-img_min2))*255
    # img_show2=img_show2.astype(np.uint8)
    img_show2=Image.open(f'/media/ubuntu/e/mengjinpeng/echo_codes_really11.18/echo_codes_really/result_image/com-img-cs/imgage{i}——{j}.png')
    img_show2=np.array(img_show2)
    image=torch.from_numpy(image).float()
    image=image.unsqueeze(0)
    path='/media/ubuntu/e/mengjinpeng/echo_codes_really11.18/echo_codes_really/Checkpoints/funet_try/Unsupervised/FlowNetS/best_92.69.pt'
    mymodel = Unsupervised()
    mymodel.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])
    mymodel = mymodel.predictor_s
    mymodel.eval()
    b=mymodel.forward(image)
    resu=b.squeeze().detach().numpy()
    norm=(resu-np.min(resu))/(np.max(resu)-np.min(resu))
    b_arr=np.where(norm>0.5,1,0)
    e = ndimage.binary_erosion(b_arr)
    d = ndimage.binary_dilation(b_arr)
    edge = np.logical_xor(d, e).astype(np.uint8) * 255
    co = np.column_stack(np.where(edge == 255))
    for coord in co:
        img_show2[coord[0], coord[1]] = [0, 255, 0]
    for coord2 in coo2:
        img_show2[coord2[0], coord2[1]] = [255, 0, 0]
        #img_show2[coord[0], coord[1]] = [255, 0, 0]
    image_save=Image.fromarray(img_show2)
    image_save.save(f'/media/ubuntu/e/mengjinpeng/echo_codes_really11.18/echo_codes_really/result_image/com-img-cs/imgage{i}——{j}.png')
# for i in range(10,20):
#     b = dataset_test.__getitem__(i)
#     for j in range(2):
#         image=b[1][j]
#         coo2 = b[2][j]
#         com_img(image,coo2,i,j)

#load_model()
# for i in range(10):
#     b = dataset_test.__getitem__(i)
#     for j in range(2):
#         image=b[1][j]
#         com_img(image,i,j)
# c = dataset_test.__getitem__(8)
# ori1=c[1][1].transpose((1, 2, 0))
# img_min2 = ori1.min()
# img_max2 = ori1.max()
# img_show2 = ((ori1 - img_min2) / (img_max2 - img_min2)) * 255
# img_show2 = img_show2.astype(np.uint8)
# img_show2=Image.fromarray(img_show2)
# img_show2.save('/media/ubuntu/e/mengjinpeng/echo_codes_really11.18/echo_codes_really/result_image/unlabel_o/imgr8.png')
