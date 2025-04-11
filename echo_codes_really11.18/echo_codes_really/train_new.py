import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torchvision.models.resnet

from models_flow_unet import Unsupervised
# from model_share import Unsupervised
from tensorboardX import SummaryWriter
import warnings
from echo import *
# from train_data import *
from dataset_new import *
from utils_new import *
from torch.utils.data import BatchSampler, SequentialSampler, DataLoader
import tqdm
import math
warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.seed(seed=1)

PRINT_INTERVAL = 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class AverageMeter(object):

    def __init__(self, keep_all=False):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.data = None
        if keep_all:
            self.data = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        if self.data is not None:
            self.data.append(value)
        self.val = value
        self.sum += value * n
        if value != 0:
            self.count += n
        if self.count !=0:
            self.avg = self.sum / self.count
        else:
            self.avg = self.sum


def epoch(model, data, criterion, optimizer=None):
    model.eval() if optimizer is None else model.train()
    avg_loss = AverageMeter()
    avg_batch_time = AverageMeter()
    avg_smooth_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    avg_seg_loss = 0
    avg_tc_loss = AverageMeter()
    dice_s = 0
    # block_size = 4
    tic = time.time()
    i = 0
    # for i, (imgs, (large_Index, small_Index, large_trace, small_trace)) in enumerate(data):
    for imgs, (large_Index, small_Index, large_trace,small_trace) in data:
        # print(imgs.size())
        # print(i)
        imgs = imgs[0].to(device)
        large_trace = large_trace.to(device)
        small_trace = small_trace.to(device)

        with torch.set_grad_enabled(optimizer is not None):
            pred_flows, warped_imgs1, pred_seg1, pred_seg2, warped_segs1 = model(imgs)
            loss, bce_loss, smooth_loss, seg_loss, tc_loss, dice_sm, dice_lg= criterion(pred_seg1,
                                                                                        warped_segs1,
                                                                                        pred_flows, warped_imgs1,
                                                                                        imgs[:,:3, :, :],
                                                                                        pred_seg2,
                                                                                        large_Index=large_Index,
                                                                                        small_Index=small_Index,
                                                                                        large_trace=large_trace,
                                                                                        small_trace=small_trace,
                                                                                        )

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_time = time.time() - tic
        tic = time.time()
        avg_bce_loss.update(bce_loss.item())
        avg_smooth_loss.update(smooth_loss.item())
        # avg_seg_loss.update(seg_loss)
        avg_tc_loss.update(tc_loss.item())
        avg_loss.update(loss.item())
        # avg_dice_seg.update(dice_seg)
        dice_s += (dice_sm + dice_lg)
        avg_seg_loss += seg_loss
        avg_batch_time.update(batch_time)

        # if i % PRINT_INTERVAL == 0:
        #     print('[{0:s} Batch {1:03d}/{2:03d}]\t'
        #           'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'smooth_loss {smooth.val:5.4f} ({smooth.avg:5.4f})\t'
        #           'bce_loss {bce.val:5.4f} ({bce.avg:5.4f})\t'
        #           'seg_loss {seg:5.4f} \t'
        #           'tc_loss {tc.val:5.4f} ({tc.avg:5.4f}) \t'
        #           'dice_sm ({dice_sm:5.4f}) \t'
        #           'dice_lg ({dice_lg:5.4f})'
        #         .format(
        #         "EVAL" if optimizer is None else "TRAIN", i, len(data), batch_time=avg_batch_time, loss=avg_loss,
        #         smooth=avg_smooth_loss,bce=avg_bce_loss,seg = seg_loss,tc = avg_tc_loss, dice_sm=dice_sm, dice_lg=dice_lg))
        # i += 1
        # if i%1000==0:
        #     seg11 = pred_seg2.cpu().detach().numpy()
        #     a, b, c, d = np.split(seg11, 4, axis=0)
        #     a = a.squeeze(0)
        #     a = a.squeeze(0)
        #     plt.subplot(1,2,1)
        #     plt.imshow(a, cmap='gray')
        #     plt.show()
    avg_dice_seg = dice_s / len(data) / 2.0
    avg_seg_loss = avg_seg_loss / len(data) / 2.0
    print('\n===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\t'
          'Avg smooth_loss {smooth.avg:5.4f} \t'
          'Avg bce_loss {bce.avg:5.4f} \t'
          'seg_loss {seg:5.4f} \t'
          'tc_loss {tc.avg:5.4f} \t'
          'dice {dice:5.4f} \n'.format(
        batch_time=int(avg_batch_time.sum), loss=avg_loss,
        smooth=avg_smooth_loss, bce=avg_bce_loss, seg = avg_seg_loss, tc = avg_tc_loss,dice=1 -avg_dice_seg))

    return avg_smooth_loss.avg, avg_bce_loss.avg, avg_seg_loss, avg_tc_loss.avg, avg_loss.avg, avg_dice_seg

# def epoch(model, data, criterion, optimizer=None):
#     model.eval() if optimizer is None else model.train()
#     avg_loss = AverageMeter()
#     avg_batch_time = AverageMeter()
#     avg_smooth_loss = AverageMeter()
#     avg_bce_loss = AverageMeter()
#     avg_seg_loss = 0
#     avg_tc_loss = AverageMeter()
#     dice_s = 0
#     block_size = 2
#     tic = time.time()
#     i = 0
#     # for i, (imgs, (large_Index, small_Index, large_trace, small_trace)) in enumerate(data):
#     for imgs, (large_Index, small_Index, large_trace, small_trace) in data:
#         # print(imgs.size())
#         # print(large_Index,//// small_Index, large_trace, small_trace)
#         imgs = imgs[0].to(device)
#         large_trace = large_trace.to(device)
#         small_trace = small_trace.to(device)
#         # print('''li''',imgs.shape[0])
#         # sampler = SequentialSampler(imgs)
#         # batch_sampler = BatchSampler(sampler, block_size, drop_last=False)
#         # for img_b in batch_sampler:
#         #     if len(img_b)<=1:
#         #         continue
#         #     im_block = imgs[img_b]
#         #     blocks = img_b[0]
#         # for blocks in range(0, imgs.shape[0], block_size):
#             # print('''bl''',blocks)
#             # if (imgs.shape[0]-(blocks+block_size)) <=1:
#             #     continue
#         with torch.set_grad_enabled(optimizer is not None):
#             # print(blocks)
#             pred_flows, warped_imgs1, pred_seg1, pred_seg2, warped_segs1 = model(imgs)
#             # print('''len''', len(pred_flows), len(warped_imgs), len(pred_seg1))
#
#             # lossB, bce_lossB, smooth_lossB, seg_lossB, tc_lossB = criterion(pred_seg1, target_seg1,warped_segs, pred_flows,
#             #                                         warped_imgs, imgs[blocks: blocks+block_size, :3, :, :], has_seg_label=has_seg_label)
#             loss, bce_loss, smooth_loss, seg_loss, tc_loss, dice_sm, dice_lg= criterion(pred_seg1,
#                                                                                         warped_segs1,
#                                                                                         pred_flows, warped_imgs1,
#                                                                                         imgs[:, :3, :, :],
#                                                                                         # bl=blocks, bls=block_size,
#                                                                                         large_Index=large_Index,
#                                                                                         small_Index=small_Index,
#                                                                                         large_trace=large_trace,
#                                                                                         small_trace=small_trace)
#
#         if optimizer is not None:
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         batch_time = time.time() - tic
#         tic = time.time()
#         avg_bce_loss.update(bce_loss.item())
#         avg_smooth_loss.update(smooth_loss.item())
#         # avg_seg_loss.update(seg_loss)
#         avg_tc_loss.update(tc_loss.item())
#         avg_loss.update(loss.item())
#         # avg_dice_seg.update(dice_seg)
#         dice_s += (dice_sm + dice_lg)
#         avg_seg_loss += seg_loss
#         avg_batch_time.update(batch_time)
#
#             # if i % PRINT_INTERVAL == 0:
#             #     print('[{0:s} Batch {1:03d}/{2:03d}]\t'
#             #           'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
#             #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#             #           'smooth_loss {smooth.val:5.4f} ({smooth.avg:5.4f})\t'
#             #           'bce_loss {bce.val:5.4f} ({bce.avg:5.4f})\t'
#             #           'seg_loss {seg:5.4f} \t'
#             #           'tc_loss {tc.val:5.4f} ({tc.avg:5.4f}) \t'
#             #           'dice_sm ({dice_sm:5.4f}) \t'
#             #           'dice_lg ({dice_lg:5.4f})'
#             #         .format(
#             #         "EVAL" if optimizer is None else "TRAIN", i, len(data), batch_time=avg_batch_time, loss=avg_loss,
#             #         smooth=avg_smooth_loss,bce=avg_bce_loss,seg = seg_loss,tc = avg_tc_loss, dice_sm=dice_sm, dice_lg=dice_lg))
#         i += 1
#
#     avg_dice_seg = dice_s / len(data) / 2.0
#     avg_seg_loss = avg_seg_loss / len(data) / 2.0
#     print('\n===============> Total time {batch_time:d}s\t'
#           'Avg loss {loss.avg:.4f}\t'
#           'Avg smooth_loss {smooth.avg:5.4f} \t'
#           'Avg bce_loss {bce.avg:5.4f} \t'
#           'seg_loss {seg:5.4f} \t'
#           'tc_loss {tc.avg:5.4f} \t'
#           'dice {dice:5.4f} \n'.format(
#         batch_time=int(avg_batch_time.sum), loss=avg_loss,
#         smooth=avg_smooth_loss, bce=avg_bce_loss, seg = avg_seg_loss, tc = avg_tc_loss,dice=1 -avg_dice_seg))
#
#     return avg_smooth_loss.avg, avg_bce_loss.avg, avg_seg_loss, avg_tc_loss.avg, avg_loss.avg, avg_dice_seg

def run_epoch(model, dataloader, train, optim, device):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
    """

    total = 0.
    n = 0

    # pos = 0
    # neg = 0
    # pos_pix = 0
    # neg_pix = 0

    model.eval()

    large_inter = 0
    large_union = 0
    small_inter = 0
    small_union = 0
    large_inter_list = []
    large_union_list = []
    small_inter_list = []
    small_union_list = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (_, (large_frame, small_frame, large_trace, small_trace)) in dataloader:
                # Count number of pixels in/out of human segmentation
                # pos += (large_trace == 1).sum().item()
                # pos += (small_trace == 1).sum().item()
                # neg += (large_trace == 0).sum().item()
                # neg += (small_trace == 0).sum().item()

                # Count number of pixels in/out of computer segmentation
                # pos_pix += (large_trace == 1).sum(0).to("cpu").detach().numpy()
                # pos_pix += (small_trace == 1).sum(0).to("cpu").detach().numpy()
                # neg_pix += (large_trace == 0).sum(0).to("cpu").detach().numpy()
                # neg_pix += (small_trace == 0).sum(0).to("cpu").detach().numpy()

                # Run prediction for diastolic frames and compute loss
                large_frame = large_frame.to(device)
                large_trace = large_trace.to(device)
                y_large = model(large_frame)
                loss_large = torch.nn.functional.binary_cross_entropy_with_logits(y_large[:, 0, :, :], large_trace, reduction="sum")
                # Compute pixel intersection and union between human and computer segmentations
                large_inter += np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_union += np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_inter_list.extend(np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                large_union_list.extend(np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                # Run prediction for systolic frames and compute loss
                small_frame = small_frame.to(device)
                small_trace = small_trace.to(device)
                y_small = model(small_frame)
                loss_small = torch.nn.functional.binary_cross_entropy_with_logits(y_small[:, 0, :, :], small_trace, reduction="sum")
                # Compute pixel intersection and union between human and computer segmentations
                small_inter += np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_union += np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_inter_list.extend(np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                small_union_list.extend(np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                # # Take gradient step if training
                loss = (loss_large + loss_small) / 2
                # if train:
                #     optim.zero_grad()
                #     loss.backward()
                #     optim.step()
                #
                # # Accumulate losses and compute baselines
                total += loss.item()
                n += large_trace.size(0)
                # p = pos / (pos + neg)
                # p_pix = (pos_pix + 1) / (pos_pix + neg_pix + 2)
                #
                # # Show info on process bar
                # pbar.set_postfix_str("{:.4f} ({:.4f}) / {:.4f} {:.4f}, {:.4f}, {:.4f}".format(total / n / 112 / 112, loss.item() / large_trace.size(0) / 112 / 112, -p * math.log(p) - (1 - p) * math.log(1 - p), (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean(), 2 * large_inter / (large_union + large_inter), 2 * small_inter / (small_union + small_inter)))
                # pbar.update()

    large_inter_list = np.array(large_inter_list)
    large_union_list = np.array(large_union_list)
    small_inter_list = np.array(small_inter_list)
    small_union_list = np.array(small_union_list)

    return (total / n / 112 / 112,
            large_inter_list,
            large_union_list,
            small_inter_list,
            small_union_list,
            )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./media/lab210/D/EchoNet-Dynamic/EchoNet-Dynamic/Videos', type=str, metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--model', default='flownet', type=str, help='the supervised model to be trained with ('
                                                                     'flownet, lightflownet, pwc_net)')
    parser.add_argument('--steps', default=746000, type=int, metavar='N', help='number of total steps to run')
    parser.add_argument('--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--lr', default=1.6e-5, type=float, metavar='LR', help='learning rate')
    parser.add_argument("--augment", help="perform data augmentation", action="store_true")
    parser.add_argument("--transfer", help="perform transfer learning from an already trained supervised model",
                        action="store_true")

    args = parser.parse_args()
    mymodel = Unsupervised(conv_predictor=args.model)
    mymodel.to(device)
    path = os.path.join("Unsupervised", type(mymodel.predictor_f).__name__)
    # path = os.path.join("Unsupervised", type(mymodel.dlp).__name__)
    loss_fnc = unsup_loss
    if args.transfer:
        best_model = torch.load(os.path.join("model_weight", type(mymodel.predictor_f).__name__, 'best_weight.pt'),
                                map_location=device)
        mymodel.predictor.load_state_dict(best_model['model_state_dict'])

    optim = torch.optim.Adam(mymodel.parameters(), args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=7460)

    mean, std =0,1 #echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))
    tasks = ["LargeIndex", "SmallIndex", "LargeTrace", "SmallTrace"]
    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std
              # "length": 2,
              # "period": 1,
              # "clips": "all"
              }

    train_dataset = Echo_2(split="train", **kwargs)
    sampler = SequentialSampler(train_dataset)
    batch_sampler = BatchSampler(sampler, batch_size= 1, drop_last=False)
    train = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4, drop_last=False)

    tasks_2 = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    kwargs_2 = {"target_type": tasks_2,
                "mean": mean,
                "std": std
                }
    dataset_test = echonet.datasets.Echo(split='test', **kwargs_2)
    test2 = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=2, shuffle=False,
                                        pin_memory=(device.type == "cuda"))
    # tasks_2 = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    # kwargs_2 = {"target_type": tasks_2,
    #             "mean": mean,
    #             "std": std
    #             }
    # val_dataset = echonet.datasets.Echo(split='val',**kwargs_2)
    # val = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=(device.type == "cuda"))
    val_model = mymodel.predictor_s

    #
    # train_length = len(train)
    epochs = 100
    # print(epochs)
    #
    # tb_frames_train = next(iter(train))[0][0:1].to(device)
    # tb_frames_train = tb_frames_train.view(-1, 6, 112, 112)
    # tb_frames_val = next(iter(val))[0][0:1].to(device)
    # tb_frames_val = tb_frames_val.view(-1, 6, 112, 112)
    # tb_frames_test = next(iter(test))[0][0:1].to(device)
    # tb_frames_test = tb_frames_test.view(-1, 6, 112, 112)

    # tb_frames_train = train
    # tb_frames_val = val
    # tb_frames_test = test

    # os.makedirs(os.path.join("Checkpoints", path), exist_ok=True)
    # os.makedirs(os.path.join("model_weight", path), exist_ok=True)
    # tb = SummaryWriter(os.path.join("runs", path), flush_secs=20)
    # starting_epoch = 0
    # best_loss = 5000
    # if os.path.exists('/media/ubuntu/e/mengjinpeng/echo_codes_really11.18/echo_codes_really/model_weight/training_state.pt'):
    #     checkpoint = torch.load('/media/ubuntu/e/mengjinpeng/echo_codes_really11.18/echo_codes_really/model_weight/training_state.pt', map_location=device)
    #     mymodel.load_state_dict(checkpoint['model_state_dict'])
    #     optim.load_state_dict(checkpoint['optimizer_state_dict'])
    #     starting_epoch = checkpoint['epoch']
    #     best_loss = checkpoint['best_loss']
    #
    # mile_stone = 500 // train_length
    # for e in range(starting_epoch, epochs):
    #
    #     print("=================\n=== EPOCH " + str(e + 1) + " =====\n=================\n")
    #     print("learning rate : ", optim.param_groups[0]["lr"])
    #     smooth_loss, bce_loss, seg_loss, tc_loss, total_loss, dice_score = epoch(mymodel, train, loss_fnc, optim)
    #
    #     torch.save({
    #         'epoch': e,
    #         'model_state_dict': mymodel.state_dict(),
    #         'best_loss': best_loss,
    #         'optimizer_state_dict': optim.state_dict(),
    #     }, os.path.join("Checkpoints", path, 'training_state.pt'))
    os.makedirs(os.path.join("Checkpoints/fivefole1", path), exist_ok=True)
    os.makedirs(os.path.join("model_weight/fivefole1", path), exist_ok=True)
    # tb = SummaryWriter(os.path.join("runs/funet_4", path), flush_secs=20)
    starting_epoch = 0
    best_loss = 5000
    # if os.path.exists(os.path.join("Checkpoints/funet_6", path, 'training_state.pt')):
    #     checkpoint = torch.load(os.path.join("Checkpoints/funet_6", path, 'training_state.pt'), map_location=device)
    #     mymodel.load_state_dict(checkpoint['model_state_dict'])
    #     optim.load_state_dict(checkpoint['optimizer_state_dict'])
    #     starting_epoch = checkpoint['epoch']
    #     best_loss = checkpoint['best_loss']
    com = 0
    e1=0
    # mile_stone = 500 // train_length
    for e in range(starting_epoch, epochs):
        print("=================\n=== EPOCH " + str(e + 1) + " =====\n=================\n")
        print("learning rate : ", optim.param_groups[0]["lr"])
        smooth_loss, bce_loss, seg_loss, tc_loss, total_loss, dice_score = epoch(mymodel, train, loss_fnc, optim)
        lr_scheduler.step()

        torch.save({
            'epoch': e,
            'model_state_dict': mymodel.state_dict(),
            'best_loss': best_loss,
            'optimizer_state_dict': optim.state_dict(),
        }, os.path.join("Checkpoints/funet_try", path, 'training_state.pt'))

        loss, large_inter, large_union, small_inter, small_union = run_epoch(val_model, test2, False, None, device)

        overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
        #echonet.utils.bootstrap(np.concatenate((large_inter, small_inter)))
        dice,big,small=echonet.utils.bootstrap(np.concatenate((large_inter, small_inter)), np.concatenate((large_union, small_union)),echonet.utils.dice_similarity_coefficient)
        if float(dice)>float(com):
            torch.save({
                'epoch': e,
                'model_state_dict': mymodel.state_dict(),
                'best_loss': best_loss,
                'optimizer_state_dict': optim.state_dict(),
            }, os.path.join("Checkpoints/funet_try", path, 'best92.79.pt'))
            com = dice
            e1=e
        else:
            pass
        print('目前最好的模型是第', e1, '轮', com)
        large_dice = 2 * large_inter / (large_union + large_inter)
        small_dice = 2 * small_inter / (small_union + small_inter)
        with open(r'result/result_best-fivefold-2.txt','a') as f:
            f.write("=================\n=== EPOCH " + str(e + 1) + " =====\n=================\n")
            f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format("test", *echonet.utils.bootstrap(
                np.concatenate((large_inter, small_inter)), np.concatenate((large_union, small_union)),
                echonet.utils.dice_similarity_coefficient)))
            f.write("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format("test", *echonet.utils.bootstrap(large_inter,
                                                                                                      large_union,
                                                                        echonet.utils.dice_similarity_coefficient)))
            f.write("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format("test", *echonet.utils.bootstrap(small_inter,
                                                                                                      small_union,
                                                                        echonet.utils.dice_similarity_coefficient)))
        print("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format("test", *echonet.utils.bootstrap(
                np.concatenate((large_inter, small_inter)), np.concatenate((large_union, small_union)),
                echonet.utils.dice_similarity_coefficient)))
        print("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format("test", *echonet.utils.bootstrap(large_inter,
                                                                                                      large_union,
                                                                        echonet.utils.dice_similarity_coefficient)))
        print("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format("test", *echonet.utils.bootstrap(small_inter,
                                                                                                      small_union,
                                                                        echonet.utils.dice_similarity_coefficient)))

    # tb.close()
