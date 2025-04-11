import argparse
import os
import time
import torch.utils.data
import torchvision.models.resnet

from model_unet_only import Unsupervised
# from model_share import Unsupervised
from tensorboardX import SummaryWriter
import warnings
from echo import *
# from train_data import *
# from dataset_new import *
# from utils_new import *
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
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
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
    path = os.path.join("Unsupervised", type(mymodel).__name__)
    # path = os.path.join("Unsupervised", type(mymodel.dlp).__name__)
    # loss_fnc = unsup_loss
    if args.transfer:
        best_model = torch.load(os.path.join("model_weight", type(mymodel.predictor_f).__name__, 'best_weight.pt'),
                                map_location=device)
        mymodel.predictor.load_state_dict(best_model['model_state_dict'])

    optim = torch.optim.Adam(mymodel.parameters(), args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=7460)

    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))
    tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std
              # "length": 2,
              # "period": 1,
              # "clips": "all"
              }

    train_dataset = echonet.datasets.Echo(split="train", **kwargs)
    sampler = SequentialSampler(train_dataset)
    batch_sampler = BatchSampler(sampler, batch_size= 1, drop_last=False)
    train = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4, drop_last=False)

    tasks_2 = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    kwargs_2 = {"target_type": tasks_2,
                "mean": mean,
                "std": std
                }
    dataset_test = echonet.datasets.Echo(split='test', **kwargs_2)
    test2 = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=4, shuffle=False,
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
    # if os.path.exists(os.path.join("Checkpoints", path, 'training_state.pt')):
    #     checkpoint = torch.load(os.path.join("Checkpoints", path, 'training_state.pt'), map_location=device)
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
    os.makedirs(os.path.join("Checkpoints/funet_unet", path), exist_ok=True)
    os.makedirs(os.path.join("model_weight/funet_unet", path), exist_ok=True)
    # tb = SummaryWriter(os.path.join("runs/funet_4", path), flush_secs=20)
    starting_epoch = 0
    best_loss = 5000
    if os.path.exists(os.path.join("Checkpoints/funet_unet", path, 'training_state.pt')):
        checkpoint = torch.load(os.path.join("Checkpoints/funet_unet", path, 'training_state.pt'), map_location=device)
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']

    # mile_stone = 500 // train_length
    for e in range(starting_epoch, epochs):
        print("=================\n=== EPOCH " + str(e + 1) + " =====\n=================\n")
        print("learning rate : ", optim.param_groups[0]["lr"])
        loss_train, large_inter_train, large_union_train, small_inter_train, small_union_train = \
            run_epoch(val_model, train, True, optim, device)
        lr_scheduler.step()
        overall_dice_train = 2 * (large_inter_train + small_inter_train) / (large_union_train + large_inter_train + small_union_train + small_inter_train)
        large_dice_train = 2 * large_inter_train / (large_union_train + large_inter_train)
        small_dice_train = 2 * small_inter_train / (small_union_train + small_inter_train)
        print("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format("train", *echonet.utils.bootstrap(
            np.concatenate((large_inter_train, small_inter_train)), np.concatenate((large_union_train, small_union_train)),
            echonet.utils.dice_similarity_coefficient)))
        # print("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format("test", *echonet.utils.bootstrap(large_inter_train,
        #                                                                                               large_union_train,
        #                                                                                               echonet.utils.dice_similarity_coefficient)))
        # print("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format("test", *echonet.utils.bootstrap(small_inter_train,
        #                                                                                               small_union_train,
        #                                                                                               echonet.utils.dice_similarity_coefficient)))

        torch.save({
            'epoch': e,
            'model_state_dict': mymodel.state_dict(),
            'best_loss': best_loss,
            'optimizer_state_dict': optim.state_dict(),
        }, os.path.join("Checkpoints/funet_unet", path, 'training_state.pt'))

        loss, large_inter, large_union, small_inter, small_union = run_epoch(val_model, test2, False, None, device)

        overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
        large_dice = 2 * large_inter / (large_union + large_inter)
        small_dice = 2 * small_inter / (small_union + small_inter)
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
