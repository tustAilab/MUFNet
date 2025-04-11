import argparse
import os
import time
import torch.utils.data
import torchvision.models.resnet

from model_flow_only import Unsupervised
# from model_share import Unsupervised
from tensorboardX import SummaryWriter
import warnings
from echo import *
# from train_data import *
from dataset_new import *
from utils_flow_only import *
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
        if self.count != 0:
            self.avg = self.sum / self.count
        else:
            self.avg = self.sum


def epoch(model, data, criterion, optimizer=None):
    model.eval() if optimizer is None else model.train()
    avg_loss = AverageMeter()
    avg_batch_time = AverageMeter()
    avg_smooth_loss = AverageMeter()
    avg_bce_loss = AverageMeter()

    # block_size = 4
    tic = time.time()
    i = 0
    # for i, (imgs, (large_Index, small_Index, large_trace, small_trace)) in enumerate(data):
    for imgs, (large_Index, small_Index, large_trace, small_trace) in data:
        # print(imgs.size())
        # print(i)
        imgs = imgs[0].to(device)
        # large_trace = large_trace.to(device)
        # small_trace = small_trace.to(device)

        with torch.set_grad_enabled(optimizer is not None):
            pred_flows, warped_imgs1 = model(imgs)
            loss, bce_loss, smooth_loss = criterion(pred_flows, warped_imgs1, imgs[:, :3, :, :])

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_time = time.time() - tic
        tic = time.time()
        avg_bce_loss.update(bce_loss.item())
        avg_smooth_loss.update(smooth_loss.item())
        avg_loss.update(loss.item())
        avg_batch_time.update(batch_time)


        i += 1

    # avg_dice_seg = dice_s / len(data) / 2.0
    # avg_seg_loss = avg_seg_loss / len(data) / 2.0
    print('\n===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\t'
          'Avg smooth_loss {smooth.avg:5.4f} \t'
          'Avg bce_loss {bce.avg:5.4f} \n'.format(
        batch_time=int(avg_batch_time.sum), loss=avg_loss,
        smooth=avg_smooth_loss, bce=avg_bce_loss))

    return avg_smooth_loss.avg, avg_bce_loss.avg, avg_loss.avg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./media/lab210/D/EchoNet-Dynamic/EchoNet-Dynamic/Videos', type=str,
                        metavar='DIR',
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

    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))
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
    batch_sampler = BatchSampler(sampler, batch_size=1, drop_last=False)
    train = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4, drop_last=False)

    tasks_2 = ["LargeIndex", "SmallIndex", "LargeTrace", "SmallTrace"]
    kwargs_2 = {"target_type": tasks_2,
                "mean": mean,
                "std": std
                }
    dataset_test = echonet.datasets.Echo(split='test', **kwargs_2)
    test2 = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=4, shuffle=False,
                                        pin_memory=(device.type == "cuda"))

    epochs = 100

    os.makedirs(os.path.join("Checkpoints/funet_flow", path), exist_ok=True)
    os.makedirs(os.path.join("model_weight/funet_flow", path), exist_ok=True)
    starting_epoch = 0
    best_loss = 5000
    # if os.path.exists(os.path.join("Checkpoints/funet_flow", path, 'training_state.pt')):
    #     checkpoint = torch.load(os.path.join("Checkpoints/funet_flow", path, 'training_state.pt'), map_location=device)
    #     mymodel.load_state_dict(checkpoint['model_state_dict'])
    #     optim.load_state_dict(checkpoint['optimizer_state_dict'])
    #     starting_epoch = checkpoint['epoch']
    #     best_loss = checkpoint['best_loss']

    for e in range(starting_epoch, epochs):
        print("=================\n=== EPOCH " + str(e + 1) + " =====\n=================\n")
        print("learning rate : ", optim.param_groups[0]["lr"])
        smooth_loss, bce_loss, total_loss = epoch(mymodel, train, loss_fnc, optim)
        lr_scheduler.step()

        torch.save({
            'epoch': e,
            'model_state_dict': mymodel.state_dict(),
            'best_loss': best_loss,
            'optimizer_state_dict': optim.state_dict(),
        }, os.path.join("Checkpoints/funet_flow", path, 'training_state.pt'))

        # smooth_loss_test, bce_loss_test, total_loss_test = epoch(mymodel, test2, loss_fnc, None)


