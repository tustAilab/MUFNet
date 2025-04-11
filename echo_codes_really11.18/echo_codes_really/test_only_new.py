import argparse
import time
import torch.utils.data
from models_flow_unet import Unsupervised
from tensorboardX import SummaryWriter
import warnings
from echo import *
# from train_data import *
from utils_new import *
from dataset_new import *
import tqdm
import scipy
from dice_loss import dice_coeff

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
            self.avg = 0

def _video_collate_fn(x):
    """Collate function for Pytorch dataloader to merge multiple videos.

    This function should be used in a dataloader for a dataset that returns
    a video as the first element, along with some (non-zero) tuple of
    targets. Then, the input x is a list of tuples:
      - x[i][0] is the i-th video in the batch
      - x[i][1] are the targets for the i-th video

    This function returns a 3-tuple:
      - The first element is the videos concatenated along the frames
        dimension. This is done so that videos of different lengths can be
        processed together (tensors cannot be "jagged", so we cannot have
        a dimension for video, and another for frames).
      - The second element is contains the targets with no modification.
      - The third element is a list of the lengths of the videos in frames.
    """
    video, target = zip(*x)  # Extract the videos and targets

    # ``video'' is a tuple of length ``batch_size''
    #   Each element has shape (channels=3, frames, height, width)
    #   height and width are expected to be the same across videos, but
    #   frames can be different.

    # ``target'' is also a tuple of length ``batch_size''
    # Each element is a tuple of the targets for the item.

    i = list(map(lambda t: t.shape[1], video))  # Extract lengths of videos in frames

    # This contatenates the videos along the the frames dimension (basically
    # playing the videos one after another). The frames dimension is then
    # moved to be first.
    # Resulting shape is (total frames, channels=3, height, width)
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))

    # Swap dimensions (approximately a transpose)
    # Before: target[i][j] is the j-th target of element i
    # After:  target[i][j] is the i-th target of element j
    target = zip(*target)

    return video, target, i

# def epoch(model, data, criterion, optimizer=None):
#     model.eval() if optimizer is None else model.train()
#     avg_loss = AverageMeter()
#     avg_batch_time = AverageMeter()
#     avg_smooth_loss = AverageMeter()
#     avg_bce_loss = AverageMeter()
#     avg_seg_loss = 0
#     avg_tc_loss = AverageMeter()
#     dice_s = 0
#     block_size = 4
#     tic = time.time()
#     i = 0
#
#     for imgs, (large_Index, small_Index, large_trace, small_trace) in data:
#         imgs = imgs[0].view(-1,6,112,112).to(device)
#         large_trace = large_trace.to(device)
#         small_trace = small_trace.to(device)
#         # print(large_Index.size())
#         sampler = SequentialSampler(imgs)
#         batch_sampler = BatchSampler(sampler, block_size, drop_last=False)
#         for img_b in batch_sampler:
#             if len(img_b) <= 1:
#                 continue
#             im_block = imgs[img_b]
#             blocks = img_b[0]
#             # for blocks in range(0, imgs.shape[0], block_size):
#             # print('''bl''',blocks)
#             # if (imgs.shape[0]-(blocks+block_size)) <=1:
#             #     continue
#             with torch.set_grad_enabled(optimizer is not None):
#                 # print(blocks)
#                 pred_flows, warped_imgs, pred_seg1, pred_seg2, warped_segs = model(im_block)
#                 # print(len(pred_flows))
#
#                 # lossB, bce_lossB, smooth_lossB, seg_lossB, tc_lossB = criterion(pred_seg1, target_seg1,warped_segs, pred_flows,
#                 #                                         warped_imgs, imgs[blocks: blocks+block_size, :3, :, :], has_seg_label=has_seg_label)
#                 loss, bce_loss, smooth_loss, seg_loss, tc_loss, dice_sm, dice_lg = criterion(pred_seg1, warped_segs,
#                                                                                              pred_flows, warped_imgs,
#                                                                                              im_block[:, :3, :, :],
#                                                                                              bl=blocks, bls=block_size,
#                                                                                              large_Index=large_Index,
#                                                                                              small_Index=small_Index,
#                                                                                              large_trace=large_trace,
#                                                                                              small_trace=small_trace)
#
#             if optimizer is not None:
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#             batch_time = time.time() - tic
#             tic = time.time()
#             avg_bce_loss.update(bce_loss.item())
#             avg_smooth_loss.update(smooth_loss.item())
#             # avg_seg_loss.update(seg_loss)
#             avg_tc_loss.update(tc_loss.item())
#             avg_loss.update(loss.item())
#             # avg_dice_seg.update(dice_seg)
#             dice_s += (dice_sm + dice_lg)
#             avg_seg_loss += seg_loss
#             avg_batch_time.update(batch_time)
#
#             if i % PRINT_INTERVAL == 0:
#                 print('[{0:s} Batch {1:03d}/{2:03d}]\t'
#                       'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'smooth_loss {smooth.val:5.4f} ({smooth.avg:5.4f})\t'
#                       'bce_loss {bce.val:5.4f} ({bce.avg:5.4f})\t'
#                       'seg_loss {seg:5.4f} \t'
#                       'tc_loss {tc.val:5.4f} ({tc.avg:5.4f}) \t'
#                       'dice_sm ({dice_sm:5.4f}) \t'
#                       'dice_lg ({dice_lg:5.4f})'
#                     .format(
#                     "EVAL" if optimizer is None else "TRAIN", i, len(data), batch_time=avg_batch_time,
#                     loss=avg_loss,
#                     smooth=avg_smooth_loss, bce=avg_bce_loss, seg=seg_loss, tc=avg_tc_loss, dice_sm=dice_sm,
#                     dice_lg=dice_lg))
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
#         smooth=avg_smooth_loss, bce=avg_bce_loss, seg=avg_seg_loss, tc=avg_tc_loss, dice=avg_dice_seg))
#
#     return avg_smooth_loss.avg, avg_bce_loss.avg, avg_seg_loss, avg_tc_loss.avg, avg_loss.avg, avg_dice_seg, pred_seg1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./media/lab210/D/EchoNet-Dynamic/EchoNet-Dynamic/Videos', type=str, metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--model', default='flownet', type=str, help='the supervised model to be trained with ('
                                                                     'flownet, lightflownet, pwc_net)')
    # parser.add_argument('--steps', default=7460, type=int, metavar='N', help='number of total steps to run')
    parser.add_argument('--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--lr', default=1.6e-5, type=float, metavar='LR', help='learning rate')
    parser.add_argument("--augment", help="perform data augmentation", action="store_true")
    parser.add_argument("--transfer", help="perform transfer learning from an already trained supervised model",
                        action="store_true")

    args = parser.parse_args()

    mymodel = Unsupervised(conv_predictor=args.model)
    mymodel.to(device)
    path = os.path.join("Unsupervised", type(mymodel.predictor_f).__name__)
    loss_fnc = unsup_loss
    if args.transfer:
        best_model = torch.load(os.path.join("model_weight", type(mymodel.predictor_f).__name__, 'best_weight.pt'),
                                map_location=device)
        mymodel.predictor.load_state_dict(best_model['model_state_dict'])

    optim = torch.optim.Adam(mymodel.parameters(), args.lr)

    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="test"))
    # tasks = ["LargeIndex", "SmallIndex", "LargeTrace", "SmallTrace"]
    # kwargs_2 = {"target_type": tasks,
    #             "mean": mean,
    #             "std": std,
    #             "length": 2,
    #             "period": 1,
    #             "clips": "all"
    #             }
    # test_dataset = Echo(split='test',**kwargs_2)
    # test = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=1, num_workers=4, shuffle=True, pin_memory=(device.type == "cuda"))
    # # print(len(test))
    #
    # epochs = 1
    # tb_frames_test = next(iter(test))[0][0:1].to(device)
    # tb_frames_test = tb_frames_test.view(-1, 6, 112, 112)


    os.makedirs(os.path.join("Checkpoints/funet_6", path), exist_ok=True)
    os.makedirs(os.path.join("model_weight/funet_6", path), exist_ok=True)
    tb = SummaryWriter(os.path.join("runs/funet_6", path), flush_secs=20)
    starting_epoch = 0
    best_loss = 100000
    if os.path.exists(os.path.join("Checkpoints/funet_6", path, 'training_state.pt')):
        checkpoint = torch.load(os.path.join("Checkpoints/funet_6", path, 'training_state.pt'), map_location=device)
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']


    # smooth_loss_test, bce_loss_test, seg_loss_test, tc_loss_test, total_loss_test, dice_score_test, prediction= epoch(mymodel, test, loss_fnc)
    # dice_s = 0
    # dice_s = dice_s + dice_coeff(pred_seg, large_trace).item()

    # Saving videos with segmentations
    dataset = echonet.datasets.Echo(split="test",
                                        target_type=["Filename", "LargeIndex", "SmallIndex"],
                                        # Need filename for saving, and human-selected frames to annotate
                                        mean=mean, std=std,  # Normalization
                                        length=None, max_length=None, period=1  # Take all frames
                                        )
    test = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=4, shuffle=False,
                                                 pin_memory=False, collate_fn=_video_collate_fn)
    # Save videos with segmentation
    output = os.path.join("Checkpoints/funet_6", path)
    block_size = 4
    if not all(os.path.isfile(os.path.join(output, "videos", f)) for f in test.dataset.fnames):
        # Only run if missing videos

        # mymodel.eval()

        os.makedirs(os.path.join(output, "videos"), exist_ok=True)
        os.makedirs(os.path.join(output, "size"), exist_ok=True)
        echonet.utils.latexify()

        with torch.no_grad():
            with open(os.path.join(output, "size_all.csv"), "w") as g:
                g.write("Filename,Frame,Size,HumanLarge,HumanSmall,ComputerSmall, ComputerLarge\n")
                # for x, (filenames, large_index, small_index) in test:
                for (x, (filenames, large_index, small_index), length) in tqdm.tqdm(test):
                    # Run segmentation model on blocks of frames one-by-one
                    # The whole concatenated video may be too long to run together
                    # x = x[0].view(-1, 3, 112, 112)
                    # print(x.shape, length)
                    # yy = mymodel.predictor_s(x[0:(0 + block_size), :, :, :].to(device))
                    # print(yy.shape)
                    y = np.concatenate([mymodel.predictor_s(x[i:(i + block_size), :, :, :].to(device)).detach().cpu().numpy() for i in range(0, x.shape[0], block_size)])

                    start = 0
                    x = x.numpy()
                    for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                        # Extract one video and segmentation predictions
                        video = x[start:(start + offset), :, :, :]
                        logit = y[start:(start + offset), 0, :, :]

                        # Un-normalize video
                        video *= std.reshape(1, 3, 1, 1)
                        video += mean.reshape(1, 3, 1, 1)

                        # Get frames, channels, height, and width
                        f, c, h, w = video.shape  # pylint: disable=W0612
                        assert c == 3

                        # Put two copies of the video side by side
                        video = np.concatenate((video, video), 3)

                        # If a pixel is in the segmentation, saturate blue channel
                        # Leave alone otherwise
                        video[:, 0, :, w:] = np.maximum(255. * (logit > 0), video[:, 0, :, w:])  # pylint: disable=E1111

                        # Add blank canvas under pair of videos
                        video = np.concatenate((video, np.zeros_like(video)), 2)

                        # Compute size of segmentation per frame
                        size = (logit > 0).sum((1, 2))

                        # Identify systole frames with peak detection
                        trim_min = sorted(size)[round(len(size) ** 0.05)-1]
                        trim_max = sorted(size)[round(len(size) ** 0.95)-1]
                        trim_range = trim_max - trim_min
                        systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])
                        disystole = set(scipy.signal.find_peaks(size, distance=20, prominence=(0.50 * trim_range))[0])

                        # Write sizes and frames to file
                        for (frame, s) in enumerate(size):
                            g.write("{},{},{},{},{},{},{}\n".format(filename, frame, s, 1 if frame == large_index[i] else 0, 1 if frame == small_index[i] else 0, 1 if frame in systole else 0, 1 if frame in disystole else 0))

                        # Plot sizes
                        fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                        plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                        ylim = plt.ylim()
                        for s in systole:
                            plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
                        plt.ylim(ylim)
                        plt.title(os.path.splitext(filename)[0])
                        plt.xlabel("Seconds")
                        plt.ylabel("Size (pixels)")
                        plt.tight_layout()
                        plt.savefig(os.path.join(output, "size", os.path.splitext(filename)[0] + ".pdf"))
                        plt.close(fig)

                        # Normalize size to [0, 1]
                        size -= size.min()
                        size = size / size.max()
                        size = 1 - size

                        # Iterate the frames in this video
                        for (f, s) in enumerate(size):

                            # On all frames, mark a pixel for the size of the frame
                            video[:, :, int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))] = 255.

                            if f in systole:
                                # If frame is computer-selected systole, mark with a line
                                video[:, :, 115:224, int(round(f / len(size) * 200 + 10))] = 255.

                            def dash(start, stop, on=10, off=10):
                                buf = []
                                x = start
                                while x < stop:
                                    buf.extend(range(x, x + on))
                                    x += on
                                    x += off
                                buf = np.array(buf)
                                buf = buf[buf < stop]
                                return buf
                            d = dash(115, 224)

                            if f == large_index[i]:
                                # If frame is human-selected diastole, mark with green dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 225, 0]).reshape((1, 3, 1))
                            if f == small_index[i]:
                                # If frame is human-selected systole, mark with red dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 0, 225]).reshape((1, 3, 1))

                            # Get pixels for a circle centered on the pixel
                            r, c = skimage.draw.circle(int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10)), 4.1)

                            # On the frame that's being shown, put a circle over the pixel
                            video[f, :, r, c] = 255.

                        # Rearrange dimensions and save
                        video = video.transpose([1, 0, 2, 3])
                        video = video.astype(np.uint8)
                        echonet.utils.savevideo(os.path.join(output, "videos", filename), video, 50)

                        # Move to next video
                        start += offset

    tb.close()
