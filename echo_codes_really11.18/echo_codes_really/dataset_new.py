"""EchoNet-Dynamic Dataset."""
import csv
import pathlib
import os
import collections

import numpy as np
import skimage.draw
import torch.utils.data
import sys

from matplotlib import pyplot as plt

# sys.path.append("/media/lab210/D/EchoNet-Dynamic/dynamic/")
import echonet
from torch.utils.data import BatchSampler, SequentialSampler
from skimage.segmentation import find_boundaries
from tensorboardX import SummaryWriter

def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)

class Echo_2(torch.utils.data.Dataset):
    """EchoNet-Dynamic Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {"train", "val", "test", "external_test"}
        target_type (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """

    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 # length=16, period=2,
                 # max_length=250,
                 # clips=1,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 external_test_location=None):

        if root is None:
            root = r'/media/ubuntu/e/mengjinpeng/echonet-dynamic'


        self.folder = pathlib.Path(root.replace('\\','/'))
        self.split = [split]
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        # self.length = length
        # self.max_length = max_length
        # self.period = period
        # self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        if split == "external_test":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            with open(self.folder / "FileList.csv") as f:
                # print('''hhhh''')
                self.header = f.readline().strip().split(",")
                filenameIndex = self.header.index("FileName")
                # print(filenameIndex)
                splitIndex = self.header.index("Split")
                # print(splitIndex)
                for line in f:
                    lineSplit = line.strip().split(',')
                    "changed error in line 118"
                    fileName = lineSplit[filenameIndex] #+ ".avi"
                    fileMode = lineSplit[splitIndex].lower()

                    if split in ["all", fileMode] and os.path.exists(self.folder / "Videos" / fileName):
                        self.fnames.append(fileName)
                        self.outcome.append(lineSplit)

            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            with open(self.folder / "VolumeTracings.csv") as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    "changed error in line 135"
                    filename = filename.split('.')[0]
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            keep = [len(self.frames[os.path.splitext(f)[0]]) >= 2 for f in self.fnames]
            # keep = keep[0:10]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]
    #            print(self.outcome)

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "external_test":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "clinical_test":
            video = os.path.join(self.folder, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.folder, "Videos", self.fnames[index])

        # Load video into np.array
        video = echonet.utils.loadvideo(video).astype(np.float32)
        # print(video.shape)

        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        # Set number of frames
        c, f, h, w = video.shape
        # print(c, f, h, w)


        # Gather targets
        target = []
        for t in self.target_type:
            # print(t)
            key = os.path.splitext(self.fnames[index])[0][0:1000]
            # print(key)
            if t == "Filename":
                # print('hhhh')
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                # print('hhhh')
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is last
                target.append(int(self.frames[key][-1]))
            elif t == "SmallIndex":
                # Largest (diastolic) frame is first
                target.append(int(self.frames[key][0]))
            elif t == "LargeFrame":
                target.append(video[:, self.frames[key][-1], :, :])
            elif t == "SmallFrame":
                target.append(video[:, self.frames[key][0], :, :])
            elif t in ["LargeTrace", "SmallTrace"]:

                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(np.rint(y).astype(int), np.rint(x).astype(int),
                                            (video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1
                #drawing line
                # mask = find_boundaries(mask).astype(np.float32)
                # Length_of_side=30
                # x1_min=np.min(x1) #left_minx
                # y1_max=np.max(y1) #left_maxy
                # x2_max=np.max(x2) #right_maxx
                # y2_max=np.max(y2) #right_maxy
                # center_left=(x1_min,y1_max)
                # center_right=(x2_max,y2_max)
                #
                # #Lower left corner
                # top_left = (center_left[0] - Length_of_side / 2, center_left[1] - Length_of_side / 2)
                # top_right = (center_left[0] + Length_of_side / 2, center_left[1] - Length_of_side / 2)
                # bottom_right = (center_left[0] + Length_of_side / 2, center_left[1] + Length_of_side / 2)
                # bottom_left = (center_left[0] - Length_of_side / 2, center_left[1] + Length_of_side / 2)
                # #Lower right corner
                # top_left2 = (center_right[0] - Length_of_side / 2, center_right[1] - Length_of_side / 2)
                # top_right2 = (center_right[0] + Length_of_side / 2, center_right[1] - Length_of_side / 2)
                # bottom_right2 = (center_right[0] + Length_of_side / 2, center_right[1] + Length_of_side / 2)
                # bottom_left2 = (center_right[0] - Length_of_side / 2, center_right[1] + Length_of_side / 2)
                #
                # d, j, k, l = int(top_right[0]), int(top_left[0]), int(top_left[1]), int(bottom_left[1])  #
                # d1, j1, k1, l1 = int(top_right2[0]), int(top_left2[0]), int(top_left2[1]), int(bottom_left2[1])
                #
                # bool_mask = mask.copy()
                # bool_mask[k:l, j:d] = False
                # bool_mask[k1:l1, j1:d1] = False
                # small_mask = np.where(bool_mask, 0, mask)

                target.append(mask)
                #target.append(small_mask)
            else:
                # print('hhhh')
                if self.split == "clinical_test" or self.split == "external_test":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))
        # print(key)
        larId = target[0]
        smaId = target[1]
        p=larId-smaId
        if smaId + p >= f:
            video1 = video[:, smaId - 2*p + np.arange(p), :, :]
            video2 = video[:, smaId - p + np.arange(p), :, :]
            video3 = video[:, larId - p + np.arange(p), :, :]
            video4 = video[:, larId + np.arange(p), :, :]
            target[1] = 1
            target[0] = 3
        elif larId + p >= f:
            video1 = video[:, smaId - p + np.arange(p), :, :]
            video2 = video[:, smaId + np.arange(p), :, :]
            video3 = video[:, larId - 2*p + np.arange(p), :, :]
            video4 = video[:, larId - p + np.arange(p), :, :]
            target[1] = 0
            target[0] = 3

        else:
            video1 = video[:, smaId-p + np.arange(p),:,:]
            video2 = video[:, smaId + np.arange(p),:,:]
            video3 = video[:, larId-p +np.arange(p),:,:]
            video4 = video[:, larId + np.arange(p),:,:]
            target[1] = 0
            target[0] = 2

        video = tuple((video1, video2,video3,video4))

        video = np.stack(video).reshape((-1,-1,112,112))
        # print('''gg''',video.shape)
        # print(len(target[0][2]))
        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)



        if self.pad is not None:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video  # pylint: disable=E1130
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i:(i + h), j:(j + w)]
        return video, target

    def __len__(self):
        return len(self.fnames)
#
if __name__ == '__main__':

    #mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))
    # print(mean)
    tasks = ["LargeIndex", "SmallIndex", "LargeTrace", "SmallTrace"]
    kwargs = {"target_type": tasks,
              #"mean": mean,
              #"std": std
              }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = Echo_2(split="train", **kwargs)
    b=train_dataset.__getitem__(1)
    print('wan')
# #
# #
#     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=False)
#     for imgs, (large_Index, small_Index, large_trace, large_smaller_trace,small_trace,smaller_trace,) in train_dataloader:
#          print('wancheng')


#     # print(train_dataset.)
#     # print(len(train_dataset))
#     i=0
#     for imgs, (large_Index, small_Index, large_trace, small_trace) in train_dataloader:
#         print(i)
#         i+=1
#         imgs = imgs[0].view(-1, 6, 112, 112).to(device)
#         print(imgs.shape)
#         sampler = SequentialSampler(imgs)
#         batch_sample = BatchSampler(sampler, 8, drop_last=False)
#         for bl in batch_sample:
#             if len(bl)<=1:
#                 continue
#             print(bl)
#     tb = SummaryWriter('/media/lab210/D/EchoNet-Dynamic/dynamic/runs/Unsupervised/FlowNetS', flush_secs=20)
#     tb.add_images('train_s', large_trace, dataformats='CHW')
# imgs = imgs[0].view(-1, 6, 112, 112).to(device)
# for blocks in range(0, imgs.shape[0], 20):
#     im = imgs[blocks:blocks+20,:,:,:]
#     print(im.size())
#
# #
#     kwargs_2 = {"target_type": tasks,
#                "mean": mean,
#                "std": std,
#                "length": None,
#                "period": 1,
#                "clips": "all"
#                }
#     val_dataset = Echo(split='val',**kwargs_2)
#     val_dataloader = torch.utils.data.DataLoader(
#     val_dataset, batch_size=20, num_workers=4, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=False)
#
#     test_dataset = Echo(split='test',**kwargs_2)
#     test_dataloader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=20, num_workers=4, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=False)






