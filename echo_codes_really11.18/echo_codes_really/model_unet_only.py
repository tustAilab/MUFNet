import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from unet_model import UNet


class Unsupervised(nn.Module):
    def __init__(self, conv_predictor="flownet"):
        super(Unsupervised, self).__init__()
        #
        # model_seg = torchvision.models.segmentation.__dict__["deeplabv3_resnet50"](pretrained=False, aux_loss=False)
        # model_seg.classifier[-1] = torch.nn.Conv2d(model_seg.classifier[-1].in_channels, 1, kernel_size=model_seg.classifier[-1].kernel_size)
        model_seg = UNet(n_channels=3, n_classes=1)

        self.predictor_s = model_seg

    def forward(self, x):
        seg1 = self.predictor_s(x[:, :3, :, :])
        seg2 = self.predictor_s(x[:, 3:, :, :])

        return seg1, seg2
