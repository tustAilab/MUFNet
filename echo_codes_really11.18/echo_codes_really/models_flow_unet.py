import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt

from unet_model import UNet

def conv(in_channels, out_channels, kernel_size=3, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=False),
        nn.ReLU(inplace=True))


def predict_flow(in_channels):
    return nn.Conv2d(in_channels, 2, 5, stride=1, padding=2, bias=False)


def upconv(in_channels, out_channels):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                         nn.ReLU(inplace=True))


def concatenate(tensor1, tensor2, tensor3):
    _, _, h1, w1 = tensor1.shape
    _, _, h2, w2 = tensor2.shape
    _, _, h3, w3 = tensor3.shape
    h, w = min(h1, h2, h3), min(w1, w2, w3)
    return torch.cat((tensor1[:, :, :h, :w], tensor2[:, :, :h, :w], tensor3[:, :, :h, :w]), 1)


# class FlowNetS(nn.Module):
#     def __init__(self):
#         super(FlowNetS, self).__init__()
#
#         self.conv1 = conv(6, 64, kernel_size=7)
#         self.conv2 = conv(64, 128, kernel_size=5)
#         self.conv3 = conv(128, 256, kernel_size=5)
#         self.conv3_1 = conv(256, 256, stride=1)
#         self.conv4 = conv(256, 512)
#         self.conv4_1 = conv(512, 512, stride=1)
#         self.conv5 = conv(512, 512)
#         self.conv5_1 = conv(512, 512, stride=1)
#         self.conv6 = conv(512, 1024)
#
#         self.predict_flow6 = predict_flow(1024)  # conv6 output
#         self.predict_flow5 = predict_flow(1026)  # upconv5 + 2 + conv5_1
#         self.predict_flow4 = predict_flow(770)  # upconv4 + 2 + conv4_1
#         self.predict_flow3 = predict_flow(386)  # upconv3 + 2 + conv3_1
#         self.predict_flow2 = predict_flow(194)  # upconv2 + 2 + conv2
#
#         self.upconv5 = upconv(1024, 512)
#         self.upconv4 = upconv(1026, 256)
#         self.upconv3 = upconv(770, 128)
#         self.upconv2 = upconv(386, 64)
#
#         self.upconvflow6 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#         self.upconvflow5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#         self.upconvflow4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#         self.upconvflow3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#
#     def forward(self, x):
#
#         out_conv2 = self.conv2(self.conv1(x))
#         out_conv3 = self.conv3_1(self.conv3(out_conv2))
#         out_conv4 = self.conv4_1(self.conv4(out_conv3))
#         out_conv5 = self.conv5_1(self.conv5(out_conv4))
#         out_conv6 = self.conv6(out_conv5)
#
#         flow6 = self.predict_flow6(out_conv6)
#         up_flow6 = self.upconvflow6(flow6)
#         out_upconv5 = self.upconv5(out_conv6)
#         concat5 = concatenate(out_upconv5, out_conv5, up_flow6)
#
#         flow5 = self.predict_flow5(concat5)
#         up_flow5 = self.upconvflow5(flow5)
#         out_upconv4 = self.upconv4(concat5)
#         concat4 = concatenate(out_upconv4, out_conv4, up_flow5)
#
#         flow4 = self.predict_flow4(concat4)
#         up_flow4 = self.upconvflow4(flow4)
#         out_upconv3 = self.upconv3(concat4)
#         concat3 = concatenate(out_upconv3, out_conv3, up_flow4)
#
#         flow3 = self.predict_flow3(concat3)
#         up_flow3 = self.upconvflow3(flow3)
#         out_upconv2 = self.upconv2(concat3)
#         concat2 = concatenate(out_upconv2, out_conv2, up_flow3)
#
#         finalflow = self.predict_flow2(concat2)
#
#         if self.training:
#             return finalflow, flow3, flow4, flow5, flow6
#         else:
#             return finalflow,

class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)
    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x

class GatedDeConv2dWithActivation(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, batch_norm=True,activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedDeConv2dWithActivation, self).__init__()
        self.conv2d = GatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor
    def forward(self, input):
        #print(input.size())
        x = F.interpolate(input, scale_factor=self.scale_factor)
        return self.conv2d(x)

class FlowNetS(nn.Module):
    def __init__(self):
        super(FlowNetS, self).__init__()

        self.conv1 = conv(6, 32, kernel_size=7)
        self.conv2 = GatedConv2dWithActivation(32, 64, 5, 2, 2)
        self.conv3 = GatedConv2dWithActivation(64, 128, 5, 2, 2)
        self.conv3_1 = GatedConv2dWithActivation(128, 128, 3, 1, 1)
        self.conv4 = GatedConv2dWithActivation(128, 256, 3, 2, 1)


        # self.predict_flow4 = predict_flow(256)  # conv4 output
        # self.predict_flow3 = predict_flow(194)  # upconv3 + 2 + conv3_1
        # self.predict_flow2 = predict_flow(130)  # upconv2 + 2 + conv2
        # self.predict_flow1 = predict_flow(32)

        self.predict_flow4 = GatedConv2dWithActivation(256, 2, 5, 1, 2)  # conv4 output
        self.predict_flow3 = GatedConv2dWithActivation(194, 2, 5, 1, 2)  # upconv3 + 2 + conv3_1
        self.predict_flow2 = GatedConv2dWithActivation(130, 2, 5, 1, 2)  # upconv2 + 2 + conv2
        self.predict_flow1 = predict_flow(32)

        self.upconv3 = GatedDeConv2dWithActivation(2, 256, 64, 3, padding=1)
        self.upconv2 = GatedDeConv2dWithActivation(2, 194, 64, 3, padding=1)
        self.upconv1 = GatedDeConv2dWithActivation(2, 130, 32, 3, padding=1)
        self.upconv0 = GatedDeConv2dWithActivation(2, 66, 32, 3, padding=1)

        self.upconvflow4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upconvflow3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upconvflow2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)

    def forward(self, x):

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4(out_conv3)

        flow4 = self.predict_flow4(out_conv4)
        up_flow4 = self.upconvflow4(flow4)
        out_upconv3 = self.upconv3(out_conv4)
        concat3 = concatenate(out_upconv3, out_conv3, up_flow4)

        flow3 = self.predict_flow3(concat3)
        up_flow3 = self.upconvflow3(flow3)
        out_upconv2 = self.upconv2(concat3)
        concat2 = concatenate(out_upconv2, out_conv2, up_flow3)

        flow2 = self.predict_flow2(concat2)
        up_flow2 = self.upconvflow2(flow2)
        out_upconv1 = self.upconv1(concat2)
        concat1 = concatenate(out_upconv1, out_conv1, up_flow2)

        up_flow1 = self.upconv0(concat1)

        finalflow = self.predict_flow1(up_flow1)

        if self.training:
            return finalflow, flow2, flow3
        else:
            return finalflow,

# class FlowNetS(nn.Module):
#     def __init__(self):
#         super(FlowNetS, self).__init__()
#
#         # self.conv1 = conv(6, 64, kernel_size=7)
#         # self.conv2 = conv(64, 128, kernel_size=5)
#         # self.conv3 = conv(128, 256, kernel_size=5)
#         # self.conv3_1 = conv(256, 256, stride=1)
#         # self.conv4 = conv(256, 512)
#         #
#         # # self.predict_flow6 = predict_flow(1024)  # conv6 output
#         # # self.predict_flow5 = predict_flow(1026)  # upconv5 + 2 + conv5_1
#         # self.predict_flow4 = predict_flow(512)  # conv4 output
#         # self.predict_flow3 = predict_flow(386)  # upconv3 + 2 + conv3_1
#         # self.predict_flow2 = predict_flow(194)  # upconv2 + 2 + conv2
#         # self.predict_flow1 = predict_flow(64)
#         #
#         # self.upconv3 = upconv(512, 128)
#         # self.upconv2 = upconv(386, 64)
#         # self.upconv1 = upconv(194, 64)
#         # self.upconv0 = upconv(130, 64)
#
#         self.conv1 = conv(6, 32, kernel_size=7)
#         self.conv2 = conv(32, 64, kernel_size=5)
#         self.conv3 = conv(64, 128, kernel_size=5)
#         # self.conv3_1 = conv(128, 128, stride=1)
#         self.conv4 = conv(128, 256)
#
#         # self.predict_flow6 = predict_flow(1024)  # conv6 output
#         # self.predict_flow5 = predict_flow(1026)  # upconv5 + 2 + conv5_1
#         self.predict_flow4 = predict_flow(256)  # conv4 output
#         self.predict_flow3 = predict_flow(194)  # upconv3 + 2 + conv3_1
#         self.predict_flow2 = predict_flow(130)  # upconv2 + 2 + conv2
#         self.predict_flow1 = predict_flow(32)
#
#         self.upconv3 = upconv(256, 64)
#         self.upconv2 = upconv(194, 64)
#         self.upconv1 = upconv(130, 32)
#         self.upconv0 = upconv(66, 32)
#
#         self.upconvflow4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#         self.upconvflow3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#         self.upconvflow2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#
#     def forward(self, x):
#
#         out_conv1 = self.conv1(x)
#         out_conv2 = self.conv2(out_conv1)
#         # out_conv3 = self.conv3_1(self.conv3(out_conv2))
#         out_conv3 = self.conv3(out_conv2)
#         out_conv4 = self.conv4(out_conv3)
#
#         flow4 = self.predict_flow4(out_conv4)
#         up_flow4 = self.upconvflow4(flow4)
#         out_upconv3 = self.upconv3(out_conv4)
#         concat3 = concatenate(out_upconv3, out_conv3, up_flow4)
#
#         flow3 = self.predict_flow3(concat3)
#         up_flow3 = self.upconvflow3(flow3)
#         out_upconv2 = self.upconv2(concat3)
#         concat2 = concatenate(out_upconv2, out_conv2, up_flow3)
#
#         flow2 = self.predict_flow2(concat2)
#         up_flow2 = self.upconvflow2(flow2)
#         out_upconv1 = self.upconv1(concat2)
#         concat1 = concatenate(out_upconv1, out_conv1, up_flow2)
#
#         up_flow1 = self.upconv0(concat1)
#
#         finalflow = self.predict_flow1(up_flow1)
#
#         if self.training:
#             return finalflow, flow2, flow3
#         else:
#             return finalflow,

# class FlowNetS(nn.Module):
#     def __init__(self):
#         super(FlowNetS, self).__init__()
#
#         self.conv1 = conv(6, 64, kernel_size=7)
#         self.conv2 = conv(64, 128, kernel_size=5)
#         self.conv3 = conv(128, 256, kernel_size=5)
#         self.conv3_1 = conv(256, 256, stride=1)
#         self.conv4 = conv(256, 512)
#         # self.conv4_1 = conv(512, 512, stride=1)
#         # self.conv5 = conv(512, 512)
#         # self.conv5_1 = conv(512, 512, stride=1)
#         # self.conv6 = conv(512, 1024)
#
#         # self.predict_flow6 = predict_flow(1024)  # conv6 output
#         # self.predict_flow5 = predict_flow(1026)  # upconv5 + 2 + conv5_1
#         self.predict_flow4 = predict_flow(512)  # conv4 output
#         self.predict_flow3 = predict_flow(386)  # upconv3 + 2 + conv3_1
#         self.predict_flow2 = predict_flow(194)  # upconv2 + 2 + conv2
#
#         # self.upconv5 = upconv(1024, 512)
#         # self.upconv4 = upconv(1026, 256)
#         self.upconv3 = upconv(512, 128)
#         self.upconv2 = upconv(386, 64)
#
#         # self.upconvflow6 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#         # self.upconvflow5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#         self.upconvflow4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#         self.upconvflow3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#
#     def forward(self, x):
#
#         out_conv2 = self.conv2(self.conv1(x))
#         out_conv3 = self.conv3_1(self.conv3(out_conv2))
#         # out_conv4 = self.conv4_1(self.conv4(out_conv3))
#         # out_conv5 = self.conv5_1(self.conv5(out_conv4))
#         out_conv4 = self.conv4(out_conv3)
#
#         # flow6 = self.predict_flow6(out_conv6)
#         # up_flow6 = self.upconvflow6(flow6)
#         # out_upconv5 = self.upconv5(out_conv6)
#         # concat5 = concatenate(out_upconv5, out_conv5, up_flow6)
#         #
#         # flow5 = self.predict_flow5(concat5)
#         # up_flow5 = self.upconvflow5(flow5)
#         # out_upconv4 = self.upconv4(concat5)
#         # concat4 = concatenate(out_upconv4, out_conv4, up_flow5)
#
#         flow4 = self.predict_flow4(out_conv4)
#         up_flow4 = self.upconvflow4(flow4)
#         out_upconv3 = self.upconv3(out_conv4)
#         concat3 = concatenate(out_upconv3, out_conv3, up_flow4)
#
#         flow3 = self.predict_flow3(concat3)
#         up_flow3 = self.upconvflow3(flow3)
#         out_upconv2 = self.upconv2(concat3)
#         concat2 = concatenate(out_upconv2, out_conv2, up_flow3)
#
#         finalflow = self.predict_flow2(concat2)
#
#         if self.training:
#             return finalflow, flow3, flow4
#         else:
#             return finalflow,


def generate_grid(B, H, W, device):
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    grid = torch.transpose(grid, 1, 2)
    grid = torch.transpose(grid, 2, 3)
    grid = grid.to(device)
    return grid


class Unsupervised(nn.Module):
    def __init__(self, conv_predictor="flownet"):
        super(Unsupervised, self).__init__()
        #
        # model_seg = torchvision.models.segmentation.__dict__["deeplabv3_resnet50"](pretrained=False, aux_loss=False)
        # model_seg.classifier[-1] = torch.nn.Conv2d(model_seg.classifier[-1].in_channels, 1, kernel_size=model_seg.classifier[-1].kernel_size)
        model_seg = UNet(n_channels=3, n_classes=1)

        self.predictor_f = FlowNetS()

        self.predictor_s = model_seg

    def stn(self, flow, frame):
        b, _, h, w = flow.shape
        # frame = F.interpolate(frame, size=(h, w), mode='bilinear', align_corners=True)
        flow = torch.transpose(flow, 1, 2)
        flow = torch.transpose(flow, 2, 3)

        grid = flow + generate_grid(b, h, w, flow.device)

        factor = torch.FloatTensor([[[[2 / w, 2 / h]]]]).to(flow.device)
        grid = grid * factor - 1  # grid blongs to -1 to 1, float, presents the flow field
        warped_frame = F.grid_sample(frame, grid)


        # b, _, h, w = flow.shape
        # frame = F.interpolate(frame, size=(h, w), mode='bilinear', align_corners=True)
        # flow = torch.transpose(flow, 1, 2)
        # flow = torch.transpose(flow, 2, 3)
        #
        # grid = flow + generate_grid(b, h, w, flow.device)
        #
        # factor = torch.FloatTensor([[[[2 / w, 2 / h]]]]).to(flow.device)
        # grid = grid * factor - 1 # grid blongs to -1 to 1, float, presents the flow field
        # warped_frame = F.grid_sample(frame, grid)

        return warped_frame
    # def drawing(self,input):
    #     tensors = np.split(input, 4, axis=0)
    #     for i in tensors:
    #         tensor = torch.ones((1, 1, 112, 112), dtype=torch.float32)
    #         last_one_index_y = np.where(i.any(axis=3))[0][-1] #y1
    #         last_one_x = np.where(i[last_one_index_y, :])[0][0] #x1
    #
    #         last_one_index_right_x = np.where(i.any(axis=2))[0][-1] #x2
    #         last_one_left_y = np.where(i[:, last_one_index_right_x])[0][-1] #y2
    #         rectangle = np.zeros_like(input)
    #         coor = [(last_one_x, last_one_index_y + 1), (last_one_x, last_one_index_y + 3),
    #             (last_one_index_right_x, last_one_left_y + 3), (last_one_index_right_x, last_one_left_y + 1)]
    #         for i in range(len(coor)):
    #             cv2.line(rectangle, coor[i % 4], coor[(i + 1) % 4], 255, 1)
    #     rectangle = cv2.dilate(rectangle, None)
    #     input[rectangle == 255] = 0
    #     return input
    def forward(self, x):
        b,c,h,c=x.size
        flow_predictions = self.predictor_f(x)
        frame2 = x[:, c//2:, :, :]

        # warped_images = self.stn(flow_predictions, frame2)
        warped_images1 = [self.stn(flow, frame2) for flow in flow_predictions]
        # print('''warped_iamges''', warped_images[0].size())10*3*28*28
        # frame1 = x[:, :3, :, :]
        # warped_images2 = [self.stn(flow, frame1) for flow in flow_predictions]


        # seg1 = self.predictor_s(x[:, :3, :, :])["out"]
        # seg2 = self.predictor_s(x[:, 3:, :, :])["out"]
        #两个分割结果
        seg1 = self.predictor_s(x[:, :c//2, :, :])
        seg2 = self.predictor_s(x[:, c//2:, :, :])

        # print(seg2.size())
        # warped_segs = self.stn(flow_predictions, seg2)
        warped_segs1 = [self.stn(flow, seg2) for flow in flow_predictions]
        # warped_segs2 = [self.stn(flow, seg1) for flow in flow_predictions]
        # print('''warped_segs''', len(warped_segs), warped_segs[0].size())10*1*28*28
        return flow_predictions, warped_images1,seg1, seg2, warped_segs1
