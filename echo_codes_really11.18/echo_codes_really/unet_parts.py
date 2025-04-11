""" Parts of the U-Net model """
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Softmax, Dropout, LayerNorm


class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.num_attention_heads = 4#config.transformer["num_heads"]#8
        self.attention_head_size = 128#int(config.hidden_size / self.num_attention_heads) #512/8:64
        self.all_head_size = 512#self.num_attention_heads * self.attention_head_size #512

        self.query = Linear(512,512) #512,512
        self.key = Linear(512, 512) #512,512
        self.value = Linear(512, 512) #512,512

        self.out = Linear(512, 512) #512,512
        self.attn_dropout = Dropout(0.0,inplace=False)
        self.proj_dropout =  Dropout(0.0,inplace=False)

        self.softmax = Softmax(dim=-1)
        self.beta=nn.Parameter(torch.zeros(1))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        attention_output=self.beta*attention_output+hidden_states
        return attention_output
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self):
        super(Embeddings, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, 49,512))
        self.dropout = Dropout(0.15)
    def forward(self, x):
        #x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.attention_norm = LayerNorm(512, eps=1e-6)
        self.embedding=Embeddings()
        self.attention=Attention()
        self.gama=nn.Parameter(torch.ones(1))
    def trans(self,x):
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        return x

    def forward(self, x):
        x=self.embedding(x)
        x=self.attention_norm(x)
        x=self.attention(x)
        x=self.trans(x)
        return x*self.gama

class Semantic_fusion(nn.Module):
    def __init__(self):
        super(Semantic_fusion,self).__init__()
        self.con1=nn.Conv2d(512,256,1,1)
        self.con2=nn.Conv2d(512,256,3,1,'same')
        self.con3=nn.Conv2d(512,256,3,1,'same',2)
        self.con4=nn.Conv2d(512,256,3,1,'same',3)
        self.con5=nn.Conv2d(1024,512,1,1)
    def forward(self,x):
        x1=self.con1(x)
        x2=self.con2(x)
        x3=self.con3(x)
        x4=self.con4(x)
        #x5=self.atten(x)
        xout=self.con5(torch.cat([x1,x2,x3,x4],dim=1))

        return xout
class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)
        self.conv_1x1_1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.x_h_m = nn.AdaptiveMaxPool2d((None, 1))
        self.x_w_m = nn.AdaptiveMaxPool2d((1, None))

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
        self.con=nn.Conv2d(in_channels=(channel//reduction)*2,out_channels=channel // reduction, kernel_size=1, stride=1,bias=False)


    def forward(self, x):
        _, _, h, w = x.size()

        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_h_m = self.x_h_m(x).permute(0, 1, 3, 2)
        x_w_m = self.x_w_m(x)
        c=torch.cat((x_h,x_w),dim=3)
        # y=torch.cat((x_h,x_w),dim=3)
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
        x_cat_conv_relu_m = self.relu(self.bn(self.conv_1x1_1(torch.cat((x_h_m, x_w_m), 3))))
        x1 = torch.cat((x_cat_conv_relu_m, x_cat_conv_relu), dim=1)
        x_cat_conv = self.con(x1)

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv.split([h, w], 3)

        #x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out
def sobel_filter(x):
    # Sobel算子核
    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device)

    # 扩展Sobel核到输入通道数
    sobel_kernel_x = sobel_kernel_x.repeat(x.size(1), 1, 1, 1)
    sobel_kernel_y = sobel_kernel_y.repeat(x.size(1), 1, 1, 1)

    # 应用Sobel算子到每个通道
    sobel_x = F.conv2d(x, sobel_kernel_x, padding=1, groups=x.size(1))
    sobel_y = F.conv2d(x, sobel_kernel_y, padding=1, groups=x.size(1))

    # 计算梯度幅度
    sobel = torch.sqrt(sobel_x ** 2 + sobel_y ** 2)

    return sobel


# 通道注意力（CA）
# 双分支网络结构
class DualBranchNetwork(nn.Module):
    def __init__(self, channel):
        super(DualBranchNetwork, self).__init__()
        self.ca = CA_Block(channel)
        self.sobel = sobel_filter

    def forward(self, x):
        # CA分支
        ca_out = self.ca(x)
        # Sobel分支
        #out=self.ca(x)
        sobel_out = self.sobel(x)

        #sobel_out=self.sigmod(sobel_out)
        # 将两个分支的结果相加
        enhanced_feature = sobel_out+ca_out+x

        return enhanced_feature


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        #my add

    def forward(self, x):
        x=self.double_conv(x)
        return x
        #return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            self.ac = DualBranchNetwork(in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1= self.ac(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Oconv(nn.Module):
    ''' conv 1*1, stride =1'''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.o_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=False)
        )
    def forward(self, x):
        return self.o_conv(x)