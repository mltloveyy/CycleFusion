import torch
import torch.nn as nn


# Conv+Act+Norm
class ConvActNorm(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        use_relu=True,
        use_bn=True,
    ):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, bias=False)
        if use_relu:
            act = nn.ReLU(inplace=True)
        else:
            act = nn.LeakyReLU(inplace=True)
        if use_bn:
            norm = nn.BatchNorm2d(out_channels)
        else:
            norm = nn.InstanceNorm2d(out_channels)
        super(ConvActNorm, self).__init__(conv, act, norm)


# Channel-attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attn = self.softmax(out)
        out = x * attn + x
        return out


# Spatial-attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        attn = self.sigmoid(x)
        out = x * attn + x
        return out


# Convolutional Block Attention Module
class CBAM(nn.Module):
    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# Dense Convolution Module
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = torch.cat([x, out], dim=1)
        return out


# Dense Block Module
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride, repeat):
        super(DenseBlock, self).__init__()
        denseblock = []
        for i in range(repeat):
            denseblock.append(DenseConv2d(in_channels * (i + 1), in_channels, kernel_size, stride))
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        x = self.denseblock(x)
        return x


# Upsample Module
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        self.conv1 = ConvActNorm(in_channels + skip_channels, out_channels, 3, 1)
        self.conv2 = ConvActNorm(out_channels, out_channels, 3, 1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.conv1(x)
        x = self.conv2(x)
        return x
