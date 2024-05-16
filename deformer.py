import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from cnn_modules import UpBlock


class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2)
        conv2d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv2d.weight.shape))
        conv2d.bias = nn.Parameter(torch.zeros(conv2d.bias.shape))
        super().__init__(conv2d)


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode="bilinear"):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class CnnRegisterer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        dims=[128, 64, 32],
        reg_head_channel=16,
        img_size=224,
    ):
        super(CnnRegisterer, self).__init__()

        self.conv = nn.Conv2d(2 * dims[0], dims[0], kernel_size=3, stride=1, padding=1)
        self.up0 = nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1)
        self.up1 = nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=1, padding=1)
        self.up2 = nn.Conv2d(dims[2], reg_head_channel, kernel_size=3, stride=1, padding=1)
        # self.up0 = UpBlock(dims[0], dims[1])
        # self.up1 = UpBlock(dims[1], dims[2])
        # self.up2 = UpBlock(dims[2], reg_head_channel)
        self.reg_head = RegistrationHead(reg_head_channel, out_channels, 3)
        self.spatial_trans = SpatialTransformer((img_size, img_size))

    def forward(self, f1, f2, source):
        x = torch.cat((f1, f2), dim=1)
        x = self.conv(x)
        x = self.up0(x)
        x = self.up1(x)
        x = self.up2(x)
        flow = self.reg_head(x)
        out = self.spatial_trans(source, flow)
        return out, flow
