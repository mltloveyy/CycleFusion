import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineTransform(nn.Module):
    """
    2-D Affine Transformer
    """

    def __init__(self, mode="bilinear"):
        super().__init__()
        self.mode = mode

    def forward(self, src, params):
        """
        src: Input source tensor(N, C, H, W).
        params: Transformation matrix(N, 6) or parameters(N, 7) which including rotation angle, scaling, translation and shear angle.
        """
        if params.shape[-1] == 6:
            mat = params.reshape(-1, 2, 3)
        else:
            theta = params[:, 0]
            scale_x = params[:, 1] + 1
            scale_y = params[:, 2] + 1
            translate_x = params[:, 3]
            translate_y = params[:, 4]
            shear_xy = params[:, 5]
            shear_yx = params[:, 6]

            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            tan_shear_xy = torch.tan(shear_xy)
            tan_shear_yx = torch.tan(shear_yx)

            # affine transformation matrix
            mat = torch.zeros((params.shape[0], 2, 3), dtype=src.dtype, device=src.device)
            mat[:, 0, 0] = scale_x * cos_theta + tan_shear_xy * sin_theta
            mat[:, 0, 1] = scale_x * -sin_theta + tan_shear_xy * cos_theta
            mat[:, 0, 2] = translate_x
            mat[:, 1, 0] = scale_y * sin_theta + tan_shear_yx * cos_theta
            mat[:, 1, 1] = scale_y * cos_theta + tan_shear_yx * sin_theta
            mat[:, 1, 2] = translate_y

        grid = F.affine_grid(mat, src.shape)
        return F.grid_sample(src, grid, mode=self.mode)


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
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids).unsqueeze(0).double()

        # register the grid as a buffer
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position and reverse channels
        new_locs = new_locs.permute(0, 2, 3, 1)[:, :, :, [1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)


class HybridDeformer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        dim=64,
        img_size=(224, 224),
        pred_affine_mat=False,
    ):
        super(HybridDeformer, self).__init__()
        channels = [dim * 2, dim, dim // 2, dim // 4]
        layer_num = len(channels) - 1

        base_layers = []
        for i in range(layer_num):
            base_layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1))
            base_layers.append(nn.LeakyReLU())
            base_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.base_convs = nn.Sequential(*base_layers)
        affine_num = 6 if pred_affine_mat else 7  # pred affine params(angle, scaling, translation and shear) or transformation matrix
        self.base_head = nn.Sequential(nn.Conv2d(channels[-1], affine_num, kernel_size=3, stride=1, padding=1), nn.AdaptiveAvgPool2d((1, 1)))

        detail_layers = []
        for i in range(layer_num):
            detail_layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1))
            detail_layers.append(nn.LeakyReLU())
        self.detail_convs = nn.Sequential(*detail_layers)
        self.detail_head = nn.Conv2d(channels[-1], 2, kernel_size=3, stride=1, padding=1)

        self.affine_transform = AffineTransform()
        self.spatial_transformer = SpatialTransformer(size=img_size)

    def forward(self, moving, f1_base, f2_base, f1_detail=None, f2_detail=None):
        f_base = torch.cat((f1_base, f2_base), dim=1)  # [n,128,h,w]
        affine_mat = self.base_convs(f_base)  # [n,16,h/8,w/8]
        affine_mat = self.base_head(affine_mat)  # [n,7,1,1]
        transformed = self.affine_transform(moving, affine_mat)

        if f1_detail is not None and f2_detail is not None:
            f_detail = torch.cat((f1_detail, f2_detail), dim=1)  # [n,128,h,w]
        else:
            f_detail = f_base
        flow = self.detail_convs(f_detail)  # [n,16,h,w]
        flow = self.detail_head(flow)  # [n,2,h,w]
        registered = self.spatial_transformer(transformed, flow)
        return registered, transformed, flow
