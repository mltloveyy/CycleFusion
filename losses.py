import torch
from torch.nn import L1Loss, MSELoss
import kornia

from CDDFuse.utils.loss import Fusionloss
from TransMorph.TransMorph.losses import NCC, SSIM, Grad

mse_loss = MSELoss()
mae_loss = L1Loss()
ssim_loss = SSIM()
grad_loss = Grad(penalty="l2")
ncc_loss = NCC()
cddfuse_loss = Fusionloss()


# Set losses
def quality_loss(src, dst):
    return mse_loss(src, dst) + 10 * ssim_loss(src, dst)


def restore_loss(src, dst):
    return mse_loss(src, dst) + 10 * ssim_loss(src, dst)


def fuse_loss(src, dst):
    return mse_loss(src, dst) + 10 * ssim_loss(src, dst)


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1**2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1.0, 1.0)
    return cc.mean()


def gradient_loss(src, dst):
    grad_src = kornia.filters.SpatialGradient()(src)
    grad_dst = kornia.filters.SpatialGradient()(dst)
    return mae_loss(grad_src, grad_dst) + mse_loss(src, dst)