from torch.nn import L1Loss, MSELoss

from TransMorph.TransMorph.losses import NCC, SSIM, Grad

mse_loss = MSELoss()
mae_loss = L1Loss()
ssim_loss = SSIM()
grad_loss = Grad(penalty="l2")
ncc_loss = NCC()
