import torch
import torch.nn as nn

from CDDFuse.net import BaseFeatureExtraction, DetailFeatureExtraction


class WeightFuser:
    def __init__(self, strategy="add"):
        if strategy not in ["add", "exp", "pow"]:
            raise ValueError(f"unknow strategy: {strategy}")
        self.strategy = strategy

    def forward(self, f1, f2, s1=None, s2=None, temperature: float = 0.3):
        if self.strategy == "add":
            return f1 + f2
        elif self.strategy == "exp":
            s1_t = torch.exp(s1 / temperature)
            s2_t = torch.exp(s2 / temperature)
        else:
            s1_t = torch.pow(s1, 2)
            s2_t = torch.pow(s2, 2)

        mask = (s1_t < 1e-3) & (s2_t < 1e-3)
        w1 = s1_t / (s1_t + s2_t)
        w2 = s2_t / (s1_t + s2_t)
        w1 = torch.where(mask, 0.5 * torch.ones_like(w1), w1)
        w2 = torch.where(mask, 0.5 * torch.ones_like(w2), w2)
        w1 = w1.repeat(1, f1.shape[1], 1, 1)
        w2 = w2.repeat(1, f2.shape[1], 1, 1)
        f = w1 * f1 + w2 * f2
        return f


class FuseConv(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        dim=64,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super(FuseConv, self).__init__()
        channels = [dim * 2, dim // 2, dim // 8, out_channels]
        layer_num = len(channels) - 1

        layers = []
        for i in range(layer_num):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.ReLU() if i < layer_num - 1 else nn.Sigmoid())
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class FeatureFuser(nn.Module):
    def __init__(self, dim=64, use_hybrid=True):
        super(FeatureFuser, self).__init__()
        self.base_convs = FuseConv(dim=dim, kernel_size=7, stride=1, padding=3)
        self.base_fuser = BaseFeatureExtraction(dim=dim, num_heads=8)

        self.use_hybrid = use_hybrid
        if self.use_hybrid:
            self.detail_convs = FuseConv(dim=dim, kernel_size=7, stride=1, padding=3)
            self.detail_fuser = DetailFeatureExtraction(num_layers=1)

    def forward(self, f1_base, f2_base, f1_detail=None, f2_detail=None):
        f_base = torch.cat((f1_base, f2_base), dim=1)  # [n,128,h,w]
        base_map = self.base_convs(f_base)  # [n,1,h,w]
        base_map = base_map.repeat(1, f1_base.shape[1], 1, 1)  # [n,64,h,w]
        fused_base = base_map * f1_base + (1 - base_map) * f2_base  # [n,64,h,w]
        fused_base = self.base_fuser(fused_base)  # [n,64,h,w]

        if self.use_hybrid:
            f_detail = torch.cat((f1_detail, f2_detail), dim=1)  # [n,128,h,w]
            detail_map = self.detail_convs(f_detail)  # [n,1,h,w]
            detail_map = detail_map.repeat(1, f1_detail.shape[1], 1, 1)  # [n,64,h,w]
            fused_detail = detail_map * f1_detail + (1 - detail_map) * f2_detail  # [n,64,h,w]
            fused_detail = self.detail_fuser(fused_detail)  # [n,64,h,w]
            return fused_base, fused_detail
        else:
            return fused_base, None


class CDDFuseFuser(nn.Module):
    def __init__(self, dim=64):
        super(CDDFuseFuser, self).__init__()
        self.base_fuser = BaseFeatureExtraction(dim=dim, num_heads=8)
        self.detail_fuser = DetailFeatureExtraction(num_layers=1)

    def forward(self, f1_base, f2_base, f1_detail, f2_detail):
        fused_base = self.base_fuser(f1_base + f2_base)  # [n,64,h,w]
        fused_detail = self.detail_fuser(f1_detail + f2_detail)  # [n,64,h,w]
        return fused_base, fused_detail
