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


class FeatureFuser(nn.Module):
    def __init__(self, dim=64):
        super(FeatureFuser, self).__init__()
        channels = [dim, dim // 4, dim // 16, 1]

        base_layers = [nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1) for i in range(len(channels) - 1)]
        self.base_convs = nn.Sequential(*base_layers)
        self.base_act = nn.Sigmoid()

        detail_layers = [nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1) for i in range(len(channels) - 1)]
        self.detail_convs = nn.Sequential(*detail_layers)
        self.detail_act = nn.Sigmoid()

    def forward(self, f1_base, f1_detail, f2_base, f2_detail):
        base_map = self.base_convs(torch.cat((f1_base, f2_base), dim=1))
        base_map = self.base_act(base_map)
        base_map = base_map.repeat(1, f1_base.shape[1], 1, 1)
        f_base = base_map * f1_base + (1 - base_map) * f2_base

        detail_map = self.detail_convs(torch.cat((f1_detail, f2_detail), dim=1))
        detail_map = self.detail_act(detail_map)
        detail_map = detail_map.repeat(1, f1_detail.shape[1], 1, 1)
        f_detail = detail_map * f1_detail + (1 - detail_map) * f2_detail
        return f_base, f_detail


class CDDFuseFuser(nn.Module):
    def __init__(self, dim=64):
        super(CDDFuseFuser, self).__init__()
        self.base_fuser = BaseFeatureExtraction(dim=dim, num_heads=8)
        self.detail_fuser = DetailFeatureExtraction(num_layers=1)

    def forward(self, f1_base, f1_detail, f2_base, f2_detail):
        f_base = self.base_fuser(f1_base + f2_base)
        f_detail = self.detail_fuser(f1_detail + f2_detail)
        return f_base, f_detail
