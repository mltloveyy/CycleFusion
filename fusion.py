import torch
import torch.nn as nn

from CDDFuse.net import BaseFeatureExtraction, DetailFeatureExtraction

EPSILON = 1e-6


def weight_fusion(
    f1: torch.Tensor,
    f2: torch.Tensor,
    s1: torch.Tensor = None,
    s2: torch.Tensor = None,
    strategy_type: str = "add",
    temperature: float = 0.3,
) -> torch.Tensor:
    shape = f1.size()
    if strategy_type == "add":
        return f1 + f2
    elif strategy_type == "exp":
        s1_t = torch.exp(s1 / temperature)
        s2_t = torch.exp(s2 / temperature)
    elif strategy_type == "pow":
        s1_t = torch.pow(s1, 2)
        s2_t = torch.pow(s2, 2)
    else:
        raise ValueError(f"unknow strategy type: {strategy_type}")

    mask = (s1_t < 1e-3) & (s2_t < 1e-3)
    w1 = s1_t / (s1_t + s2_t)
    w2 = s2_t / (s1_t + s2_t)
    w1 = torch.where(mask, 0.5 * torch.ones_like(w1), w1)
    w2 = torch.where(mask, 0.5 * torch.ones_like(w2), w2)

    w1 = w1.repeat(1, shape[1], 1, 1)
    w2 = w2.repeat(1, shape[1], 1, 1)

    f = w1 * f1 + w2 * f2

    return f


class FeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super(FeatureFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels // 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels // 16, 1, kernel_size=3, stride=1, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, f1, f2):
        shape = f1.size()
        map = torch.cat((f1, f2), dim=1)
        map = self.conv1(map)
        map = self.conv2(map)
        map = self.conv3(map)
        map = self.act(map)
        map = map.repeat(1, shape[1], 1, 1)
        out = map * f1 + (1 - map) * f2
        return out


class CDDFuseFuser(nn.Module):
    def __init__(self):
        super(CDDFuseFuser, self).__init__()
        self.base_fuser = BaseFeatureExtraction(dim=64, num_heads=8)
        self.detail_fuser = DetailFeatureExtraction(num_layers=1)

    def forward(self, f_base, f_detail):
        f_base = self.base_fuser(f_base)
        f_detail = self.detail_fuser(f_detail)
        return f_base, f_detail
