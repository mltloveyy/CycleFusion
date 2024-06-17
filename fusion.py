import torch
import torch.nn as nn

EPSILON = 1e-6


def weight_fusion(
    f1: torch.Tensor,
    f2: torch.Tensor,
    s1: torch.Tensor,
    s2: torch.Tensor,
    strategy_type="weight",
) -> torch.Tensor:
    shape = f1.size()

    if strategy_type == "weight":
        s1_t = s1
        s2_t = s2
    elif strategy_type == "exponential":
        s1_t = torch.exp(s1)
        s2_t = torch.exp(s2)
    elif strategy_type == "power":
        s1_t = torch.pow(s1, 3)
        s2_t = torch.pow(s2, 3)
    else:
        raise ValueError(f"unknow strategy type: {strategy_type}")

    w1 = s1_t / (s1_t + s2_t + EPSILON)
    w2 = s2_t / (s1_t + s2_t + EPSILON)

    w1 = w1.repeat(1, shape[1], 1, 1)
    w2 = w2.repeat(1, shape[1], 1, 1)

    f = w1 * f1 + w2 * f2

    return f


class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        input_dim = 256
        dim = 64

        self.conv1 = nn.Conv2d(input_dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, f1, f2):
        shape = f1.size()
        map = torch.cat((f1, f2), dim=1)
        map = self.conv1(map)
        map = self.conv2(map)
        map = self.act(map)
        map = map.repeat(1, shape[1], 1, 1)
        out = map * f1 + (1 - map) * f2
        return out
