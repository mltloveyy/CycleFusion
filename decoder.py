import torch
import torch.nn as nn

from CDDFuse.net import TransformerBlock
from cnn_modules import ConvActNorm
from fuser import FeatureFuser


# from densefuse
class DenseFuseDecoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        dim=64,
    ):
        super(DenseFuseDecoder, self).__init__()
        channels = [dim, dim, dim // 2, dim // 4, out_channels]

        layers = [nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1) for i in range(len(channels) - 1)]
        self.convs = nn.Sequential(*layers)
        self.act = nn.Sigmoid()

    def forward(self, f):
        out = self.convs(f)
        out = self.act(out)
        return out


# from cddfuse
class CDDFuseDecoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        dim=64,
        num_blocks=[4, 4],
        heads=[8, 8, 8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="WithBias",
    ):

        super(CDDFuseDecoder, self).__init__()
        self.reduce_channel = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(dim // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, base_feature, detail_feature):
        f = torch.cat((base_feature, detail_feature), dim=1)
        out = self.reduce_channel(f)
        out = self.encoder_level2(out)
        out = self.output(out)
        out = self.sigmoid(out)
        return out
