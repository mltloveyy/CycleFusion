import torch
import torch.nn as nn

from CDDFuse.net import TransformerBlock


# from densefuse
class DenseFuseDecoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        dim=64,
    ):
        super(DenseFuseDecoder, self).__init__()
        channels = [dim, dim // 2, dim // 4, out_channels]
        layer_num = len(channels) - 1

        layers = []
        for i in range(layer_num):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1))
            layers.append(nn.LeakyReLU() if i < layer_num - 1 else nn.Sigmoid())
        self.convs = nn.Sequential(*layers)

    def forward(self, f):
        out = self.convs(f)  # [n,64,h,w] -> [n,1,h,w]
        return out


# from cddfuse
class CDDFuseDecoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        dim=64,
        num_block=4,
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
                for _ in range(num_block)
            ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(dim // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, base_feature, detail_feature):
        f = torch.cat((base_feature, detail_feature), dim=1)  # [n,128,h,w]
        out = self.reduce_channel(f)  # [n,64,h,w]
        out = self.encoder_level2(out)  # [n,64,h,w]
        out = self.output(out)  # [n,1,h,w]
        return out
