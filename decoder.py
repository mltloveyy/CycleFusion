import torch
import torch.nn as nn

from CDDFuse.net import TransformerBlock
from cnn_modules import ConvActNorm
from fusion import FeatureFusion


# from densefuse
class DenseFuseDecoder(nn.Module):
    def __init__(self):
        super(DenseFuseDecoder, self).__init__()
        dims = [64, 32, 16, 1]

        # decoder
        self.conv1 = ConvActNorm(dims[0], dims[0], kernel_size=3, stride=1)
        self.conv2 = ConvActNorm(dims[0], dims[1], kernel_size=3, stride=1)
        self.conv3 = ConvActNorm(dims[1], dims[2], kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(dims[2], dims[3], kernel_size=3, stride=1, padding=1)
        self.act = nn.Sigmoid()
        self.fusion = FeatureFusion(dims[0] * 2)

    def forward(self, f1, f2=None):
        if f2 is not None:
            f = self.fusion(f1, f2)
        else:
            f = f1
        out = self.conv1(f)  # [n,64,h,w] -> [n,64,h,w]
        out = self.conv2(out)  # [n,64,h,w] -> [n,32,h,w]
        out = self.conv3(out)  # [n,32,h,w] -> [n,16,h,w]
        out = self.conv4(out)  # [n,16,h,w] -> [n,1,h,w]
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
        self.reduce_channel = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
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
            nn.Conv2d(int(dim), int(dim) // 2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim) // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
        )
        self.sigmoid = nn.Sigmoid()
        self.fusion = FeatureFusion(dim * 4)

    def forward(self, f1, f2=None, inp_img=None):
        if f2 is not None:
            f = self.fusion(f1, f2)  # ours
            # f = torch.cat((f1, f2), dim=1) # CDDFuse
        else:
            f = f1
        out_enc_level0 = self.reduce_channel(f)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1)
