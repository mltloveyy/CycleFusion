import torch
import torch.nn as nn

from CDDFuse.net import (
    BaseFeatureExtraction,
    DetailFeatureExtraction,
    OverlapPatchEmbed,
    TransformerBlock,
)
from cnn_modules import DenseBlock


# from densefuse
class DenseFuseEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        dim=64,
        repeat=3,
        with_quality=False,
    ):
        super(DenseFuseEncoder, self).__init__()
        denseblock = DenseBlock
        first_dim = dim // (2 ** (repeat - 1))

        self.conv = nn.Conv2d(in_channels, first_dim, kernel_size=3, stride=1, padding=1)
        self.DB = denseblock(in_channels=first_dim, kernel_size=3, stride=1, repeat=repeat)
        self.with_quality = with_quality
        if self.with_quality:
            self.score_head = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False), nn.Sigmoid())

    def forward(self, input):
        f = self.conv(input)  # [n,1,h,w] -> [n,16,h,w]
        f = self.DB(f)  # [n,64,h,w]
        if self.with_quality:
            score = torch.cat([torch.mean(f, 1, keepdim=True), torch.max(f, 1, keepdim=True)[0]], 1)  # [n,2,h,w]
            score = self.score_head(score)  # [n,1,h,w]
            return f, score
        else:
            return f, None


# from cddfuse
class CDDFuseEncoder(nn.Module):
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
        with_quality=False,
    ):

        super(CDDFuseEncoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(in_channels, dim)
        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_block)
            ]
        )
        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.detailFeature = DetailFeatureExtraction()
        self.with_quality = with_quality
        if self.with_quality:
            self.score_head = nn.Sequential(
                nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=bias),
                nn.LeakyReLU(),
                nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
                nn.Sigmoid(),
            )

    def forward(self, input):
        in_enc_level1 = self.patch_embed(input)  # [n,1,h,w] -> [n,64,h,w]
        out_enc_level1 = self.encoder_level1(in_enc_level1)  # [n,64,h,w]
        base_feature = self.baseFeature(out_enc_level1)  # [n,64,h,w]
        detail_feature = self.detailFeature(out_enc_level1)  # [n,64,h,w]
        if self.with_quality:
            features = torch.cat((base_feature, detail_feature), dim=1)  # [n,128,h,w]
            score = self.score_head(features)  # [n,1,h,w]
            return base_feature, detail_feature, score
        else:
            return base_feature, detail_feature, None
