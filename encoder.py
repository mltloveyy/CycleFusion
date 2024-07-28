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
    def __init__(self, repeat=3):
        super(DenseFuseEncoder, self).__init__()
        denseblock = DenseBlock
        nb_filter = [1, 16]
        kernel_size = 3
        stride = 1

        self.conv = nn.Conv2d(nb_filter[0], nb_filter[1], kernel_size, stride, padding=kernel_size // 2)  # [n,1,h,w] -> [n,16,h,w]
        self.DB = denseblock(nb_filter[1], kernel_size, stride, repeat)  # [n,16,h,w] -> [n,64,h,w]
        self.score_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, input):
        f = self.conv(input)
        f = self.DB(f)
        f = torch.cat([torch.mean(f, 1, keepdim=True), torch.max(f, 1, keepdim=True)[0]], 1)
        score = self.score_conv(f)
        score = self.act(score)
        return f, score


# from cddfuse
class CDDFuseEncoder(nn.Module):
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
                for i in range(num_blocks[0])
            ]
        )
        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.detailFeature = DetailFeatureExtraction()
        self.score_head = nn.Sequential(
            # nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=bias),
            # nn.LeakyReLU(),
            # nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            # nn.Sigmoid(),
            nn.Conv2d(dim * 2, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, in_img, with_score=True):
        in_enc_level1 = self.patch_embed(in_img)  # b*64*h*w
        out_enc_level1 = self.encoder_level1(in_enc_level1)  # b*64*h*w
        base_feature = self.baseFeature(out_enc_level1)  # b*64*h*w
        detail_feature = self.detailFeature(out_enc_level1)  # b*64*h*w
        features = torch.cat((base_feature, detail_feature), dim=1)  # b*128*h*w
        if with_score:
            score = self.score_head(features)
            # score = self.score_head(detail_feature)  # use detail feature
            return features, score
        return features
