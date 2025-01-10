import torch.nn as nn

from decoder import CDDFuseDecoder, DenseFuseDecoder
from encoder import CDDFuseEncoder, DenseFuseEncoder
from fuser import CDDFuseFuser, FeatureFuser, WeightFuser


class CDDFuse:
    def __int__(self, feature_dim=64):
        self.encoder = CDDFuseEncoder(dim=feature_dim)
        self.decoder = CDDFuseDecoder(dim=feature_dim)
        self.fuser = CDDFuseFuser(dim=feature_dim)

    def encode(self, input1, input2):
        self.f1_base, self.f1_detail = self.encoder(input1, False)
        self.f2_base, self.f2_detail = self.encoder(input2, False)

    def decode(self):
        output1 = self.decoder(self.f1_base, self.f1_detail)
        output2 = self.decoder(self.f2_base, self.f2_detail)
        return output1, output2

    def fuse(self):
        f_base, f_detail = self.fuser(self.f1_base, self.f1_detail, self.f2_base, self.f2_detail)
        fused = self.decoder(f_base, f_detail)
        return fused


class QualityFuse:
    """
    params:
    - network_type (str): densefuse or cddfuse
    - fuse_type (str): add, exp, pow or feature
    - with_quality (bool): whether to use quality information
    - with_reval (bool): whether to re-evaluate quality of fused image
    """

    def __int__(self, network_type, fuse_type, with_quality, with_reval):
        super(QualityFuse, self).__init__()
        if fuse_type not in ["add", "exp", "pow", "feature"]:
            raise ValueError(f"unknow fuse type: {fuse_type}")
        if network_type not in ["densefuse", "cddfuse"]:
            raise ValueError(f"unknow network type: {network_type}")
        self.encoder = CDDFuseEncoder() if network_type == "cddfuse" else DenseFuseEncoder()
        self.decoder = CDDFuseDecoder() if network_type == "cddfuse" else DenseFuseDecoder()
        self.fuser = FeatureFuser() if fuse_type == "feature" else WeightFuser(strategy=fuse_type)

        self.with_quality = with_quality
        self.fuse_type = fuse_type
        self.with_reval = with_reval

    def encode(self, input1, input2):
        if self.with_quality:
            self.f1, self.score1 = self.encoder(input1, True)
            self.f2, self.score2 = self.encoder(input2, True)
            return self.score1, self.score2
        else:
            self.f1 = self.encoder(input1, False)
            self.f2 = self.encoder(input2, False)

    def decode(self):
        output1 = self.decoder(self.f1)
        output2 = self.decoder(self.f2)
        return output1, output2

    def fuse(self):
        if self.fuse_type in ["exp", "pow"] and self.with_quality:
            f = self.fuser(self.f1, self.f2, self.score1, self.score2, strategy=self.fuse_type)
        else:
            f = self.fuser(self.f1, self.f2)
        fused = self.decoder(f)
        if self.with_reval and self.with_quality:
            _, score = self.encoder(fused)
            return fused, score
        else:
            return fused
