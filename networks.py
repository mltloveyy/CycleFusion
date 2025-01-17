from decoder import CDDFuseDecoder, DenseFuseDecoder
from deformer import HybridDeformer
from encoder import CDDFuseEncoder, DenseFuseEncoder
from fuser import CDDFuseFuser, FeatureFuser, WeightFuser
from utils import load_model, save_models


class QualityFuser:
    """
    params:
    - network_type (str): densefuse or cddfuse
    - fuse_type (str): add, exp, pow, feature or cddfuse
    - with_quality (bool): whether to use quality information
    - with_reval (bool): whether to re-evaluate quality of fused image
    - path (str): path to pretrained weight
    """

    def __int__(
        self,
        network_type: str,
        fuse_type: str,
        with_quality: bool,
        with_reval: bool,
        path: str = None,
    ):
        super(QualityFuser, self).__init__()
        if fuse_type not in ["add", "exp", "pow", "feature", "cddfuse"]:
            raise ValueError(f"unknow fuse type: {fuse_type}")
        if network_type not in ["densefuse", "cddfuse"]:
            raise ValueError(f"unknow network type: {network_type}")
        self.with_reval = with_reval if with_quality else False
        self.use_hybrid = network_type == "cddfuse"
        fuse_type = "add" if not with_quality and fuse_type in ["exp", "pow"] else fuse_type
        self.use_fusion_network = fuse_type in ["feature", "cddfuse"]

        self.encoder = CDDFuseEncoder(with_quality=with_quality) if self.use_hybrid else DenseFuseEncoder(with_quality=with_quality)
        self.decoder = CDDFuseDecoder() if self.use_hybrid else DenseFuseDecoder()
        if fuse_type == "cddfuse":
            self.fuser = CDDFuseFuser()
            self.encoder = CDDFuseEncoder(with_quality=with_quality)
            self.decoder = CDDFuseDecoder()
        elif fuse_type == "feature":
            self.fuser = FeatureFuser(use_hybrid=self.use_hybrid)
        else:
            self.fuser = WeightFuser(strategy=fuse_type)

        self.load(path)

    def load(self, path: str):
        load_model(path, self.encoder, "encoder")
        load_model(path, self.decoder, "decoder")
        if self.use_fusion_network:
            load_model(path, self.fuser, "fuser")

    def to(self, state, device=None):
        modules = [self.encoder, self.decoder, self.fuser] if self.use_fusion_network else [self.encoder, self.decoder]
        for m in modules:
            m.train() if state == "train" else m.eval()
            if device is not None:
                m.to(device)

    def encode(self, input1, input2):
        if self.use_hybrid:
            self.f1_base, self.f1_detail, self.score1 = self.encoder(input1)
            self.f2_base, self.f2_detail, self.score2 = self.encoder(input2)
        else:
            self.f1_base, self.score1 = self.encoder(input1)
            self.f2_base, self.score2 = self.encoder(input2)
            self.f1_detail, self.f2_detail = None, None
        return self.score1, self.score2

    def decode(self):
        output1 = self.decoder(self.f1_base, self.f1_detail) if self.use_hybrid else self.decoder(self.f1_base)
        output2 = self.decoder(self.f2_base, self.f2_detail) if self.use_hybrid else self.decoder(self.f2_base)
        return output1, output2

    def fuse(self):
        if self.use_fusion_network:
            f_base, f_detail = self.fuser(self.f1_base, self.f2_base, self.f1_detail, self.f2_detail)
        else:
            f_base = self.fuser.forward(self.f1_base, self.f2_base, self.score1, self.score2)
            f_detail = self.fuser.forward(self.f1_detail, self.f2_detail, self.score1, self.score2) if self.use_hybrid else None

        fused = self.decoder(f_base, f_detail) if self.use_hybrid else self.decoder(f_base)
        score = self.encoder(fused)[-1] if self.with_reval else None
        return fused, score

    def save(self, path):
        models_state = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
        }
        if self.use_fusion_network:
            models_state["fuser"] = self.fuser.state_dict()
        save_models(path, models_state)


class DeformedFuser(QualityFuser):
    def __init__(
        self,
        network_type: str,
        fuse_type: str = None,
        path: str = None,
        pred_affine_mat: bool = False,
        img_size: tuple = (224, 224),
    ):
        self.has_fuser = fuse_type is not None
        fuse_type = fuse_type if self.has_fuser else "add"
        super().__init__(network_type, fuse_type, False, False, path)
        self.deformer = HybridDeformer(img_size=img_size, pred_affine_mat=pred_affine_mat)

    def load(self, path: str):
        load_model(path, self.encoder, "encoder")
        load_model(path, self.decoder, "decoder")
        load_model(path, self.deformer, "deformer")
        if self.use_fusion_network:
            load_model(path, self.fuser, "fuser")

    def deform(self, moving):
        registered, transformed, flow = self.deformer(moving, self.f1_base, self.f2_base, self.f1_detail, self.f2_detail)
        if self.use_hybrid:
            self.f3_base, self.f3_detail, self.score3 = self.encoder(registered)
        else:
            self.f3_base, self.score3 = self.encoder(registered)
            self.f3_detail = None
        return registered, transformed, flow

    def fuse(self):
        if self.has_fuser:
            if self.use_fusion_network:
                f_base, f_detail = self.fuser(self.f1_base, self.f3_base, self.f1_detail, self.f3_detail)
            else:
                f_base = self.fuser.forward(self.f1_base, self.f3_base, self.score1, self.score3)
                f_detail = self.fuser.forward(self.f1_detail, self.f3_detail, self.score1, self.score3) if self.use_hybrid else None

            fused = self.decoder(f_base, f_detail) if self.use_hybrid else self.decoder(f_base)
            return fused
        else:
            pass

    def save(self, path):
        models_state = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "deformer": self.deformer.state_dict(),
        }
        if self.use_fusion_network:
            models_state["fuser"] = self.fuser.state_dict()
        save_models(path, models_state)


# densefuse
densefuse_offical = QualityFuser(network_type="densefuse", fuse_type="add", with_quality=False, with_reval=False)
densefuse_feature = QualityFuser(network_type="densefuse", fuse_type="feature", with_quality=False, with_reval=False)
densefuse_pow = QualityFuser(network_type="densefuse", fuse_type="pow", with_quality=True, with_reval=False)
densefuse_exp = QualityFuser(network_type="densefuse", fuse_type="exp", with_quality=True, with_reval=False)
densefuse_feature_q = QualityFuser(network_type="densefuse", fuse_type="feature", with_quality=True, with_reval=False)
# cddfuse
cddfuse_offical = QualityFuser(network_type="cddfuse", fuse_type="cddfuse", with_quality=False, with_reval=False)
cddfuse_feature = QualityFuser(network_type="cddfuse", fuse_type="feature", with_quality=False, with_reval=False)
cddfuse_pow = QualityFuser(network_type="cddfuse", fuse_type="pow", with_quality=True, with_reval=False)
cddfuse_exp = QualityFuser(network_type="cddfuse", fuse_type="exp", with_quality=True, with_reval=False)
cddfuse_feature_q = QualityFuser(network_type="cddfuse", fuse_type="feature", with_quality=True, with_reval=False)
# qualityfuse
qualityfuse = QualityFuser(network_type="cddfuse", fuse_type="feature", with_quality=True, with_reval=True)
# hybriddeform
hybriddeform = DeformedFuser(network_type="cddfuse", fuse_type=None)
# deformedfuse
deformedfuse = DeformedFuser(network_type="cddfuse", fuse_type="feature")
