"""
CvT-13 encoder for segmentation. Returns multi-scale features for U-Net decoder.
Uses mod_cvt (cvt13-mod) to match weights/model_best.pth. Loads pretrained ImageNet weights.
"""
import importlib.util
import sys
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

# Load mod_cvt from BU-Mamba (matches model_best.pth). Inject BU-Mamba models into sys.modules
# so mod_cvt's "from models.registry import" resolves correctly.
BU_MAMBA = Path(__file__).resolve().parents[2] / "BU-Mamba"
_bu_models_path = BU_MAMBA / "models"
if str(BU_MAMBA) not in sys.path:
    sys.path.insert(0, str(BU_MAMBA))

_orig_models = sys.modules.get("models")
_orig_registry = sys.modules.get("models.registry")
import importlib.util as _iu
_reg_spec = _iu.spec_from_file_location("models.registry", _bu_models_path / "registry.py")
_reg = _iu.module_from_spec(_reg_spec)
_reg_spec.loader.exec_module(_reg)
_bu_models = type(sys)("models")
_bu_models.registry = _reg
sys.modules["models"] = _bu_models
sys.modules["models.registry"] = _reg

# Load as models.mod_cvt so mod_cvt's "from .registry import register_model" works
_mod_cvt_path = _bu_models_path / "mod_cvt.py"
_spec = importlib.util.spec_from_file_location("models.mod_cvt", _mod_cvt_path)
_mod_cvt = importlib.util.module_from_spec(_spec)
sys.modules["models.mod_cvt"] = _mod_cvt
_spec.loader.exec_module(_mod_cvt)

if _orig_models is not None:
    sys.modules["models"] = _orig_models
if _orig_registry is not None:
    sys.modules["models.registry"] = _orig_registry

ConvEmbed = _mod_cvt.ConvEmbed
VisionTransformer = _mod_cvt.VisionTransformer
QuickGELU = _mod_cvt.QuickGELU
LayerNorm = _mod_cvt.LayerNorm
trunc_normal_ = _mod_cvt.trunc_normal_

CVT13_MOD_SPEC = {
    "INIT": "trunc_norm",
    "NUM_STAGES": 3,
    "PATCH_SIZE": [7, 3, 3],
    "PATCH_STRIDE": [4, 2, 2],
    "PATCH_PADDING": [2, 1, 1],
    "DIM_EMBED": [64, 192, 384],
    "NUM_HEADS": [1, 1, 1],
    "DEPTH": [1, 2, 10],
    "MLP_RATIO": [4.0, 4.0, 4.0],
    "ATTN_DROP_RATE": [0.0, 0.0, 0.0],
    "DROP_RATE": [0.0, 0.0, 0.0],
    "DROP_PATH_RATE": [0.0, 0.0, 0.1],
    "QKV_BIAS": [True, True, True],
    "CLS_TOKEN": [False, False, False],
    "POS_EMBED": [False, False, False],
    "QKV_PROJ_METHOD": ["dw_bn", "dw_bn", "dw_bn"],
    "KERNEL_QKV": [3, 3, 3],
    "PADDING_KV": [1, 1, 1],
    "STRIDE_KV": [1, 1, 1],
    "PADDING_Q": [1, 1, 1],
    "STRIDE_Q": [1, 1, 1],
    "SA_TAIL_DEPTH": [0, 0, 5],
    "SA_TAIL_NUM_HEADS": [1, 1, 4],
}


class CvT13Encoder(nn.Module):
    """
    CvT-13 encoder that returns multi-scale features for segmentation.
    Outputs: [feat_stage0, feat_stage1, feat_stage2] with dims [64, 192, 384]
    and spatial strides [4, 8, 16] relative to input.
    """

    def __init__(self, in_chans=1, pretrained_path=""):
        super().__init__()
        self.spec = CVT13_MOD_SPEC
        self.num_stages = self.spec["NUM_STAGES"]
        in_c = in_chans
        for i in range(self.num_stages):
            kwargs = {
                "patch_size": self.spec["PATCH_SIZE"][i],
                "patch_stride": self.spec["PATCH_STRIDE"][i],
                "patch_padding": self.spec["PATCH_PADDING"][i],
                "embed_dim": self.spec["DIM_EMBED"][i],
                "depth": self.spec["DEPTH"][i],
                "num_heads": self.spec["NUM_HEADS"][i],
                "mlp_ratio": self.spec["MLP_RATIO"][i],
                "qkv_bias": self.spec["QKV_BIAS"][i],
                "drop_rate": self.spec["DROP_RATE"][i],
                "attn_drop_rate": self.spec["ATTN_DROP_RATE"][i],
                "drop_path_rate": self.spec["DROP_PATH_RATE"][i],
                "with_cls_token": self.spec["CLS_TOKEN"][i],
                "method": self.spec["QKV_PROJ_METHOD"][i],
                "kernel_size": self.spec["KERNEL_QKV"][i],
                "padding_q": self.spec["PADDING_Q"][i],
                "padding_kv": self.spec["PADDING_KV"][i],
                "stride_kv": self.spec["STRIDE_KV"][i],
                "stride_q": self.spec["STRIDE_Q"][i],
                "sa_tail_depth": self.spec["SA_TAIL_DEPTH"][i],
                "sa_tail_heads": self.spec["SA_TAIL_NUM_HEADS"][i],
            }
            stage = VisionTransformer(
                in_chans=in_c,
                init=self.spec["INIT"],
                act_layer=QuickGELU,
                norm_layer=partial(LayerNorm, eps=1e-5),
                **kwargs,
            )
            setattr(self, f"stage{i}", stage)
            in_c = self.spec["DIM_EMBED"][i]

        self.dims = self.spec["DIM_EMBED"]
        if pretrained_path:
            self._load_pretrained(pretrained_path, in_chans)

    def _load_pretrained(self, path, in_chans):
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[7:]
            if "head" in k or k.endswith("fc.weight") or k.endswith("fc.bias"):
                continue
            # Handle 3-channel pretrained with 1-channel input: average RGB to 1 ch
            # No kernel averaging: encoder always receives 3ch from InputProjection
            new_sd[k] = v
        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        print(f"[CvT13Encoder] Loaded pretrained: missing={len(missing)} unexpected={len(unexpected)}")

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns list of features: [f0, f1, f2] with shapes
        f0: (B, 64, H/4, W/4), f1: (B, 192, H/8, W/8), f2: (B, 384, H/16, W/16)
        """
        feats = []
        for i in range(self.num_stages):
            x, _ = getattr(self, f"stage{i}")(x)
            feats.append(x)
        return feats
