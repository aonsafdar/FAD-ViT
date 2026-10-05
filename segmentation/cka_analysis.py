"""
Feature-reuse probe via linear CKA between ImageNet-pretrained and from-scratch
encoders, on BUSI images, for the feature-axis (FAD-ViT) and spatial-axis (ViT) backbones.
Higher CKA(pretrained, scratch) at a given depth => more pretrained structure is reused.
"""
import os, sys, glob
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.argv = ["x"]
ROOT = Path("/path/to/FAD-ViT")
OUT = Path("/path/to/FAD-ViT/Experiments/seg_scratch_transfer/outputs")
WEIGHTS = ROOT / "weights" / "model_best.pth"
FIGDIR = Path(os.environ.get("FIGDIR", "figs"))

from models import CvT13UNet, ViTBaseUNet

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def load_busi(n=128, size=256):
    files = sorted(glob.glob(str(ROOT / "segmentation/data/busi/val/images/*.png")))
    files += sorted(glob.glob(str(ROOT / "segmentation/data/busi/train/images/*.png")))
    files = files[:n]
    imgs = []
    for f in files:
        im = Image.open(f).convert("L").resize((size, size))
        imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
    x = torch.from_numpy(np.stack(imgs))[:, None]  # (N,1,H,W)
    return x


def linear_cka(X, Y):
    # X: (N, Dx), Y: (N, Dy); center columns
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xy = (Y.t() @ X).norm() ** 2
    xx = (X.t() @ X).norm()
    yy = (Y.t() @ Y).norm()
    return (xy / (xx * yy)).item()


def load_sd(path):
    sd = torch.load(path, map_location="cpu")
    for k in ("state_dict", "model", "model_state_dict"):
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]
            break
    return {kk[7:] if kk.startswith("module.") else kk: vv for kk, vv in sd.items()}


@torch.no_grad()
def feat_axis_feats(pretrained, x):
    m = CvT13UNet(in_chans=1, num_classes=2,
                  pretrained_path=str(WEIGHTS) if pretrained else "")
    if not pretrained:
        m.load_state_dict(load_sd(OUT / "BUSI/cvt13_unet/best.pth"), strict=False)
    m.eval().to(dev)
    h = m.encoder(m.input_proj(x.to(dev)))  # [f0,f1,f2] each (B,C,H,W)
    return [f.mean(dim=(2, 3)).cpu() for f in h]  # GAP -> (B,C) per stage


@torch.no_grad()
def spatial_axis_feats(pretrained, x):
    m = ViTBaseUNet(in_chans=1, num_classes=2, pretrained=pretrained)
    if not pretrained:
        m.load_state_dict(load_sd(OUT / "BUSI/vit_base_unet/best.pth"), strict=False)
    m.eval().to(dev)
    xi = torch.nn.functional.interpolate(m.input_proj(x.to(dev)), size=(224, 224),
                                          mode="bilinear", align_corners=False)
    outs = {}
    hooks = []
    depths = [3, 7, 11]
    for d in depths:
        hooks.append(m.backbone.blocks[d].register_forward_hook(
            lambda mod, i, o, d=d: outs.__setitem__(d, o.detach())))
    _ = m.backbone(xi)
    for hh in hooks:
        hh.remove()
    return [outs[d][:, 1:, :].mean(dim=1).cpu() for d in depths]  # mean over tokens -> (B,768)


def main():
    x = load_busi()
    print(f"loaded {x.shape[0]} BUSI images")
    fp = feat_axis_feats(True, x)
    fs = feat_axis_feats(False, x)
    sp = spatial_axis_feats(True, x)
    ss = spatial_axis_feats(False, x)
    cka_feat = [linear_cka(a, b) for a, b in zip(fp, fs)]
    cka_spat = [linear_cka(a, b) for a, b in zip(sp, ss)]
    print("feature-axis CKA (pretrained vs scratch) per stage:", cka_feat)
    print("spatial-axis CKA (pretrained vs scratch) per depth:", cka_spat)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 11, "font.family": "serif",
                         "axes.grid": True, "grid.alpha": 0.3})
    RED, NAVY = "#c0392b", "#2c3e50"
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.7))

    # --- Panel (a): layerwise feature reuse (CKA) ---
    rel = [0.0, 0.5, 1.0]
    axL.plot(rel, cka_feat, "-o", color=RED, lw=2.4, ms=9, label="Feature-axis (FAD-ViT)")
    axL.plot(rel, cka_spat, "--s", color=NAVY, lw=2.0, ms=7, label="Spatial-axis (ViT-B)")
    for x, y in zip(rel, cka_feat):
        axL.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                     fontsize=8.5, color=RED, ha="center", fontweight="bold")
    for x, y in zip(rel, cka_spat):
        axL.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -14),
                     fontsize=8.5, color=NAVY, ha="center")
    axL.axvspan(-0.05, 0.75, color="#2ecc71", alpha=0.06)
    axL.axvspan(0.75, 1.05, color="#e67e22", alpha=0.08)
    axL.text(0.30, axL.get_ylim()[1], "reuse", fontsize=8.5, color="#27ae60",
             va="top", ha="center")
    axL.text(0.93, axL.get_ylim()[1], "specialize", fontsize=8.5, color="#d35400",
             va="top", ha="center")
    axL.set_xticks(rel); axL.set_xticklabels(["early", "mid", "late"])
    axL.set_xlim(-0.08, 1.12)
    axL.set_xlabel("Relative encoder depth")
    axL.set_ylabel("CKA(pretrained, from-scratch) \u2191")
    axL.set_title("(a) Layerwise feature reuse (BUSI)")
    axL.legend(frameon=False, fontsize=9.0, loc="lower left")

    # --- Panel (b): reuse translates into a larger transfer gap (BUSI, held-out test) ---
    # Transfer gap = pretrained - from-scratch Dice, by backbone and capacity.
    labels = ["FAD-ViT\n(feat, 23.1M)", "ViT-S\n(spat, 22.9M)", "ViT-B\n(spat, 90.6M)"]
    gaps = [15.1, 2.5, 5.7]
    colors = [RED, NAVY, NAVY]
    x = np.arange(len(labels)); w = 0.6
    axR.bar(x, gaps, w, color=colors)
    for xi, v, c in zip(x, gaps, colors):
        axR.annotate(f"+{v:.1f}", (xi, v), textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=9.0, color=c,
                     fontweight="bold" if c == RED else "normal")
    axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=8.5)
    axR.set_ylabel("BUSI transfer gap: pretrained \u2212 scratch (Dice)")
    axR.set_title("(b) At matched capacity, feature axis transfers 6\u00d7 more")
    axR.set_ylim(0, max(gaps) * 1.25)

    fig.tight_layout()
    fig.savefig(FIGDIR / "cka.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGDIR / "cka.pdf", bbox_inches="tight")
    print("wrote", FIGDIR / "cka.png")
    print("VALUES", cka_feat, cka_spat)


if __name__ == "__main__":
    main()
