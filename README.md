# FAD-ViT: Feature-Axis Attention as a Transfer Prior for Medical Imaging

Official code for **"Learning Features, Not Layouts: Feature-Axis Attention as a Transfer Prior for Medical Imaging"** (ACCV 2026).

FAD-ViT is a strictly **axis-decoupled, channel-primary** Vision Transformer: depthwise
operators carry all *spatial* structure while single-head self-attention over **feature
(channel) tokens** performs all *global* mixing, so no operator both extracts space and
mixes channels. Treating the **axis of global attention** (spatial vs. feature) as a
controlled variable, we find that making it channel-primary leaves from-scratch
performance essentially unchanged but markedly strengthens transfer from ImageNet to
medical imaging — largest where texture rather than geometry is discriminative — at a
fraction of the parameters and compute of standard spatial transformers.

> Code and configurations for all experiments in the paper. Pretrained weights are
> released separately (see [Pretrained weights](#pretrained-weights)).

---

## Repository structure

```
FAD-ViT/
├── backbone/          # Core FAD-ViT model (classification backbone)
│   ├── fadvit_backbone.py           # axis-decoupled, channel-primary ViT
│   ├── fadvit_encoder.py     # encoder wrapper (loads optional pretrained weights)
│   └── fadvit_unet.py        # FAD-ViT-UNet (encoder + U-Net decoder)
├── segmentation/      # Segmentation transfer (BUSI + 10 MSD-derived 2D tasks)
│   ├── train.py             # unified training entry point (all backbones)
│   ├── eval_test.py         # held-out test evaluation
│   ├── metrics.py           # Dice / mIoU / HD95
│   ├── cka_analysis.py      # centered kernel alignment (feature reuse)
│   ├── aggregate_cr.py      # 3-seed aggregation of results
│   ├── models/              # FAD-ViT-UNet + baselines (ViT-S/B, DINOv2, XCiT, DaViT, ResNet50, U-Net)
│   ├── data/                # dataset loaders (data payload not included)
│   ├── scripts/             # data download/preprocessing, benchmarking, visualization
│   └── *.slurm, submit_*.sh # cluster job scripts (edit paths before use)
└── classification/    # Low-data linear probing on MedMNIST-2D
    ├── run_linear_probe_classification.py
    ├── cls_ms.slurm, submit_cls_ms.sh, cls_aggregate_ms.py   # multi-seed sweep + aggregation
    └── *.slurm, *.py        # locked-LR protocol, tables, plots
```

---

## Installation

```bash
git clone https://github.com/aonsafdar/FAD-ViT.git
cd FAD-ViT
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r segmentation/requirements.txt
# key deps: torch, torchvision, timm, einops, numpy, scipy, scikit-learn,
#           nibabel (MSD), medpy (HD95), matplotlib, medmnist (classification)
pip install medmnist
```

Tested with PyTorch ≥ 1.10 on NVIDIA V100 / A100 / L40S / H100.

---

## Pretrained weights

The ImageNet-1K pretrained FAD-ViT checkpoint (~85 MB) is released separately via
GitHub Releases / Hugging Face. Point the training code at it:

```bash
# segmentation
python segmentation/train.py --model fadvit_unet --fadvit-weights /path/to/model_best.pth ...
# backbone
python -c "from backbone.fadvit_unet import FADViTUNet; FADViTUNet(pretrained_path='/path/to/model_best.pth')"
```

Use `--scratch` to train the FAD-ViT backbone from random initialization (the from-scratch control).

---

## Data

**Segmentation.**
```bash
# 1) Download MSD tasks
python segmentation/scripts/download_msd.py --tasks Task01_BrainTumour Task02_Heart ... --root ./data/msd
# 2) Slice volumes to 2D (patient-level 70/15/15 split, no leakage)
python segmentation/scripts/process_msd.py --root ./data/msd --output ./data/msd_2d --tasks <...>
# 3) Breast ultrasound (BUSI) segmentation
python segmentation/scripts/prepare_busi_seg.py --root /path/to/Dataset_BUSI_with_GT --output ./data/busi
```

**Classification.** MedMNIST-2D datasets are fetched via the `medmnist` package (exported
to `ImageFolder` layout at 224×224); see `classification/README.md`.

---

## Reproducing the paper

### Segmentation transfer (the global-attention axis is the only variable)

```bash
cd segmentation
# FAD-ViT (feature axis, default) — pretrained transfer
python train.py --model fadvit_unet --token-mode channel --task Task02_Heart \
    --data ./data/msd_2d --fadvit-weights /path/to/model_best.pth --epochs 100 --early-stop 15
# spatial-axis ablation (same skeleton, attention over spatial tokens)
python train.py --model fadvit_unet --token-mode spatial  ...
# single-factor control: identical channel-mode Q/K/V, only the pooling axis flipped
python train.py --model fadvit_unet --token-mode spatial_matched ...
# matched-capacity spatial baseline and other encoders
python train.py --model vit_small_unet ...      # ViT-S/16 (22.9M, matched)
python train.py --model {vit_base_unet,dinov2_unet,xcit_unet,davit_unet,resnet50_unet,unet} ...
# disable the spatial-attention tail (23.1M -> 16.4M)
python train.py --model fadvit_unet --sa-tail-depth-last 0 ...
# held-out test evaluation
python eval_test.py --model fadvit_unet --task Task02_Heart --checkpoint outputs/.../best.pth
```

`--token-mode {channel|spatial|spatial_matched}` selects the attention axis;
`--scratch` gives the from-scratch control; the transfer gap is `pretrained − scratch`.

**Camera-ready sweeps** (3 seeds, unified controlled protocol):
`submit_cr.sh` (matched-capacity + 2×2 tail ablation), `submit_cr_baselines.sh`
(XCiT/DaViT across all tasks), aggregated by `aggregate_cr.py`. `cka_analysis.py`
reproduces the feature-reuse (CKA) analysis.

### Low-data linear probing (MedMNIST-2D)

```bash
cd classification
# multi-seed sweep: 3 models (FAD-ViT, ViT-S/16, DINOv2-B) x 3 fractions (1/10/100%)
bash submit_cls_ms.sh
python cls_aggregate_ms.py           # 3-seed mean +/- sd tables
```

---

## Citation

```bibtex
@inproceedings{safdar2026fadvit,
  title     = {Learning Features, Not Layouts: Feature-Axis Attention as a Transfer Prior for Medical Imaging},
  author    = {Safdar, Aon and Saadeldin, Mohamed},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE).

## Acknowledgement

This publication has emanated from research conducted with the financial support of
Taighde Éireann – Research Ireland under Grant number 18/CRT/6183.
