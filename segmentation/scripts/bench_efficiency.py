#!/usr/bin/env python3
"""
GPU efficiency benchmark for segmentation backbones (shared U-Net decoder).
Reports: params, forward GFLOPs (fvcore, batch=1), inference throughput (img/s),
latency (ms/img), and peak GPU memory (MB) at a fixed batch size.

  python scripts/bench_efficiency.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from models import CvT13UNet, ViTBaseUNet, DINOv2UNet, XCiTUNet, DaViTUNet, UNet

try:
    from fvcore.nn import FlopCountAnalysis
    import logging; logging.getLogger("fvcore").setLevel(logging.ERROR)
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

H = 256
BATCH = 16
ITERS = 50
WARMUP = 10

CONFIGS = [
    ("FAD-ViT-UNet (ours)", lambda: CvT13UNet(in_chans=1, num_classes=2, pretrained_path="")),
    ("ViT-B-UNet",          lambda: ViTBaseUNet(in_chans=1, num_classes=2, pretrained=False)),
    ("DINOv2-B-UNet",       lambda: DINOv2UNet(in_chans=1, num_classes=2, pretrained=False)),
    ("XCiT-UNet",           lambda: XCiTUNet(in_chans=1, num_classes=2, pretrained=False)),
    ("DaViT-UNet",          lambda: DaViTUNet(in_chans=1, num_classes=2, pretrained=False)),
    ("U-Net",               lambda: UNet(in_chans=1, num_classes=2)),
]


def main():
    dev = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)} input=1x{H}x{H} batch={BATCH} iters={ITERS}")
    print("-" * 92)
    print(f"{'Model':22s} {'Params(M)':>10s} {'GFLOPs':>8s} {'Thr(img/s)':>11s} {'Lat(ms)':>9s} {'Mem(MB)':>9s}")
    for name, ctor in CONFIGS:
        try:
            m = ctor().to(dev).eval()
        except Exception as e:
            print(f"{name:22s}  BUILD FAILED: {e}")
            continue
        params = sum(p.numel() for p in m.parameters()) / 1e6
        gflops = float("nan")
        if HAS_FVCORE:
            try:
                gflops = FlopCountAnalysis(m, torch.zeros(1, 1, H, H, device=dev)).total() / 1e9
            except Exception:
                pass
        x = torch.randn(BATCH, 1, H, H, device=dev)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        with torch.no_grad():
            for _ in range(WARMUP):
                m(x)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(ITERS):
                m(x)
            torch.cuda.synchronize(); t1 = time.time()
        dt = (t1 - t0) / ITERS
        thr = BATCH / dt
        lat = dt / BATCH * 1000.0
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"{name:22s} {params:10.2f} {gflops:8.2f} {thr:11.1f} {lat:9.2f} {mem:9.0f}")
        del m, x
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
