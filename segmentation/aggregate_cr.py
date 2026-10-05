#!/usr/bin/env python3
"""
Aggregate camera-ready sweep results into 3-seed means and transfer gaps.

Reads:
  outputs/axis_transfer/results_ms.csv   (FAD-ViT arms: channel/spatial/spatial_matched, tail def/0)
  outputs/axis_transfer/vits_ms.csv      (ViT-S/16 matched-capacity baseline)

Prints:
  - matched-capacity gap table (feature-axis vs ViT-S) per task
  - 2x2 tail on/off gap table on representative tasks
De-duplicates by (task,arm,init,seed,tail) keeping the last occurrence.
"""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

BASE = Path(__file__).resolve().parent / "outputs" / "axis_transfer"
RES = BASE / "results_ms.csv"
VRES = BASE / "vits_ms.csv"

TASKS = ["BUSI", "Task01_BrainTumour", "Task02_Heart", "Task03_Liver",
         "Task04_Hippocampus", "Task05_Prostate", "Task06_Lung",
         "Task07_Pancreas", "Task08_HepaticVessel", "Task09_Spleen", "Task10_Colon"]
MOD = {"BUSI": "US", "Task01_BrainTumour": "MRI", "Task02_Heart": "MRI",
       "Task03_Liver": "CT", "Task04_Hippocampus": "MRI", "Task05_Prostate": "MRI",
       "Task06_Lung": "CT", "Task07_Pancreas": "CT", "Task08_HepaticVessel": "CT",
       "Task09_Spleen": "CT", "Task10_Colon": "CT"}


def load():
    # key -> {seed: dice}
    d = defaultdict(dict)
    if RES.exists():
        for r in csv.reader(open(RES)):
            if not r or r[0] != "RESULT":
                continue
            # RESULT,task,tmode,init,seed,tail,dice,iou,hd
            _, task, tm, init, seed, tail, dice = r[0], r[1], r[2], r[3], r[4], r[5], r[6]
            try:
                d[(task, tm, init, tail)][int(seed)] = float(dice)
            except ValueError:
                pass
    if VRES.exists():
        for r in csv.reader(open(VRES)):
            if not r or r[0] != "RESULT":
                continue
            # RESULT,task,vits,init,seed,dice,iou,hd
            _, task, tm, init, seed, dice = r[0], r[1], r[2], r[3], r[4], r[5]
            try:
                d[(task, "vits", init, "def")][int(seed)] = float(dice)
            except ValueError:
                pass
    return d


def ms(d, key):
    v = list(d.get(key, {}).values())
    if not v:
        return None
    return (mean(v), pstdev(v) if len(v) > 1 else 0.0, len(v))


def fmt(m):
    return f"{m[0]:.1f}±{m[1]:.1f}(n={m[2]})" if m else "  --  "


def gap(d, task, arm, tail="def"):
    s = ms(d, (task, arm, "scratch", tail))
    p = ms(d, (task, arm, "pretrained", tail))
    if s and p:
        return s, p, p[0] - s[0]
    return s, p, None


def main():
    d = load()
    print("\n== #1 Matched-capacity gap (feature-axis cvt13 vs ViT-S/16) ==")
    print(f"{'Task':22} {'Mod':3} | {'FAD scr':14} {'FAD pt':14} {'gap':6} | "
          f"{'ViT-S scr':14} {'ViT-S pt':14} {'gap':6} | dGap")
    for t in TASKS:
        fs, fp, fg = gap(d, t, "channel")
        vs, vp, vg = gap(d, t, "vits")
        dg = f"{fg-vg:+.1f}" if (fg is not None and vg is not None) else "  -- "
        print(f"{t:22} {MOD[t]:3} | {fmt(fs):14} {fmt(fp):14} "
              f"{('%+.1f'%fg) if fg is not None else '  -- ':6} | "
              f"{fmt(vs):14} {fmt(vp):14} "
              f"{('%+.1f'%vg) if vg is not None else '  -- ':6} | {dg}")

    print("\n== #2 2x2 spatial-extractor ablation (gap = pt - scr) ==")
    print(f"{'Task':14} {'Arm':9} | {'tail-on gap':12} {'tail-off gap':12}")
    for t in ["BUSI", "Task02_Heart", "Task09_Spleen"]:
        for arm in ["channel", "spatial"]:
            _, _, gon = gap(d, t, arm, "def")
            _, _, goff = gap(d, t, arm, "0")
            print(f"{t:14} {arm:9} | "
                  f"{('%+.1f'%gon) if gon is not None else '  -- ':12} "
                  f"{('%+.1f'%goff) if goff is not None else '  -- ':12}")
    print()


if __name__ == "__main__":
    main()
