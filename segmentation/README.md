# Medical Segmentation Pipeline (MSD)

2D semantic segmentation on Medical Segmentation Decathlon (MSD) using:
- **Proposed**: CvT13-mod encoder + U-Net decoder (weights: `cvt13_imnet1k_mod.pth`)
- **Baselines**: U-Net, nnU-Net

## Tasks
- **Task01_BrainTumour** (MRI) - Brain tumour, 4 classes
- **Task07_Pancreas** (CT) - Pancreas and tumour, 3 classes

## Setup
```bash
cd /path/to/FAD-ViT/segmentation
pip install -r requirements.txt
# medpy for HD95; nnunetv2 for nnU-Net baseline
```

## Usage

### 1. Download MSD data
```bash
python scripts/download_msd.py --tasks Task01_BrainTumour Task07_Pancreas --root ./data/msd --method urllib
# Or: --method aws (requires AWS CLI)
```

### 2. Process to 2D slices
```bash
python scripts/process_msd.py --root ./data/msd --tasks Task01_BrainTumour Task07_Pancreas --output ./data/msd_2d
```

### 3. Train
```bash
# CvT13-UNet (proposed, uses ../weights/cvt13_imnet1k_mod.pth)
python train.py --model cvt13_unet --task Task01_BrainTumour --data ./data/msd_2d --num-classes 4 --epochs 100

# U-Net baseline
python train.py --model unet --task Task01_BrainTumour --data ./data/msd_2d --num-classes 4 --epochs 100
```

### 4. Evaluate & visualize overlays
```bash
python evaluate.py --model cvt13_unet --task Task01_BrainTumour --checkpoint outputs/seg/Task01_BrainTumour/cvt13_unet/best.pth --num-classes 4 --save-overlays
```

### 5. nnU-Net baseline (3D, optional)
```bash
pip install nnunetv2
python scripts/run_nnunet_baseline.py --task Task07_Pancreas
# Then run nnUNetv2_plan_and_preprocess, nnUNetv2_train as instructed
```

## SLURM
```bash
sbatch msd_seg.slurm
```

## Metrics
- **Dice** (DSC)
- **HD95** (95th percentile Hausdorff Distance, requires medpy)
