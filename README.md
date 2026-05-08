# Kvasir-Structured: Medical Image Segmentation Framework

A comprehensive framework for training and evaluating deep learning models for polyp segmentation on the Kvasir-SEG dataset. This project supports multiple state-of-the-art segmentation architectures including UNet, BiseNet, BiSeUNet variants, DuckNet, and HardNet.

## Table of Contents

- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Available Models](#available-models)

## Project Overview

This framework provides a structured pipeline for:
- **Data Preparation**: Preprocessing and augmentation of medical imaging datasets
- **Model Training**: Unified training interface for multiple segmentation architectures
- **Evaluation**: Comprehensive metrics and performance evaluation
- **Post-processing**: Calibration and result analysis tools

## Prerequisites

- Python 3.8+
- CUDA 12.x (for GPU support)
- pip package manager
- Linux/Unix environment (bash)

## Installation

### 1. Create Virtual Environment

```bash
cd /media/iffat/DataDrive/Projects/Kvasir-structured
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

The project includes PyTorch with CUDA 12 support, along with essential libraries:
- **Deep Learning**: torch, torchvision
- **Data Processing**: numpy, pandas, opencv-python, pillow
- **Utilities**: matplotlib, tqdm, coloredlogs
- **Advanced Features**: diffusers, onnxruntime, accelerate

### 3. Set PYTHONPATH (Optional but Recommended)

```bash
export PYTHONPATH=$PYTHONPATH:/media/iffat/DataDrive/Projects/Kvasir-structured
```

## Project Structure

```
Kvasir-structured/
├── train/                      # Training module
│   ├── train.py               # Main training script
│   ├── train_combined.py      # Multi-dataset training
│   ├── train_ldpolyp.py       # LDPolyp-specific training
│   ├── configs/               # Training configurations
│   ├── losses/                # Loss functions
│   ├── metrics/               # Evaluation metrics
│   └── models/                # Model architectures
│       ├── unet_*.py
│       ├── bisenetv*.py
│       ├── biseunetv*.py
│       ├── ducknet_model.py
│       ├── hardnet_model.py
│       └── factory.py
├── preprocess/                # Data preprocessing
│   ├── dataset_prep.py        # Dataset utilities
│   └── ldpolyp_prep.py        # LDPolyp preprocessing
├── postprocess/               # Evaluation & post-processing
│   ├── evaluate.py            # Main evaluation script
│   ├── evaluate_all.py        # Batch evaluation
│   ├── calibrate.py           # Probability calibration
│   ├── eval_utils.py          # Evaluation utilities
│   └── calibration/           # Calibration methods
├── scripts/                   # Shell scripts for easy execution
│   ├── run_train_universal.sh # Universal training script
│   ├── run_eval_generic.sh    # Generic evaluation script
│   └── run_train_*.sh         # Model-specific training scripts
├── results/                   # Training results & checkpoints
├── calibration_results/       # Calibration weights
└── requirements.txt           # Python dependencies
```

## Quick Start

### 1. Dataset Preparation

Ensure your Kvasir-SEG dataset is organized as:
```
dataset/
├── images/
├── masks/
└── split.txt
```

### 2. Train a Model

```bash
source .venv/bin/activate

# Using the universal training script
python3 -m train.train \
    --model unet \
    --data-dir /media/iffat/DataDrive/dataset/kvasir-seg/Kvasir-SEG \
    --out-dir results/my_run \
    --epochs 200 \
    --batch 8 \
    --lr 1e-3
```

### 3. Evaluate a Model

```bash
source .venv/bin/activate

python3 -m postprocess.evaluate \
    --model unet \
    --ckpt results/my_run/best.pt \
    --data-dir /media/iffat/DataDrive/dataset/kvasir-seg/Kvasir-SEG \
    --out-dir results/my_run/evaluation
```

## Training

### Basic Training Command

```bash
python3 -m train.train \
    --model <model_name> \
    --data-dir <path_to_dataset> \
    --out-dir <output_directory> \
    [optional arguments]
```

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | **required** | Model architecture (unet, bisenet, biseunetv1-v7, ducknet, hardnet, etc.) |
| `--data-dir` | str | **required** | Path to Kvasir-SEG dataset root directory |
| `--out-dir` | str | **required** | Directory to save training results and checkpoints |
| `--epochs` | int | 200 | Maximum number of training epochs |
| `--batch` | int | 8 | Batch size |
| `--lr` | float | 1e-3 | Learning rate |
| `--img-size` | int | 256 | Input image size (256x256) |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--patience` | int | 10 | Early stopping patience (stop if no improvement) |
| `--mixed` | flag | false | Enable mixed precision training (AMP) |
| `--workers` | int | 4 | Number of data loading workers |
| `--base-ch` | int | 32 | Base channels for model architecture |

### Training Examples

#### Train UNet with default settings
```bash
python3 -m train.train \
    --model unet \
    --data-dir /path/to/Kvasir-SEG \
    --out-dir results/unet_run
```

#### Train BiSeUNetV7 with mixed precision and custom batch size
```bash
python3 -m train.train \
    --model biseunetv7 \
    --data-dir /path/to/Kvasir-SEG \
    --out-dir results/biseunetv7_run \
    --batch 16 \
    --epochs 300 \
    --lr 5e-4 \
    --mixed
```

#### Train on multiple datasets
```bash
python3 -m train.train_combined \
    --data-dir /path/to/Kvasir-SEG \
    --ldpolyp-dir /path/to/LDPolyp \
    --out-dir results/combined_run \
    --model biseunetv7
```

### Output Structure

Training produces:
```
results/my_run/
├── best.pt              # Best model checkpoint (by validation Dice)
├── last.pt              # Latest model checkpoint
├── history.json         # Training history (loss, metrics per epoch)
└── config.json          # Training configuration used
```

## Evaluation

### Basic Evaluation Command

```bash
python3 -m postprocess.evaluate \
    --model <model_name> \
    --ckpt <path_to_checkpoint> \
    --data-dir <path_to_dataset> \
    [optional arguments]
```

### Evaluation Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | **required** | Model architecture |
| `--ckpt` | str | **required** | Path to model checkpoint (.pt file) |
| `--data-dir` | str | **required** | Path to dataset for evaluation |
| `--img-size` | int | 256 | Input image size |
| `--batch` | int | 1 | Batch size (typically 1 for evaluation) |
| `--seed` | int | 42 | Random seed |
| `--mixed` | flag | false | Use automatic mixed precision |
| `--out-dir` | str | None | Directory to save evaluation results |

### Evaluation Examples

#### Evaluate UNet on Kvasir-SEG
```bash
python3 -m postprocess.evaluate \
    --model unet \
    --ckpt results/unet_run/best.pt \
    --data-dir /path/to/Kvasir-SEG
```

#### Evaluate BiSeUNetV7 with results saved
```bash
python3 -m postprocess.evaluate \
    --model biseunetv7 \
    --ckpt results/biseunetv7_run/best.pt \
    --data-dir /path/to/Kvasir-SEG \
    --out-dir results/biseunetv7_run/evaluation \
    --mixed
```

#### Batch evaluation using shell script
```bash
cd /media/iffat/DataDrive/Projects/Kvasir-structured
bash scripts/run_eval_generic.sh unet
```

### Output Metrics

Evaluation produces comprehensive metrics:

| Metric | Description |
|--------|-------------|
| **Dice** | Dice coefficient (F1-score) |
| **IoU** | Intersection over Union (Jaccard index) |
| **Accuracy** | Pixel-level accuracy |
| **Precision** | True positives / (true positives + false positives) |
| **Recall/Sensitivity** | True positives / (true positives + false negatives) |
| **Specificity** | True negatives / (true negatives + false positives) |
| **HD95** | 95th percentile Hausdorff Distance |
| **ASD** | Average Surface Distance |
| **FPS** | Inference speed (frames per second) |
| **Params** | Number of model parameters |
| **MACs** | Multiply-accumulate operations |

### Results Directory Structure

```
results/
├── Kvasir/                 # Kvasir-SEG results
├── ldpolyp/                # LDPolyp results
├── combined/               # Combined dataset results
├── dry_run/                # Test run results
└── <model>_run/
    ├── best.pt
    ├── last.pt
    ├── history.json
    ├── config.json
    └── evaluation/         # (if --out-dir specified)
        ├── metrics.json
        └── predictions/    # Segmentation masks
```

## Available Models

The framework supports the following segmentation architectures:

### UNet Variants
- **unet**: Standard U-Net architecture

### BiSeNet Variants
- **bisenet**: Original BiSeNet
- **bisenetv1**: BiSeNet V1

### BiSeUNet Variants (Hybrid Encoder-Decoder)
- **biseunetv1** through **biseunetv7**: Progressive improvements
- **biseunetv7_new**: Latest version

### Other Architectures
- **ducknet**: DuckNet segmentation model
- **hardnet**: HardNet architecture
- **hardunet**: HardUNet (lightweight variant)
- **bisenetformer**: Vision Transformer-based variant
- **medsegdiff**: Diffusion-based segmentation
- **odise**: Open-vocabulary segmentation

### Model Selection Tips

| Use Case | Recommended Model |
|----------|-------------------|
| General purpose | biseunetv7, biseunetv6 |
| Fast inference | hardnet, ducknet |
| High accuracy | biseunetv7_new |
| Lightweight | hardunet |
| Multi-class | medsegdiff |

## Post-Processing & Calibration

After training, you can calibrate model predictions:

```bash
python3 -m postprocess.calibrate \
    --model <model_name> \
    --ckpt <checkpoint> \
    --data-dir <dataset_path> \
    --method platt  # or 'temperature_scaling'
```

Calibrated weights are saved in `calibration_results/`.

## Troubleshooting

### Virtual Environment Issues
```bash
# Ensure you're in the correct directory
cd /media/iffat/DataDrive/Projects/Kvasir-structured
source .venv/bin/activate

# Verify Python executable
which python3
```

### CUDA/GPU Issues
```bash
# Check GPU availability
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.cuda.get_device_name())"
```

### Missing Dataset
Ensure your dataset path is correct and contains:
- `images/` directory with .jpg/.png files
- `masks/` directory with corresponding mask files
- Optional: `split.txt` file defining train/val/test splits

### Import Errors
```bash
# Set PYTHONPATH if running from different directory
export PYTHONPATH=$PYTHONPATH:/media/iffat/DataDrive/Projects/Kvasir-structured
```

## Performance Benchmarks

Typical training and inference times (on NVIDIA GPU):

| Model | Train Time/Epoch | Inference Time | Memory (GB) |
|-------|------------------|-----------------|------------|
| UNet | ~60s | ~10ms | 2-3 |
| BiSeUNetV7 | ~90s | ~15ms | 3-4 |
| HardNet | ~40s | ~5ms | 1-2 |
| DuckNet | ~70s | ~12ms | 2-3 |

## Citation

If you use this framework in your research, please cite the relevant papers:

- Kvasir-SEG: [Jha et al., 2020]
- BiSeNet: [Yu et al., 2018]
- U-Net: [Ronneberger et al., 2015]

## License

Please refer to the project's license file for usage terms.

## Support

For issues or questions:
1. Check the `scripts/` directory for example usage
2. Review training logs in `results/` directories
3. Verify dataset paths and formats
4. Ensure all dependencies are installed correctly
