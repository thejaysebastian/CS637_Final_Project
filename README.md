# CS637 Final Project: DenseNet Architecture Evaluation on EuroSAT

A comprehensive study evaluating DenseNet architecture for land-cover classification on the EuroSAT RGB dataset, comparing against ResNet-50 and GoogLeNet baselines.

## Table of Contents

- [Overview](#overview)
- [Research Question](#research-question)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Variants](#model-variants)
- [Results](#results)
- [References](#references)
- [Contributing](#contributing)

## Overview

This project implements and evaluates a memory-efficient PyTorch implementation of DenseNet on the EuroSAT RGB dataset for 10-class land-cover classification. It explores three key aspects of DenseNet performance:

1. **Memory-efficient implementation** - The impact of PyTorch checkpointing on training efficiency
2. **Transfer learning effectiveness** - Pre-training on ImageNet versus training from scratch
3. **Competitive comparison** - Performance against established baselines (ResNet-50, GoogLeNet)

The project follows the experimental methodology from the original EuroSAT paper to ensure fair architectural comparison.

## Research Question

**How does a memory-efficient PyTorch implementation of DenseNet perform on EuroSAT RGB land-cover classification compared with the ResNet-50 and GoogLeNet baselines reported in the EuroSat paper?**

This research question is addressed through systematic experiments that isolate and evaluate:
- Architectural differences between DenseNet and competing models
- The effect of ImageNet pre-training
- The computational efficiency of the memory-optimized implementation

## Key Features

- **Memory-Efficient DenseNet**: Implementation using PyTorch checkpointing (reduces memory from quadratic to linear at ~15-20% training cost)
- **Multiple Model Variants**: Direct comparison between memory-efficient, standard, and pre-trained DenseNet implementations
- **Comprehensive Evaluation**: Tracks accuracy, precision, recall, F1-score, and confusion matrices
- **Configuration-Driven**: YAML-based experiment configuration for easy parameter management
- **Reproducible**: Fixed random seeds and documented training protocols
- **Dataset**: 27,000 hand-labeled RGB satellite images across 10 land-cover classes

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- 8+ GB RAM (16+ GB recommended for training)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CS637_Final_Project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify GPU support (optional)**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### First Experiment

Run a quick experiment with the memory-efficient DenseNet:

```bash
python run_experiment.py --config configs/efficient_densenet_scratch.yaml
```

Expected output:
- Progress updates during training
- Model checkpoint saved to `results/`
- Final evaluation metrics and training summary

## Usage

### Running Experiments

All experiments are configured via YAML files in the `configs/` directory. Each config file overrides base parameters.

**Run a specific model configuration:**
```bash
python run_experiment.py --config configs/<model_config>.yaml
```

**Available model configurations:**
```bash
# DenseNet variants
python run_experiment.py --config configs/efficient_densenet_scratch.yaml
python run_experiment.py --config configs/densenet121_scratch.yaml
python run_experiment.py --config configs/densenet121_pretrained.yaml

# Baseline models (optional)
python run_experiment.py --config configs/resenet50_pretrained.yaml
python run_experiment.py --config configs/googlenet_pretrained.yaml
```

### Configuration Options

Edit `configs/base.yaml` for global defaults, or create model-specific YAML files to override parameters:

```yaml
# Data
dataset: eurosat          # Dataset name
batch_size: 32           # Training batch size
num_workers: 4           # Data loader workers

# Training
epochs: 50               # Number of training epochs
learning_rate: 0.001     # Initial learning rate
optimizer: adam          # Optimizer choice
weight_decay: 0.0001     # L2 regularization

# Scheduler
scheduler: step          # Learning rate schedule
step_size: 20            # Decay every N epochs
gamma: 0.1               # Decay factor

# Loss and device
criterion: cross_entropy  # Loss function
device: cuda             # 'cuda' or 'cpu'
seed: 42                 # Random seed for reproducibility
```

### Output Structure

Each experiment creates timestamped results in `results/`:

```
results
├── experiments/
│   └── <model>_lr<lr>_bs<bs>_<timestamp>/
│       ├── best_model.pt          # Best model weights
│       ├── config.yaml            # Experiment configuration
│       ├── training_summary.yaml  # Final metrics and training log
│       └── checkpoint_epoch_*.pt  # Periodic checkpoints
│
└── logs/
    └── <experiment_name>/     # TensorBoard logs (if enabled)
```

## Project Structure

```
CS637_Final_Project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run_experiment.py            # Main experiment runner
│
├── architectures/
│   ├── densenet.py             # Memory-efficient DenseNet implementation
│   └── model_factory.py        # Model builder with all variants
│
├── configs/
│   ├── base.yaml               # Base configuration
│   ├── efficient_densenet_scratch.yaml
│   ├── densenet121_scratch.yaml
│   ├── densenet121_pretrained.yaml
│   ├── resenet50_pretrained.yaml
│   └── googlenet_pretrained.yaml
│
├── data/
│   └── eurosat.py              # EuroSAT dataset loader
│
├── engine/
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Evaluation and metrics
│   └── metrics.py              # Metric computations
│
├── utils/
│   └── config.py               # Configuration utilities
│
└── results/                     # Experiment outputs (generated)
    ├── checkpoints/
    ├── logs/
    └── tables/
```

## Dataset

### EuroSAT RGB

**Source**: [Hugging Face Datasets](https://huggingface.co/datasets/blanchon/EuroSAT_RGB)

**Specifications**:
- **Total Images**: 27,000 hand-labeled RGB satellite images
- **Resolution**: 64×64 pixels
- **Bands**: 3 (RGB)
- **Classes**: 10 land-cover types
  - Annual Crop
  - Forest
  - Herbaceous Vegetation
  - Highway
  - Industrial Buildings
  - Pasture
  - Permanent Crop
  - Residential Buildings
  - River
  - Sea/Lake

**Split**: 
- Train: 16,200 (60%)
- Validation: 5,400 (20%)
- Test: 5,400 (20%)

### Data Loading

The dataset is automatically downloaded and cached by the `data/eurosat.py` module on first run.

## Model Variants

This project evaluates three primary DenseNet configurations plus optional baselines:

### Primary Experiments

| Model | Architecture | Pre-training | Purpose |
|-------|--------------|--------------|---------|
| `efficient_densenet_scratch` | DenseNet (memory-optimized) | None | Tests PyTorch checkpointing efficiency |
| `densenet121_scratch` | DenseNet-121 (standard) | None | Baseline: architecture without transfer learning |
| `densenet121_pretrained` | DenseNet-121 (standard) | ImageNet | Best-case performance: transfer learning |

### Baseline Models (Optional)

| Model | Architecture | Pre-training | Purpose |
|-------|--------------|--------------|---------|
| `resnet50_pretrained` | ResNet-50 | ImageNet | EuroSAT baseline from original paper |
| `googlenet_pretrained` | GoogLeNet | ImageNet | EuroSAT baseline from original paper |

### DenseNet Architecture Details

**Efficient DenseNet (gpleiss)**:
- Growth rate (k): 32
- Blocks: [6, 12, 24, 16] (DenseNet-121 equivalent)
- Initial features: 64
- Bottleneck size: 4
- Dropout: 0.0
- Checkpointing: Enabled (memory-efficient)

## Results

Experiment results are automatically saved with comprehensive metrics:

- **Training Summary**: `training_summary.yaml` contains:
  - Best validation accuracy and epoch
  - Final test accuracy, precision, recall, F1-score
  - Confusion matrix
  - Training time

- **Model Checkpoints**: Periodic and best-model weights saved for:
  - Reproducibility
  - Ensemble methods
  - Fine-tuning

View results in `results/tables/` for formatted experiment comparisons.

## Important Methodology Notes

### Fair Comparison Considerations

1. **Split Methodology**: The Hugging Face EuroSAT split (16.2k/5.4k/5.4k) may differ from the original paper's methodology. Results are presented as "DenseNet under comparable conditions" unless the original EuroSAT split is reproduced.

2. **Image Resizing**: EuroSAT images (64×64) are resized to model-specific input sizes for ImageNet-pretrained models.

3. **Implementation Efficiency**: The memory-efficient DenseNet uses PyTorch checkpointing, which is an implementation detail (reduces memory usage) rather than an architectural change. Training time is ~15-20% longer due to recomputation.

## References

### Papers

- **DenseNet**: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)  
  Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). CVPR.

- **Memory-Efficient DenseNet**: [Efficient Densely Connected Convolutional Networks with Checkpoint Learning](https://arxiv.org/abs/1707.06990)  
  Pleiss, G., Chen, D., Huang, G., Li, T., van der Maaten, L., & Weinberger, K. Q. (2017).

- **EuroSAT Dataset**: [EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification](https://arxiv.org/abs/1709.00029)  
  Helber, P., Bischke, B., Dengel, A., & Borth, D. (2017).

### Repositories

- [Original DenseNet (Lua)](https://github.com/liuzhuang13/DenseNet)
- [Memory-Efficient DenseNet (PyTorch)](https://github.com/gpleiss/efficient_densenet_pytorch)
- [Original EuroSAT Repository](https://github.com/phelber/EuroSAT)
- [EuroSAT Dataset (Hugging Face)](https://huggingface.co/datasets/blanchon/EuroSAT_RGB)

## Contributing

This is a course project (CS637 Final Project). For inquiries about the code or methodology:

1. Contact the project maintainers

### Code Style

- Python 3.9+
- Type hints recommended for new code
- Follow PEP 8 conventions
- Document configuration parameters in base.yaml

## Authors

Jay ([@thejaysebastian](https://github.com/thejaysebastian))
Mares ([@siroceans](https://github.com/siroceans))

---

**Last Updated**: April 2026
