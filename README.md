# CS637 Final Project: DenseNet on EuroSAT RGB

A multi-axis experimental evaluation of DenseNet for land-cover classification on the EuroSAT RGB dataset, comparing implementation efficiency, the effect of ImageNet pretraining, and performance against the ResNet-50 and GoogLeNet baselines from the original EuroSAT paper.

## Authors

Jay Sebastian ([@thejaysebastian](https://github.com/thejaysebastian)) · Mares Zamora ([@siroceans](https://github.com/siroceans))

---

## Table of Contents

- [Research Questions](#research-questions)
- [Experimental Design](#experimental-design)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Variants](#model-variants)
- [Results](#results)
- [Methodology Notes](#methodology-notes)
- [References](#references)

---

## Research Questions

This project addresses three experimental axes:

1. **Implementation efficiency** — Does gradient checkpointing in the memory-efficient DenseNet implementation affect classification performance, and what is the computational cost?
2. **Pretraining effect** — How much does ImageNet pretraining improve DenseNet performance on EuroSAT compared to training from scratch?
3. **Architecture comparison** — How does DenseNet perform against the ResNet-50 and GoogLeNet baselines reported in the original EuroSAT paper?

---

## Experimental Design

Six models are evaluated in total. Each experiment is controlled for dataset, preprocessing, optimizer, and random seed. The only variables changed between experiments are the model architecture, implementation, and pretraining status.

| Model | Implementation | Pretraining | Image Size | Purpose |
|---|---|---|---|---|
| `gpleiss_densenet121_checkpoint` | gpleiss | None | 64×64 | Checkpointing ON — memory-efficient variant |
| `gpleiss_densenet121_standard` | gpleiss | None | 64×64 | Checkpointing OFF — isolates checkpointing overhead |
| `densenet121_scratch` | torchvision | None | 64×64 | Architecture baseline, no transfer learning |
| `densenet121_pretrained` | torchvision | ImageNet | 224×224 | Best-case DenseNet with transfer learning |
| `resnet50_pretrained` | torchvision | ImageNet | 224×224 | EuroSAT paper baseline |
| `googlenet_pretrained` | torchvision | ImageNet | 224×224 | EuroSAT paper baseline |

**Note on the gpleiss vs. torchvision comparison:** The two DenseNet-121 implementations share the same block configuration `(6, 12, 24, 16)`, growth rate `32`, and initial features `64`, but are independent codebases. Comparisons between them should be interpreted as implementation-level comparisons rather than exact architecture-matched comparisons.

---

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA 11.8+ recommended
- 16 GB RAM recommended for training

### Installation

```bash
git clone <repository-url>
cd CS637_Final_Project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Verify GPU access:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Run a single experiment

```bash
python run_experiment.py --config configs/gpleiss_densenet121_standard.yaml
```

### Run all experiments sequentially

```bash
python run_all.py
```

This runs all six model configurations in sequence and prints total elapsed time on completion.

---

## Usage

### Running experiments

All experiments are driven by YAML config files. Each model config overrides only the parameters that differ from `configs/base.yaml`.

```bash
python run_experiment.py --config configs/<model_config>.yaml
```

### Running multiple experiments

Use `run_all.py` to queue all six experiments sequentially:

```bash
python run_all.py
```

Or use `run_gpleiss_all.py` to run only the two gpleiss DenseNet variants:

```bash
python run_gpleiss_all.py
```

### Configuration

Edit `configs/base.yaml` for shared defaults. Override per-experiment parameters in the individual model YAML files.

Key base parameters:

```yaml
dataset: eurosat
batch_size: 32
num_workers: 4
epochs: 50
learning_rate: 0.001
optimizer: adam
weight_decay: 0.0001
scheduler: step
step_size: 20
gamma: 0.1
criterion: cross_entropy
device: cuda
seed: 42
```

### Output structure

Each experiment produces a timestamped folder under `results/experiments/`:

```
results/
└── experiments/
    └── GOLDEN_<model>_lr<lr>_bs<bs>_<timestamp>/
        ├── best_model.pt           # Weights at best validation accuracy
        ├── final_model.pt          # Weights at final epoch
        ├── config.yaml             # Exact config used for this run
        ├── training_summary.yaml   # Best epoch, final loss/acc, timing
        ├── epoch_log.csv           # Per-epoch loss, accuracy, LR, time
        └── metrics.json            # Test accuracy, precision, recall, F1,
                                    # confusion matrix, per-class report
```

---

## Project Structure

```
CS637_Final_Project/
├── README.md
├── requirements.txt
├── run_experiment.py           # Main experiment driver
├── run_all.py                  # Sequential batch runner (all 6 models)
├── run_gpleiss_all.py          # Sequential batch runner (gpleiss variants only)
│
├── architectures/
│   ├── densenet.py             # Memory-efficient DenseNet (gpleiss implementation)
│   └── model_factory.py        # Unified model builder for all variants
│
├── configs/
│   ├── base.yaml                           # Shared defaults
│   ├── gpleiss_densenet121_checkpoint.yaml # gpleiss, efficient=True
│   ├── gpleiss_densenet121_standard.yaml   # gpleiss, efficient=False
│   ├── densenet121_scratch.yaml            # torchvision, no pretraining
│   ├── densenet121_pretrained.yaml         # torchvision, ImageNet pretrained
│   ├── resnet50_pretrained.yaml            # ResNet-50, ImageNet pretrained
│   └── googlenet_pretrained.yaml           # GoogLeNet, ImageNet pretrained
│
├── data/
│   └── eurosat.py              # EuroSAT dataset loader and transforms
│
├── engine/
│   ├── train.py                # Training loop, checkpointing, logging
│   ├── evaluate.py             # Test set evaluation
│   └── metrics.py              # Accuracy, precision, recall, F1, confusion matrix
│
├── utils/
│   └── config.py               # YAML config loader with base/override merging
│
└── results/                    # Generated at runtime — not committed
    └── experiments/
```

---

## Dataset

**EuroSAT RGB** — 27,000 hand-labeled satellite images across 10 land-cover classes.

| Property | Value |
|---|---|
| Source | [Hugging Face: blanchon/EuroSAT_RGB](https://huggingface.co/datasets/blanchon/EuroSAT_RGB) |
| Image size | 64×64 pixels, 3-channel RGB |
| Classes | 10 |
| Train split | 16,200 images (60%) |
| Validation split | 5,400 images (20%) |
| Test split | 5,400 images (20%) |

**Classes:** Annual Crop · Forest · Herbaceous Vegetation · Highway · Industrial Buildings · Pasture · Permanent Crop · Residential Buildings · River · Sea/Lake

**Preprocessing:** Images are normalized using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Models trained from scratch on 64×64 inputs receive no resizing. ImageNet-pretrained models receive images resized to 224×224 to match their expected input distribution.

**Split note:** The Hugging Face split (60/20/20) may differ from the split used in the original EuroSAT paper. Results are presented as "DenseNet under comparable conditions" rather than a strict replication.

---

## Model Variants

### DenseNet architecture details

All DenseNet variants use the DenseNet-121 block configuration:

| Parameter | Value |
|---|---|
| Growth rate (k) | 32 |
| Block config | (6, 12, 24, 16) |
| Initial features | 64 |
| Bottleneck size | 4 |
| Dropout | 0.0 |
| Classes | 10 |

### Gradient checkpointing

The gpleiss implementation supports an `efficient` flag that enables gradient checkpointing. When `efficient=True`, intermediate feature maps are discarded during the forward pass and recomputed during backpropagation. This reduces memory consumption from O(L²) to O(L) at the cost of additional compute time. One of the goals of this project is to quantify that tradeoff on the EuroSAT task with the available hardware (NVIDIA RTX 4070, 12 GB VRAM).

### Small input stem

The gpleiss DenseNet uses an ImageNet-style stem (`7×7 conv, stride 2` → `3×3 maxpool, stride 2`), which downsamples 64×64 inputs to 16×16 before the first dense block. This is consistent with the torchvision DenseNet-121 stem and keeps the two implementations comparable, but it is worth noting that this stem was designed for larger ImageNet inputs.

---

## Results

> Results will be updated upon completion of all six experimental runs.

### Summary table

| Model | Implementation | Pretrained | Image Size | Accuracy | Precision | Recall | F1 (macro) | Train Time (min) |
|---|---|---|---|---|---|---|---|---|
| gpleiss DenseNet-121 (checkpoint) | gpleiss | No | 64×64 | — | — | — | — | — |
| gpleiss DenseNet-121 (standard) | gpleiss | No | 64×64 | — | — | — | — | — |
| DenseNet-121 scratch | torchvision | No | 64×64 | — | — | — | — | — |
| DenseNet-121 pretrained | torchvision | Yes | 224×224 | — | — | — | — | — |
| ResNet-50 pretrained | torchvision | Yes | 224×224 | — | — | — | — | — |
| GoogLeNet pretrained | torchvision | Yes | 224×224 | — | — | — | — | — |

### Checkpointing comparison

| Model | Accuracy | F1 (macro) | Avg Epoch Time (s) | Total Time (min) |
|---|---|---|---|---|
| gpleiss DenseNet-121 (checkpoint ON) | — | — | — | — |
| gpleiss DenseNet-121 (checkpoint OFF) | — | — | — | — |

### Per-class F1 scores

| Class | gpleiss standard | DenseNet-121 scratch | DenseNet-121 pretrained | ResNet-50 | GoogLeNet |
|---|---|---|---|---|---|
| Annual Crop | — | — | — | — | — |
| Forest | — | — | — | — | — |
| Herbaceous Vegetation | — | — | — | — | — |
| Highway | — | — | — | — | — |
| Industrial Buildings | — | — | — | — | — |
| Pasture | — | — | — | — | — |
| Permanent Crop | — | — | — | — | — |
| Residential Buildings | — | — | — | — | — |
| River | — | — | — | — | — |
| Sea/Lake | — | — | — | — | — |

---

## Methodology Notes

- **Optimizer:** Adam with weight decay 1e-4 for all models. Learning rate 0.001 for scratch models, 0.0001 for pretrained (fine-tuning).
- **Scheduler:** StepLR — learning rate reduced by factor 0.1 every 20 epochs for scratch models (75 epochs), every 10 epochs for pretrained models (30 epochs).
- **Best model selection:** The checkpoint with the highest validation accuracy is saved and used for final test evaluation.
- **Reproducibility:** All runs use `seed: 42`. Results may vary slightly across hardware due to non-deterministic CUDA operations.
- **Checkpointing:** Periodic epoch checkpoints are saved during training and deleted on successful run completion. Only `best_model.pt` and `final_model.pt` are retained.
- **Hardware:** All experiments run on a single NVIDIA GeForce RTX 4070 (12 GB VRAM) under Windows 11.

---

## References

### Papers

- **DenseNet:** [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) — Huang et al., CVPR 2017
- **Memory-Efficient DenseNet:** [Memory-Efficient Implementation of DenseNets](https://arxiv.org/pdf/1707.06990.pdf) — Pleiss et al., 2017
- **EuroSAT:** [EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification](https://arxiv.org/abs/1709.00029) — Helber et al., 2017

### Repositories

- [Memory-Efficient DenseNet (PyTorch) — gpleiss](https://github.com/gpleiss/efficient_densenet_pytorch)
- [Original DenseNet (Lua) — liuzhuang13](https://github.com/liuzhuang13/DenseNet)
- [Original EuroSAT repository — phelber](https://github.com/phelber/EuroSAT)
- [EuroSAT RGB dataset — Hugging Face](https://huggingface.co/datasets/blanchon/EuroSAT_RGB)

---

*CS637 Final Project · Spring 2026*
