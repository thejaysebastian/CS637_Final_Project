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
4. **Split sensitivity** - How does DenseNet-121 performance change as the trainigng/testing split varies from 10/90 to 90/10, following the style of the original EuroSAT paper benchmarks?

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
- 16 GB RAM recommended for training
- Hardware acceleration supports CUDA, Apple, MPS, and CPU. The code automatically selects the best available device in the following order:
CUDA -> MPS -> CPU

### Installation

```bash
git clone <repository-url>
cd CS637_Final_Project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Verify hardware acceleration access:

```bash
python -c "import torch; print('CUDA: ',torch.cuda.is_available()); print('MPS: ', torch.backends.mps.is_available())"
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

### Running split-sensitivity experiments

Use `run_split_experiment.py` to run a single custom train/test split:
```bash 
python run_split_experiment.py --train-fraction 0.8
```

Use `run_all_splits.py` to run all split experiments from 10/90 through 90/10. 
```bash 
python run_all_splits.py 
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
├── run_split_experiment.py     # Runs one custom train/test split
├── run_all_splits.py           # Sequential batch runner (all splits)
│
├── architectures/
│   ├── densenet.py             # Memory-efficient DenseNet (gpleiss implementation)
│   └── model_factory.py        # Unified model builder for all variants
│
├── configs/
|   ├── base_splits.yaml                    # mps defaults for split testing
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
│   ├── config.py               # YAML config loader with base/override merging
|   └── device.py               # Automatic CUDA/MPS/CPU device selection
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

![EuroSAT RGB class samples](assets/eurosat_class_samples.png)
*Sample images for each of the 10 EuroSAT RGB land-cover classes. Green borders indicate consistently high-performing classes; red borders indicate consistently lower-performing classes across all models.*

**Preprocessing:** Images are normalized using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Models trained from scratch on 64×64 inputs receive no resizing. ImageNet-pretrained models receive images resized to 224×224 to match their expected input distribution.

**Split note:** The Hugging Face split (60/20/20) may differ from the split used in the original EuroSAT paper. Results are presented as "DenseNet under comparable conditions" rather than a strict replication.

**Split testing note:** For split-sensitivity experiments, the original Hugging Face train/validation/test partitions are recombined into a single dataset. New stratified train/test splits are then generated using the requested training fraction. A validation subset is taken from the training portion only for model selection.

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

All six models were trained and evaluated on the EuroSAT RGB test set (5,400 images). Hardware: NVIDIA GeForce RTX 4070, 12 GB VRAM, Windows 11. Optimizer: Adam, seed: 42.

### Summary table

Models sorted by F1 (macro).

| Model | Implementation | Pretrained | Image Size | Accuracy | Precision | Recall | F1 (macro) | Train Time (min) |
|---|---|---|---|---|---|---|---|---|
| DenseNet-121 pretrained | torchvision | Yes | 224×224 | **98.96%** | 98.94% | 98.92% | **98.92%** | 37.89 |
| GoogLeNet pretrained | torchvision | Yes | 224×224 | 98.81% | 98.82% | 98.75% | 98.78% | 18.71 |
| ResNet-50 pretrained | torchvision | Yes | 224×224 | 98.75% | 98.75% | 98.73% | 98.73% | 35.79 |
| gpleiss DenseNet-121 (checkpoint) | gpleiss | No | 64×64 | 98.17% | 98.17% | 98.11% | 98.14% | 112.32 |
| gpleiss DenseNet-121 (standard) | gpleiss | No | 64×64 | 97.81% | 97.85% | 97.72% | 97.77% | 99.72 |
| DenseNet-121 scratch | torchvision | No | 64×64 | 96.24% | 96.23% | 96.15% | 96.19% | 40.62 |

### Checkpointing comparison

Same gpleiss DenseNet-121 architecture — only the `efficient` flag differs.

| Model | Accuracy | F1 (macro) | Best Epoch | Avg Epoch Time (s) | Total Time (min) |
|---|---|---|---|---|---|
| gpleiss DenseNet-121 (checkpoint ON) | 98.17% | 98.14% | 30 | 89.84 | 112.32 |
| gpleiss DenseNet-121 (checkpoint OFF) | 97.81% | 97.77% | 37 | 79.76 | 99.72 |

Checkpointing adds approximately 12.6% training overhead (10.08 s/epoch, 12.6 min total) with no meaningful improvement in classification performance. This indicates that gradient checkpointing provides no practical benefit on hardware with sufficient VRAM for this task.

### Pretraining effect — DenseNet-121

Same torchvision implementation, identical architecture.

| Model | Pretrained | Image Size | Epochs | Accuracy | F1 (macro) | Best Epoch | Train Time (min) |
|---|---|---|---|---|---|---|---|
| DenseNet-121 pretrained | Yes (ImageNet) | 224×224 | 30 | 98.96% | 98.92% | 26 | 37.89 |
| DenseNet-121 scratch | No | 64×64 | 75 | 96.24% | 96.19% | 28 | 40.62 |

ImageNet pretraining yields a 2.73% improvement in F1, converges in 30 epochs versus 75, and does so in comparable wall-clock time.

### Per-class F1 scores

| Class | DN-121 pretrained | GoogLeNet | ResNet-50 | gpleiss checkpoint | gpleiss standard | DN-121 scratch |
|---|---|---|---|---|---|---|
| Annual Crop | 0.989 | 0.987 | 0.970 | 0.980 | 0.970 | 0.953 |
| Forest | **0.997** | 0.995 | 0.994 | 0.996 | 0.994 | 0.981 |
| Herbaceous Vegetation | 0.981 | 0.975 | 0.963 | 0.982 | 0.963 | 0.925 |
| Highway | 0.990 | 0.994 | 0.980 | 0.985 | 0.980 | 0.961 |
| Industrial Buildings | 0.994 | 0.992 | 0.981 | 0.990 | 0.981 | 0.980 |
| Pasture | 0.980 | 0.976 | 0.971 | 0.985 | 0.971 | 0.947 |
| Permanent Crop | 0.976 | 0.977 | 0.954 | 0.980 | 0.954 | 0.930 |
| Residential Buildings | **0.996** | 0.995 | 0.991 | 0.994 | 0.991 | 0.990 |
| River | 0.992 | 0.991 | 0.982 | 0.984 | 0.982 | 0.966 |
| Sea/Lake | **0.997** | 0.996 | 0.992 | 0.993 | 0.992 | 0.986 |

Forest, Sea/Lake, and Residential Buildings are the highest-performing classes across all models. Permanent Crop and Herbaceous Vegetation are the lowest-performing classes in every model, reflecting visual similarity between crop and vegetation land-cover types rather than any architectural weakness.

### Split-sensitivity experiments
DenseNet-121 pretrained was also evaluated across multiple stratified train/test splits: 10/90, 20/80, ..., 90/10. This extends the split analysis from the original EuroSAT paper to our DenseNet-based pipeline.

| Train/Test Split | Accuracy | Macro F1 | Precision (Macro) | Recall (Macro) |
|------------------|----------|----------|--------------------|----------------|
| 10/90            | 0.9662   | 0.9653   | 0.9665             | 0.9643         |
| 20/80            | 0.9752   | 0.9744   | 0.9744             | 0.9745         |
| 30/70            | 0.9803   | 0.9797   | 0.9807             | 0.9790         |
| 40/60            | 0.9843   | 0.9837   | 0.9842             | 0.9833         |
| 50/50            | 0.9838   | 0.9832   | 0.9833             | 0.9831         |
| 60/40            | 0.9826   | 0.9818   | 0.9818             | 0.9820         |
| 70/30            | 0.9788   | 0.9780   | 0.9779             | 0.9783         |
| 80/20            | 0.9831   | 0.9826   | 0.9830             | 0.9823         |
| 90/10            | 0.9833   | 0.9827   | 0.9833             | 0.9823         |

Performance improves rapidly as the training fraction increases from 10% to 30%, after which gains plateau, indicating that DenseNet-121 achieves near-optimal performance with relatively limited training data on EuroSAT.
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