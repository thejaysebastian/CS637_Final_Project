
# Experiment info:


### 3 DenseNet tests, plus two optional:
1. efficient_densenet121_scratch   → implementation effect (memory-efficient)
2. densenet121_scratch             → architecture baseline (no pretraining)
3. densenet121_pretrained          → best-case performance (ImageNet transfer)

Here is a breakdown:

1. Efficient DenseNet from gpleiss repo (i.e., architectures/densenet.py)
   - trained from scratch
   - demonstrates the memory-efficient implementation

1. Torchvision DenseNet-121 (from torchvision)
   - ImageNet pretrained
   - fairer comparison against EuroSat baselines

1. Torchvision DenseNet-121 (from torchvision)
   - trained from scratch as control baseline
   - comparison of DenseNet pretrained on ImageNet vs. EuroSat only

1. ResNet-50 (from torchvision) OPTIONAL
   - ImageNet pretrained
   - EuroSat baseline - reproduction

1. GoogLeNet (from torchvision) OPTIONAL
   - ImageNet pretrained
   - EuroSat baseline - reproduction


| Model Variant                 | Architecture           | Pretraining    | Purpose                         |
| ----------------------------- | ---------------------- | -------------- | ------------------------------- |
| efficient_densenet121_scratch | DenseNet (gpleiss)     | No             | Memory-efficient implementation |
| densenet121_pretrained        | DenseNet (torchvision) | Yes (ImageNet) | Fair comparison                 |
| densenet121_scratch           | DenseNet (torchvision) | No             | Control baseline                |
| resnet50_pretrained           | ResNet-50              | Yes            | EuroSat baseline                |
| googlenet_pretrained          | GoogLeNet              | Yes            | EuroSat baseline                |
