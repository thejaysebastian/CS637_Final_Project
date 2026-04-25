## experiment info:

Conduct two.

1. Efficient DenseNet from gpleiss repo (i.e., architectures/densenet.py)
   - trained from scratch
   - demonstrates the memory-efficient implementation

2. Torchvision DenseNet-121 (from )
   - ImageNet pretrained
   - fairer comparison against EuroSat baselines

| Model Variant                 | Architecture           | Pretraining    | Purpose                         |
| ----------------------------- | ---------------------- | -------------- | ------------------------------- |
| efficient_densenet121_scratch | DenseNet (gpleiss)     | No             | Memory-efficient implementation |
| densenet121_pretrained        | DenseNet (torchvision) | Yes (ImageNet) | Fair comparison                 |
| densenet121_scratch           | DenseNet (torchvision) | No             | Control baseline                |
| resnet50_pretrained           | ResNet-50              | Yes            | EuroSat baseline                |
| googlenet_pretrained          | GoogLeNet              | Yes            | EuroSat baseline                |
