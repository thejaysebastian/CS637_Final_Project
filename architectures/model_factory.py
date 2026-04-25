# The purpose of the model factory is to hide any differences in architectures of the different NNs being tested. It sets parameters based on model selected.


import torch.nn as nn
from torchvision import models as tv_models

from architectures.densenet import DenseNet # Our densenet.py file from gpleiss

def build_model(model_name, num_classes):
    """
    model_name options:
        - efficient_densenet_scratch
        - densenet121_pretrained
        - densenet121_scratch
        - resnet50_pretrained
        - googlenet_pretrained
    """
    # ---- DenseNet (memory-efficient, from scratch) ----
    if model_name == "efficient_densenet_scratch":
        model = DenseNet(
            growth_rate=32, # how many filters to add to each layer (k in paper)
            block_config=(6, 12, 24, 16), # layers in each pooling block
            num_init_features=64, # Check to make sure this will work - number of filters to learn in first conv
            bn_size=4, # multiplicative factor for number of bottleneck layers
            drop_rate=0, # dropout rate after each dense layer
            num_classes=num_classes, # 10 classes for EuroSat RGB
            efficient=True # True if using checkpointing (efficient model)
        )

    # ---- Torchvision DenseNet (pretrained) ----
    elif model_name == "densenet121_pretrained":
        weights = tv_models.DenseNet121_Weights.DEFAULT
        model = tv_models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # ---- Torchvision DenseNet (scratch) ----
    elif model_name == "densenet121_scratch":
        model = tv_models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # ---- ResNet-50 (pretrained) ----
    elif model_name == "resnet50_pretrained":
        weights = tv_models.ResNet50_Weights.DEFAULT
        model = tv_models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # ---- GoogLeNet (pretrained) ----
    elif model_name == "googlenet_pretrained":
        weights = tv_models.GoogLeNet_Weights.DEFAULT
        model = tv_models.googlenet(weights=weights, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Built model: {model_name}")
    return model