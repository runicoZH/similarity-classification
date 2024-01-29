import torch.nn as nn

from torchvision.models import ResNet, resnet34


def initialize_resnet_model(number_of_classes : int) -> ResNet:

    model = resnet34(weights='IMAGENET1K_V1')
    backbone_features = model.fc.in_features
    model.fc = nn.Linear(backbone_features, number_of_classes)

    return model