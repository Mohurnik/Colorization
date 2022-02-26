import torch
import torch.nn as nn
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet


class ColorizationModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()

        body = create_body(resnet18, pretrained=True, n_in=in_channels, cut=-2)
        self.model = DynamicUnet(body, out_channels, (256, 256))

    def forward(self, x):
        return torch.tanh(self.model(x))
