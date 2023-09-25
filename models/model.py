import torch
import torch.nn as nn
from torchvision import models

class newResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = models.resnet50(pretrained = True)
        self.outputs = {}


    def activation(self, name):
        def hook(module, input, output):
            self.outputs[name] = output.detach()
        return hook
    
    def get_activation(self):
        return self.outputs
    
    def forward(self, x):
        return self.net(x)

