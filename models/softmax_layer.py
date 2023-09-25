import torch

import torch.nn as nn


class SoftMaxLayer(nn.Module):

    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax()
    
    def forward(self, x):
        return self.softmax(x)