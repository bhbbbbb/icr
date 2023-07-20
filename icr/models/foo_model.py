
import torch
from torch import nn
# from torch.nn import functional as F

class ICRModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.linear = nn.Linear(56, 1)
        self.config = config
        return
    
    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x
    