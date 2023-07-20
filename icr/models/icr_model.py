
from torch import nn

class ICRModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.linear = nn.Linear(10, 10)
        self.config = config
        return
    
    def forward(self, x):
        return x