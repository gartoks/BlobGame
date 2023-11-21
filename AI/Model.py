import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, additional_layers=[]):
        super().__init__()
        if (isinstance(additional_layers, int)):
            additional_layers = [nn.LazyLinear(additional_layers)]
        
        self.pipeline = nn.Sequential(
            nn.LazyConv2d(64, 5),
            nn.ReLU(),
            nn.LazyConv2d(32, 3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Flatten(-3, -1),
            nn.LazyLinear(1024),
            nn.Tanh(),
            nn.LazyLinear(512),
            nn.Tanh(),
            nn.LazyLinear(128),
            nn.Tanh(),
            *additional_layers
        )
    
    def forward(self, x):
        return self.pipeline(x)