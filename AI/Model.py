import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, additional_layers=[]):
        super().__init__()

        self.linear_pipeline = nn.Sequential(
            nn.LazyLinear(256),
            nn.Tanh(),
            nn.LazyLinear(128),
            nn.Tanh(),
        )
        self.pixel_pipeline = nn.Sequential(
            nn.LazyConv2d(16, 5),
            nn.ReLU(),
            nn.LazyConv2d(8, 4),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Flatten(),
        )

        self.combined_pipeline = nn.Sequential(
            nn.LazyLinear(128),
            nn.Tanh(),
            nn.LazyLinear(64),
            nn.Tanh(),
            *additional_layers
        )
    
    def forward(self, pixel_data, linear_data):
        image_transformed = self.pixel_pipeline(pixel_data)
        linear_transformed = self.linear_pipeline(linear_data)
        combined = torch.concat((torch.flatten(image_transformed), torch.flatten(linear_transformed)))
        combined_transformed = self.combined_pipeline(combined)
        return combined_transformed