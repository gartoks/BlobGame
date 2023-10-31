import torch
from torch import nn
import copy

from Constants import *

class ImageAndData(nn.Module):
    def __init__(self, input_dim, additional_dim, intermediate_dim, feedback_dim, output_dim):
        super().__init__()
        c, w, h = input_dim

        self.output_dim = output_dim
        self.feedback_dim = feedback_dim

        self.image_pipeline = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(93568, intermediate_dim),
            nn.ReLU(),
        )
        self.additional_pipeline = nn.Sequential(
            nn.Linear(intermediate_dim+additional_dim+feedback_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim+feedback_dim),
        )
    
    def forward(self, image, data, feedback):
        image = self.image_pipeline(image).squeeze()
        image = torch.cat((image, data, feedback), -1)
        return torch.split(self.additional_pipeline(image), [self.output_dim, self.feedback_dim], -1)


class Model(nn.Module):
    '''mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    '''
    def __init__(self, input_dim, additional_dim, output_dim, device):
        super().__init__()
        self.device = device
        c, w, h = input_dim

        if c != 1:
            raise ValueError(f"Expecting channel count: 1, got: {c}")
        if h != ARENA_HEIGHT:
            raise ValueError(f"Expecting input height: {ARENA_HEIGHT}, got: {h}")
        if w != ARENA_WIDTH:
            raise ValueError(f"Expecting input width: {ARENA_WIDTH}, got: {w}")

        self.feedback_dim = 128
        self.online = ImageAndData(input_dim, additional_dim, 512, self.feedback_dim, output_dim)

        self.target = copy.deepcopy(self.online)

        self.reset_feedback("online")
        self.reset_feedback("target")

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def reset_feedback(self, model):
        if model == 'online':
            self.online_feedback = torch.zeros(self.feedback_dim).to(self.device)
        elif model == 'target':
            self.target_feedback = torch.zeros(self.feedback_dim).to(self.device)

    def forward(self, image, data, model, feedback=None):
        if model == 'online':
            if (feedback is not None):
                old_feedback = self.online_feedback
                self.online_feedback = feedback

            output, self.online_feedback = self.online(image, data, self.online_feedback)

            if (feedback is not None):
                self.online_feedback = old_feedback
        elif model == 'target':
            if (feedback is not None):
                old_feedback = self.target_feedback
                self.target_feedback = feedback

            output, self.target_feedback = self.target(image, data, self.target_feedback)

            if (feedback is not None):
                self.target_feedback = old_feedback
        return output