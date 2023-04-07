"""
Creating the model architecture for training the Alphazero model
"""

# Importing dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F


# Definition for the Residual Network
class ResNet(nn.Module):
    """
    Start Block = Conv layer + Batch Norm layer + ReLU
    Backbone = Array of different Res Blocks
    """
    def __init__(self, game, num_resBlocks, num_hidden):
        super().__init__()

        self.start_Block = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )


# Definition for the Residual Blocks
class ResBlock(nn.Module):
    """
    Resnet block which has 2 convolutional and batch normalization layers along with skip connections in
    the form of ReLU
    """
    def __init__(self, num_hidden):
        super().__init__()
        self.convBlock1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.convBlock2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    """
    Forward function performs F(x) + x
    """
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.convBlock1(x)))
        x = self.bn2(self.convBlock2(x))
        x += residual
        x = F.relu(x)
        return x


