import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """
    My Convolutional Neural Network
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """

        Initialise parameters of CNN

        Args:
            num_channels: Number of input channels
            num_classes: Number of output classes
        """

        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=10,
            kernel_size=5,
            padding=1,
            stride=2,
        )

        self.bn1 = torch.nn.BatchNorm2d(10)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(490, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Forward pass of CNN

        Args:
            x: Input

        Returns: Output result

        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)

        return x
