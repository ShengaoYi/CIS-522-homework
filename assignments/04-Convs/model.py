import torch
import torch.nn as nn


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
            in_channels=num_channels, out_channels=32, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Forward pass of CNN

        Args:
            x: Input

        Returns: Output result

        """

        x = self.conv1(x)
        x = nn.ReLU(x)

        x = self.conv2(x)
        x = nn.ReLU(x)

        x = self.conv2(x)
        x = nn.ReLU(x)

        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = nn.ReLU(x)

        x = self.fc2(x)

        return x
