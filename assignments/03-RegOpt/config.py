from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    batch_size = 100
    num_epochs = 10
    initial_learning_rate = 0.0001
    initial_weight_decay = 0

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "step_size": 1000,
        "gamma": 0.9998,
        "mode": "triangular",
        "max_lr": 0.01,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose([ToTensor(), Normalize(mean=0.5, std=0.5)])


# transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
#         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#         transforms.RandomRotation(10)
