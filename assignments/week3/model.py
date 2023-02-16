import torch
from typing import Callable


class MLP(torch.nn.Module):
    """
    A simple MLP model with hidden layers, activation and initializer
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.hidden_count = hidden_count
        self.activation = activation()
        self.initializer = initializer

        # create the input layer
        self.input_layer = torch.nn.Linear(self.input_size, self.hidden_size)
        self.initializer(self.input_layer.weight)

        # create hidden layers
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(self.hidden_count):
            layer = torch.nn.Linear(hidden_size, hidden_size)
            self.initializer(layer.weight)
            self.hidden_layers += [layer]

        # create output layer
        self.output_layer = torch.nn.Linear(hidden_size, num_classes)
        self.initializer(self.output_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """

        # pass input to input layer and activation function
        x = self.input_layer(x)
        x = self.activation(x)

        # pass through the hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        # pass through the output layer
        x = self.output_layer(x)

        return x
