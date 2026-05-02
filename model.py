import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Multilayer Perceptron for state prediction[cite: 1].
        Args:
            input_size (int): Matches the sliding window length.
            hidden_size (int): Number of neurons in the hidden layers[cite: 1].
        """
        super(MLP, self).__init__()
        # Architecture: 3-layer design with ReLU activations[cite: 1]
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1) # Output represents the next predicted state[cite: 1]
        )

    def forward(self, x):
        # Standard forward pass through the dense layers
        return self.layers(x)
