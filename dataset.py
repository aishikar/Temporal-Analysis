import torch
from torch.utils.data import Dataset
import numpy as np

class TrajectoryDataset(Dataset):
    def __init__(self, data, window_size):
        """
        Args:
            data (ndarray): The raw time-series data from train/val CSVs.
            window_size (int): The number of past time steps used for prediction.
        """
        self.data = data
        self.window_size = window_size

    def __len__(self):
        # Total available sequences given the window size
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # x: The look-back window (input)
        # y: The next state in the sequence (target)
        x = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
