import torch
from torch.utils.data import Dataset
import pandas as pd
import glob
import os

class CondensateDataset(Dataset):
    def __init__(self, data_dir, feature_cols, target_cols, transform=None):
        self.data_dir = data_dir
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.transform = transform
        
        # Load all parquet files
        all_files = glob.glob(os.path.join(data_dir, "*.parquet"))
        dfs = [pd.read_parquet(f) for f in all_files]
        self.df = pd.concat(dfs, ignore_index=True)
        
        # Convert to tensors
        self.X = torch.tensor(self.df[feature_cols].values, dtype=torch.float32)
        self.Y = torch.tensor(self.df[target_cols].values, dtype=torch.float32)
        
        # Simple normalization if no transform provided
        if transform is None:
            self.mean = self.X.mean(dim=0)
            self.std = self.X.std(dim=0)
            self.std[self.std == 0] = 1.0
            self.X = (self.X - self.mean) / self.std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
