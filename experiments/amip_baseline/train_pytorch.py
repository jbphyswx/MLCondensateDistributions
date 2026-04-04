import sys
import os

# Ensure the project root and scripts are in the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from python.dataset import CondensateDataset
from python.model import CondensateMLP
from python.train import train

# Research Workflow: Train the baseline PyTorch model on the AMIP baseline data.
# This script records the hyperparameters used for the initial model version in Python.

data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
epochs = 100
lr = 1e-3
batch_size = 512

if __name__ == "__main__":
    print("--- Starting AMIP Baseline PyTorch Training ---")
    print(f"Data Directory: {data_dir}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {lr}")

    train(
        data_dir=data_dir,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size
    )

    print("--- Training Complete ---")
