import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .dataset import CondensateDataset
from .model import CondensateMLP
import os

def train(data_dir, epochs=100, lr=1e-3, batch_size=512):
    # 1. Define columns (must match the 31-column schema logic)
    # Note: These indices depend on SCHEMA_SYMBOL_ORDER in DatasetBuilder
    feature_cols = ['qt', 'theta_li', 'p', 'rho', 'w', 'tke', 'resolution_h', 'resolution_z']
    target_cols = ['q_liq', 'q_ice']
    
    print(f"Loading data from {data_dir}...")
    dataset = CondensateDataset(data_dir, feature_cols, target_cols)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Initialize Model
    model = CondensateMLP(len(feature_cols), len(target_cols))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    print(f"Starting training on {len(dataset)} samples...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader):.8f}")
            
    # 4. Save Model
    model_dir = os.path.join(os.path.dirname(data_dir), "models")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "pytorch_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'mean': dataset.mean,
        'std': dataset.std
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train("data")
