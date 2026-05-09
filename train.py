import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TrajectoryDataset
from model import MLP
import pandas as pd
import numpy as np

# 1. Hyperparameters
WINDOW_SIZE = 100
HIDDEN_SIZE = 128
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50

def train_model():
    # 2. Data Loading 
    train_data = np.random.rand(2000, 1) 
    val_data = np.random.rand(500, 1)

    train_loader = DataLoader(TrajectoryDataset(train_data, WINDOW_SIZE), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TrajectoryDataset(val_data, WINDOW_SIZE), batch_size=BATCH_SIZE, shuffle=False)

    # 3. Initialization
    model = MLP(input_size=WINDOW_SIZE, hidden_size=HIDDEN_SIZE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training & Validation Loop
    print(f"Starting Training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            # Flatten window for MLP input
            outputs = model(batch_x.view(batch_x.size(0), -1))
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation Phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_outputs = model(val_x.view(val_x.size(0), -1))
                v_loss = criterion(val_outputs, val_y)
                total_val_loss += v_loss.item()

        # Printing progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_train = total_train_loss / len(train_loader)
            avg_val = total_val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Train MSE: {avg_train:.6f} - Val MSE: {avg_val:.6f}")

    # 5. Saving Model
    torch.save(model.state_dict(), "model_checkpoint.pth")
    print("Training complete. Model saved as model_checkpoint.pth")

if __name__ == "__main__":
    train_model()
