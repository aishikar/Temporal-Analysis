import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TrajectoryDataset
from model import MLP
import pandas as pd
import numpy as np

# Finalized Hyperparameters from Week 3
WINDOW_SIZE = 100
HIDDEN_SIZE = 128
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50

def load_data():
    # Load the actual dataset tables from your data directory
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')
    
    # Exclude the ID column and convert numerical trajectories to a flat matrix/sequence
    train_values = train_df.iloc[:, 1:].values.astype(np.float32).flatten()
    val_values = val_df.iloc[:, 1:].values.astype(np.float32).flatten()
    
    return train_values, val_values

def train_model():
    print("Loading actual trajectory sequences...")
    train_data, val_data = load_data()

    train_loader = DataLoader(TrajectoryDataset(train_data, WINDOW_SIZE), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TrajectoryDataset(val_data, WINDOW_SIZE), batch_size=BATCH_SIZE, shuffle=False)

    model = MLP(input_size=WINDOW_SIZE, hidden_size=HIDDEN_SIZE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting final baseline validation loop ({EPOCHS} epochs)...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            # Reshape input to (batch_size, input_size) to fit the linear layer
            outputs = model(batch_x.view(batch_x.size(0), -1))
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation monitoring 
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for val_x, val_y in val_loader:
                    val_outputs = model(val_x.view(val_x.size(0), -1))
                    v_loss = criterion(val_outputs, val_y)
                    total_val_loss += v_loss.item()
            
            avg_train = total_train_loss / len(train_loader)
            avg_val = total_val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f}")

    # Save the optimized weights 
    torch.save(model.state_dict(), "model_checkpoint.pth")
    print("Model checkpoint successfully saved as model_checkpoint.pth")

if __name__ == "__main__":
    train_model()
