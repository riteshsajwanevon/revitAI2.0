#!/usr/bin/env python3
"""
Train Improved CNN WITHOUT Data Leakage
Validation dataset loaded from separate folder path
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

print("=" * 80)
print("IMPROVED CNN TRAINING - PATH BASED DATA SPLIT")
print("=" * 80)

# Paths
TRAIN_DATA_PATH = "Dataset/train/beam_training_dataset_smoothed.npz"
VAL_DATA_PATH = "Dataset/validation/beam_training_dataset_smoothed.npz"

# ==========================================================
# LOAD DATASETS
# ==========================================================

train_data = np.load(TRAIN_DATA_PATH)
val_data = np.load(VAL_DATA_PATH)

X_train = train_data["X"]
Y_train = train_data["Y"]

X_val = val_data["X"]
Y_val = val_data["Y"]

print("\nDataset shapes:")
print(f"  Train X: {X_train.shape}")
print(f"  Train Y: {Y_train.shape}")
print(f"  Val   X: {X_val.shape}")
print(f"  Val   Y: {Y_val.shape}")

print(f"\nTraining beams: {len(X_train)}")
print(f"Validation beams: {len(X_val)}")


# ==========================================================
# DATASET CLASS
# ==========================================================

class BeamDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


train_dataset = BeamDataset(X_train, Y_train)
val_dataset = BeamDataset(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ==========================================================
# MODEL
# ==========================================================

class ImprovedColumnPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(64)

        self.conv5 = nn.Conv1d(64, 1, kernel_size=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout(self.relu(self.bn4(self.conv4(x))))

        x = self.sigmoid(self.conv5(x))

        return x.squeeze(1)


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = ImprovedColumnPredictor().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ==========================================================
# LOSS / OPTIMIZER
# ==========================================================

criterion = nn.SmoothL1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)

# ==========================================================
# TRAINING
# ==========================================================

EPOCHS = 100
PATIENCE = 15

best_val_loss = float("inf")
patience_counter = 0

print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

for epoch in range(EPOCHS):

    # -----------------------
    # TRAIN
    # -----------------------
    model.train()
    train_loss = 0

    for X_batch, Y_batch in train_loader:

        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        pred = model(X_batch)

        # weighted loss
        loss = ((pred - Y_batch) ** 2 * (1 + 10 * Y_batch)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -----------------------
    # VALIDATION
    # -----------------------
    model.eval()
    val_loss = 0

    with torch.no_grad():

        for X_batch, Y_batch in val_loader:

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            pred = model(X_batch)

            loss = criterion(pred, Y_batch)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch+1:3d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    # -----------------------
    # EARLY STOPPING
    # -----------------------
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        patience_counter = 0

        torch.save(
            model.state_dict(),
            "column_predictor_no_leakage.pth"
        )

        print(f"  → Best model saved (val_loss: {val_loss:.4f})")

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation loss: {best_val_loss:.4f}")

            break


# ==========================================================
# SAVE TRAINING LOG
# ==========================================================

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

print(f"\nBest validation loss: {best_val_loss:.4f}")

with open("cnn_training_log.txt", "w") as f:

    f.write("CNN Training (PATH BASED SPLIT)\n")
    f.write("=" * 80 + "\n")

    f.write(f"Train beams: {len(X_train)}\n")
    f.write(f"Validation beams: {len(X_val)}\n")

    f.write(f"Best validation loss: {best_val_loss:.4f}\n")

    f.write(f"Model parameters: {total_params:,}\n")

    f.write("Architecture: 5-layer CNN with BatchNorm and Dropout\n")

print("\nTraining log saved: cnn_training_log.txt")