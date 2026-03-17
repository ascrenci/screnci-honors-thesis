#!/home/alex/miniconda3/envs/torch/bin/python

import torch
import numpy as np
from model_classes import ClassADataset, ClassAModel
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from config import data_file
from config import n_datapoints

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(33882356)

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=100, patience=10, plot=True):
    best_val_loss = float("inf")
    patience_counter = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.6, patience=5)

    # History for plotting
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()

        val_loss /= len(val_loader)

        # Step LR scheduler
        scheduler.step(val_loss)

        # Record history
        current_lr = optimizer.param_groups[0]['lr']
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # --- Print progress ---
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # --- Plot learning curves ---
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig('loss_curve.png')

        plt.figure(figsize=(8, 3))
        plt.plot(history["lr"])
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("LR Schedule")
        plt.grid(True)
        plt.savefig('lr_schedule.png')

    return history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset splitting
full_dataset = ClassADataset(data_file)

total_len = len(full_dataset)
train_len = int(0.7 * total_len)
val_len = total_len - train_len

train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

# Batch size
bs = 16

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

# Model, criterion, optimizer
model = ClassAModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Train model
history = train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs=300,
    patience=30,
    plot=True
)

torch.save(model.state_dict(), f"class_a_model_3_{n_datapoints}.pth")