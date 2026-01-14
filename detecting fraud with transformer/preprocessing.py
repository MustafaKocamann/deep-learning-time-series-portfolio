"""
Fraud Detection with Transformer - Data Preprocessing
=====================================================
Prepares synthetic transaction sequences for transformer-based classification.
Steps: Load → Scale → Split → Convert to PyTorch tensors → Create DataLoaders

Author: ML Engineer
Project: Detecting Fraud with Transformer
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

# ============================================================
# 1. LOAD SYNTHETIC FRAUD DATA
# ============================================================
# X shape: (num_sequences, sequence_length, num_features)
# y shape: (num_sequences,) - Binary labels (0: Normal, 1: Fraud)
X = np.load("X_fraud.npy")
y = np.load("y_fraud.npy")

print(f"Loaded data - X: {X.shape}, y: {y.shape}")
print(f"Fraud rate: {(y == 1).sum() / len(y) * 100:.2f}%")

# ============================================================
# 2. FEATURE SCALING
# ============================================================
# Standardize features (mean=0, std=1) for stable transformer training
scaler = StandardScaler()

# Reshape from (samples, timesteps, features) to (samples*timesteps, features)
X_reshaped = X.reshape(-1, X.shape[2])
X_scaled = scaler.fit_transform(X_reshaped)

# Reshape back to original shape
X = X_scaled.reshape(X.shape)

# ============================================================
# 3. TRAIN/TEST SPLIT
# ============================================================
# Stratified split ensures balanced fraud distribution in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# ============================================================
# 4. CONVERT TO PYTORCH TENSORS
# ============================================================
# Convert numpy arrays to PyTorch tensors for GPU acceleration
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Long for CrossEntropyLoss
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# ============================================================
# 5. CREATE DATASETS AND DATALOADERS
# ============================================================
# TensorDataset combines features and labels
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# DataLoader handles batching and shuffling
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for test

# ============================================================
# 6. VERIFICATION
# ============================================================
print(f"\n✓ Preprocessing completed")
print(f"  Train batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")

# Verify batch shape
for batch_X, batch_y in train_loader:
    print(f"  Batch shape - X: {batch_X.shape}, y: {batch_y.shape}")
    break