"""
Transformer Electricity Consumption - Data Preprocessing
========================================================
Prepares household power consumption data for transformer-based forecasting.
Steps: Load → Scale → Create sequences → PyTorch Dataset → Train/test split

Author: ML Engineer
Project: Electricity Consumption Forecasting with Transformer
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

# ============================================================
# 1. LOAD CLEANED TIME SERIES DATA
# ============================================================
# Household power consumption dataset (1-minute resolution)
df = pd.read_csv("cleaned_power_consumption.csv", index_col="datetime", parse_dates=True)

print(f"Loaded data: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# ============================================================
# 2. NORMALIZATION
# ============================================================
# Scale Global_active_power to [0, 1] for stable training
scaler = MinMaxScaler()
df["values"] = scaler.fit_transform(df[["Global_active_power"]])

print(f"✓ Applied MinMax scaling to Global_active_power")
print(df.head())

# ============================================================
# 3. CREATE SEQUENCES FOR TIME SERIES FORECASTING
# ============================================================
def create_sequence(data, seq_len):
    """
    Create input-output pairs for sequence-to-sequence learning
    
    Args:
        data: Time series values
        seq_len: Number of past timesteps to use as input
    
    Returns:
        X: Input sequences (shape: num_samples, seq_len)
        y: Target values (next timestep after each sequence)
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])  # Predict next timestep
    return np.array(X), np.array(y)

# Use 24 hours (1440 minutes) of history to predict next value
SEQ_LEN = 24
series = df["values"].values
X, y = create_sequence(series, SEQ_LEN)

print(f"\n✓ Created sequences:")
print(f"  Input shape: {X.shape} (samples, timesteps)")
print(f"  Target shape: {y.shape}")

# ============================================================
# 4. PYTORCH DATASET CLASS
# ============================================================
class EnergyDataset(Dataset):
    """Custom Dataset for electricity consumption sequences"""
    
    def __init__(self, X, y):
        # Add feature dimension: (N, 24) → (N, 24, 1)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        # Target shape: (N, 1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# 5. TRAIN/TEST SPLIT (80/20)
# ============================================================
split = int(0.8 * len(X))
train_dataset = EnergyDataset(X[:split], y[:split])
test_dataset = EnergyDataset(X[split:], y[split:])

print(f"\n✓ Dataset split completed:")
print(f"  Train samples: {len(train_dataset)}")
print(f"  Test samples: {len(test_dataset)}")

