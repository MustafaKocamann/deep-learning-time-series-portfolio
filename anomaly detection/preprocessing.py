"""
Anomaly Detection - Data Preprocessing Pipeline
===============================================
This script prepares time series data for LSTM autoencoder training.
Steps: Load data → Normalize → Create sliding windows → Save for training

Author: ML Engineer
Project: Anomaly Detection in Time Series
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# 1. DATA LOADING
# ============================================================
# Load synthetic time series data
df = pd.read_csv("sentetik_dizi.csv")

# Extract values and reshape for scaling (samples, features)
values = df["value"].values.reshape(-1, 1)

# ============================================================
# 2. NORMALIZATION
# ============================================================
# Scale values to [0, 1] range for better neural network training
scaler = MinMaxScaler()
values_scaled = scaler.fit_transform(values)

# Save scaler for inverse transformation during anomaly detection
joblib.dump(scaler, "scaler.save")

# ============================================================
# 3. SLIDING WINDOW CREATION
# ============================================================
def creating_sliding_windows(data, window_size):
    """
    Create sliding windows for sequence learning
    
    Args:
        data: Normalized time series data
        window_size: Number of timesteps per window
    
    Returns:
        numpy array of shape (num_windows, window_size)
    """
    X = []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:i + window_size])
    return np.array(X)

# Define window size (autoencoder will learn patterns in 10-step sequences)
WINDOW_SIZE = 10
X = creating_sliding_windows(values_scaled, WINDOW_SIZE)

# ============================================================
# 4. RESHAPE FOR LSTM INPUT
# ============================================================
# Reshape to (samples, timesteps, features) for LSTM autoencoder
X = X.reshape((X.shape[0], X.shape[1], 1))

# ============================================================
# 5. SAVE PROCESSED DATA
# ============================================================
np.save("X_train.npy", X)
print(f"✓ Preprocessing completed: {X.shape[0]} windows created")
print(f"  Shape: {X.shape} (samples, timesteps, features)")