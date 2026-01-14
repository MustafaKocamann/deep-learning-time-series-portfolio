"""
Sensor Error Detection - SECOM Manufacturing Data Preprocessing
===============================================================
Prepares highly imbalanced semiconductor manufacturing data for defect detection.
Steps: Load → Clean → Impute → Balance with SMOTE → Split → PyTorch DataLoaders

Author: ML Engineer
Project: Sensor Error Detection (SECOM Dataset)
Challenge: Extreme class imbalance (~6% defect rate)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE

# ============================================================
# 1. LOAD SECOM DATASET
# ============================================================
# 590 features from semiconductor manufacturing sensors
# Binary classification: Pass (0) or Fail (1)
df = pd.read_csv("secom.data", sep=r"\s+", header=None)
labels = pd.read_csv("secom_labels.data", sep=r"\s+", header=None)

df["label"] = labels[0].values
print(f"Loaded SECOM data: {df.shape}")

# ============================================================
# 2. MISSING DATA HANDLING
# ============================================================
# Drop columns with excessive missingness (>=90%)
missing_ratio = df.isnull().mean()
cols_to_drop = missing_ratio[missing_ratio >= 0.9].index
df.drop(columns=cols_to_drop, inplace=True)
print(f"✓ Dropped {len(cols_to_drop)} columns with >= 90% missing values")

# Impute remaining missing values with column mean
df.fillna(df.mean(), inplace=True)

# ============================================================
# 3. LABEL ENCODING
# ============================================================
# Convert labels: -1 (Fail) → 1, 1 (Pass) → 0
df["label"] = df["label"].apply(lambda x: 1 if x == -1 else 0)

# Check class distribution
print(f"\nClass distribution:")
print(df["label"].value_counts())
print(f"Defect rate: {df['label'].mean() * 100:.2f}%")

# ============================================================
# 4. FEATURE SCALING
# ============================================================
X = df.drop(columns=["label"]).values
y = df["label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✓ Standardized {X_scaled.shape[1]} features")

# ============================================================
# 5. HANDLE CLASS IMBALANCE WITH SMOTE
# ============================================================
# Synthetic Minority Oversampling Technique
# Generates synthetic samples for minority class (defects)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print(f"\n✓ Applied SMOTE:")
print(f"  Before: {len(y)} samples")
print(f"  After: {len(y_resampled)} samples")
print(f"  New class distribution: {np.bincount(y_resampled)}")

# ============================================================
# 6. TRAIN/TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_resampled
)

# ============================================================
# 7. CONVERT TO PYTORCH TENSORS
# ============================================================
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# ============================================================
# 8. CREATE DATALOADERS
# ============================================================
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ============================================================
# 9. EXPORT METADATA
# ============================================================
num_features = X_train.shape[1]
print(f"\n✓ Preprocessing completed")
print(f"  Number of features: {num_features}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")