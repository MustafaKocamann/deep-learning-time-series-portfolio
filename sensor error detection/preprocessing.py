import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE

df = pd.read_csv("secom.data", sep = r"\s+", header=None)
labels = pd.read_csv("secom_labels.data", sep = r"\s+", header = None)

df["label"] = labels[0].values

## Remove columns with 90% or more missing values
missing_ratio = df.isnull().mean()
cols_to_drop = missing_ratio[missing_ratio >= 0.9].index
df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped {len(cols_to_drop)} columns with >= 90% missing values.")

## Handle missing values by imputing with column mean
df.fillna(df.mean(), inplace=True)

## convert labels 
df["label"] = df["label"].apply(lambda x: 1 if x == -1 else 0)

X = df.drop(columns=["label"]).values
y = df["label"].values

## Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## SMOTE for balancing classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

## Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

## Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
y_train_tensor = torch.tensor(y_train, dtype = torch.long)
X_test_tensor = torch.tensor(X_test, dtype = torch.float32)
y_test_tensor = torch.tensor(y_test, dtype = torch.long)

## DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

## Export number of features for model
num_features = X_train.shape[1]

## Export number of features for model
num_features = X_train.shape[1]
print(f"Number of features: {num_features}")