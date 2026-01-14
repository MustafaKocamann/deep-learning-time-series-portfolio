import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

df = pd.read_csv("cleaned_power_consumption.csv", index_col = "datetime", parse_dates=True)

scaler = MinMaxScaler()
df["values"] = scaler.fit_transform(df[["Global_active_power"]])

print(df.head())

def create_sequence(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])  
    return np.array(X), np.array(y)

SEQ_LEN = 24
series = df["values"].values
X, y = create_sequence(series, SEQ_LEN)

class EnergyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, 24, 1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # (N, 1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    
split = int(0.8 * len(X))
train_dataset = EnergyDataset(X[:split], y[:split])
test_dataset = EnergyDataset(X[split:], y[split:])

