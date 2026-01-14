import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader # TensorDataset eklendi

# 1. Veriyi Yükle
X = np.load("X_fraud.npy")
y = np.load("y_fraud.npy")

# 2. Ölçeklendirme (Scaling)
scaler = StandardScaler()
X_reshaped = X.reshape(-1, X.shape[2])
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(X.shape)

# 3. Train/Test Split
# Data düzeldikten sonra burası hatasız çalışacaktır.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Tensor Dönüşümü
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# DÜZELTME: X_train yerine X_test kullanıldı
X_test_tensor = torch.tensor(X_test, dtype=torch.float32) 
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long) 

# 5. Dataset ve DataLoader Oluşturma
# DÜZELTME: Önce TensorDataset oluşturulmalı, sonra DataLoader'a verilmeli.
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # Test set genellikle shuffle edilmez

print(f"train shape: {X_train_tensor.shape}")
print(f"test shape: {X_test_tensor.shape}")
# Örnek bir batch kontrolü
for batch_X, batch_y in train_loader:
    print(f"Batch X shape: {batch_X.shape}, Batch y shape: {batch_y.shape}")
    break