"""
LSTM Autoencoder ile Anomali Tespiti

Problem tanımı:
Modelimiz normal veri örüntülerini öğrenerek, normal dışı örneklerde tespit gerçekleştirir.

Dataset: sentetik zaman verisi

sinüs + gürültü

3 bölgede kontrollü anomali ekleme (pozitif sıçrama, negatif çökme vb.)

saatlik veri, her saat için 1 değer olacak şekilde 1000 zaman adımı tanımlama
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

t = np.arange(0,1000)

data = np.sin(0.02*t) + 0.1 * np.random.normal(size = len(t))

data[200:210] +=  2
data[500:510] -= 3
data[800:805] += 1.5

df = pd.DataFrame({
    "timestamp":pd.date_range(start ="2025-01-01", periods=len(t), freq="H"),
    "value": data
})

print(df.head())

df.to_csv("sentetik_dizi.csv", index=False)


plt.figure(figsize=(10,5))
plt.plot(df["timestamp"], df["value"], label = "Data")
plt.title("Synthetic Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.show()