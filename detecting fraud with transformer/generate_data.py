"""
Problem tanımı: 
- E ticaret platformunda kullanıcıların oturumları boyunca gerçekleştirdiği davranışlara dayanıklı olarak onların bot mu, normal bir kullanıcı mı, dolandırıcı mı olup olmadığını tahmin etmek
"""

import numpy as np
import pandas as pd

np.random.seed(42)

sequence_length = 20

n_features = 5

n_samples_per_class = 500

total_samples = n_samples_per_class * 3

data = []
labels = []

# class 0

for _ in range(n_samples_per_class):
    click_count = np.random.normal(loc = 3, scale = 1, size = sequence_length)
    page_duration = np.random.normal(loc = 5, scale = 2, size = sequence_length)
    amount_spent = np.random.normal(loc = 10, scale = 5, size = sequence_length)
    scroll_depth = np.random.normal(loc = 50, scale = 15, size = sequence_length)
    card_events = np.random.poisson(lam = 0.5,size = sequence_length )

    sample = np.stack([click_count, page_duration, amount_spent, scroll_depth, card_events], axis = 1)
    data.append(sample)
    labels.append(0)

# class 1

for _ in range(n_samples_per_class):
    click_count = np.random.normal(loc = 10, scale = 3, size = sequence_length)
    page_duration = np.random.normal(loc = 1, scale = 0.5, size = sequence_length)
    amount_spent = np.zeros(sequence_length)
    scroll_depth = np.random.normal(loc = 10, scale = 5, size = sequence_length)
    card_events = np.zeros(sequence_length )

    sample = np.stack([click_count, page_duration, amount_spent, scroll_depth, card_events], axis = 1)
    data.append(sample)
    labels.append(1)

# class 2

# ... (önceki kodlar aynı)

# class 2
for _ in range(n_samples_per_class):
    click_count = np.random.normal(loc = 4, scale = 4, size = sequence_length)
    page_duration = np.random.normal(loc = 6, scale = 3, size = sequence_length)
    amount_spent = np.random.exponential(scale = 20, size = sequence_length)
    scroll_depth = np.random.normal(loc = 80, scale = 20, size = sequence_length)
    card_events = np.random.poisson(lam = 2, size = sequence_length)

    sample = np.stack([click_count, page_duration, amount_spent, scroll_depth, card_events], axis = 1)

    # BU SATIRLAR DÖNGÜ İÇİNE ALINMALI (TAB İLE İÇERİ ALIN)
    data.append(sample)
    labels.append(2)

# ... (kalan kodlar aynı)

data = np.array(data)
labels = np.array(labels)

print("data shape BEFORE shuffle:", data.shape)
print("labels shape BEFORE shuffle:", labels.shape)

# Burada total_samples yerine gerçek uzunluğu kullan
num_samples = data.shape[0]  # veya len(data)

indices = np.arange(num_samples)
np.random.shuffle(indices)

data = data[indices]
labels = labels[indices]

print("data shape AFTER shuffle:", data.shape)
print("labels shape AFTER shuffle:", labels.shape)

np.save("X_fraud.npy", data)
np.save("y_fraud.npy", labels)

df = pd.DataFrame(
    data[0],
    columns=["click_count", "page_duration", "amount_spent", "scroll_depth", "card_events"]
)
df["step"] = range(1, sequence_length + 1)
print("Sample data")
print(df)
