import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os 

DATA_DIR = "data"

X = np.load(os.path.join(DATA_DIR, "X_signals.npy"))
y = np.load(os.path.join(DATA_DIR, "y_labels.npy"))

def z_score_normalize(signal_batch):
    normalized = []
    for signal in signal_batch:
        mean = np.mean(signal)
        std = np.std(signal)

        if std == 0:
            std = 1
        norm_signal = (signal-mean) / std
        normalized.append(norm_signal)
    return np.array(normalized)

X_normalized = z_score_normalize(X)

X_train, X_test, y_train,y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42, stratify = y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)
