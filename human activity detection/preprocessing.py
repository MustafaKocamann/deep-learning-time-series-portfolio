"""
Human Activity Detection - Multi-Sensor Data Preprocessing
==========================================================
Preprocesses accelerometer and gyroscope data from UCI HAR dataset.
Steps: Load raw data → Scale per sensor → One-hot encode labels → Save

Author: ML Engineer
Project: Human Activity Recognition Using Smartphones
Dataset: UCI HAR (6 activities, 9 sensor channels)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.utils import to_categorical

# ============================================================
# 1. LOAD RAW SENSOR DATA
# ============================================================
# Shape: (samples, timesteps=128, features=9)
# Features: body_acc_x/y/z, body_gyro_x/y/z, total_acc_x/y/z
X_train = np.load("X_train_raw.npy")
X_test = np.load("X_test_raw.npy")

# Labels: 0-5 representing 6 activities
# (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)
y_train = np.load("y_train_raw.npy")
y_test = np.load("y_test_raw.npy")

print(f"Raw data loaded - Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================
# 2. ONE-HOT ENCODE LABELS
# ============================================================
# Convert integer labels to categorical for multi-class classification
num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

print(f"Label encoding: {num_classes} classes")

# ============================================================
# 3. STANDARDIZE EACH SENSOR CHANNEL INDEPENDENTLY
# ============================================================
# Scale each of the 9 sensor channels separately (important for sensor fusion)
scalers = {}
X_train_scaled = np.zeros_like(X_train)
X_test_scaled = np.zeros_like(X_test)

for i in range(X_train.shape[2]):  # Loop through 9 features
    scaler = StandardScaler()
    # Fit on train, transform both train and test
    X_train_scaled[:, :, i] = scaler.fit_transform(X_train[:, :, i])
    X_test_scaled[:, :, i] = scaler.transform(X_test[:, :, i])
    scalers[i] = scaler

print(f"✓ Scaled {X_train.shape[2]} sensor channels independently")

# ============================================================
# 4. SAVE PROCESSED DATA
# ============================================================
# Save scaled features
np.save("X_train.npy", X_train_scaled)
np.save("X_test.npy", X_test_scaled)

# Save one-hot encoded labels
np.save("y_train.npy", y_train_cat)
np.save("y_test.npy", y_test_cat)

# Save scalers for future inference
joblib.dump(scalers, "scalers.pkl")

print("✓ Data and scalers successfully saved")
print(f"  Final shapes - X_train: {X_train_scaled.shape}, y_train: {y_train_cat.shape}")