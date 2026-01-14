"""
Earthquake Early Warning System - Seismic Signal Preprocessing
==============================================================
Normalizes seismic waveform data for CNN-based earthquake detection.
Steps: Load signals → Z-score normalize → Train/test split → Save

Author: ML Engineer
Project: Earthquake Early Warning using CNN
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os 

# ============================================================
# 1. CONFIGURATION
# ============================================================
DATA_DIR = "data"

# ============================================================
# 2. LOAD SEISMIC SIGNAL DATA
# ============================================================
# X: Seismic waveform signals (samples, timesteps)
# y: Binary labels (0: No earthquake, 1: Earthquake detected)
X = np.load(os.path.join(DATA_DIR, "X_signals.npy"))
y = np.load(os.path.join(DATA_DIR, "y_labels.npy"))

print(f"Loaded seismic data - X: {X.shape}, y: {y.shape}")

# ============================================================
# 3. Z-SCORE NORMALIZATION PER SIGNAL
# ============================================================
def z_score_normalize(signal_batch):
    """
    Normalize each seismic signal independently using z-score
    Critical for comparing signals with different amplitudes
    
    Args:
        signal_batch: Array of seismic waveforms
    
    Returns:
        Normalized signals with mean=0, std=1
    """
    normalized = []
    for signal in signal_batch:
        mean = np.mean(signal)
        std = np.std(signal)
        
        # Avoid division by zero for constant signals
        if std == 0:
            std = 1
        
        norm_signal = (signal - mean) / std
        normalized.append(norm_signal)
    
    return np.array(normalized)

# Apply normalization to all signals
X_normalized = z_score_normalize(X)
print(f"✓ Applied z-score normalization to {len(X)} signals")

# ============================================================
# 4. STRATIFIED TRAIN/TEST SPLIT
# ============================================================
# Stratify ensures balanced earthquake/non-earthquake distribution
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"\n✓ Data split completed")
print(f"  Training set size: {X_train.shape[0]}")
print(f"  Testing set size: {X_test.shape[0]}")

# ============================================================
# 5. SAVE PROCESSED DATA
# ============================================================
np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)

print(f"✓ Saved preprocessed data to '{DATA_DIR}/' directory")
