import numpy as np
import os
import matplotlib.pyplot as plt
import random

os.makedirs("data", exist_ok=True)

SAMPLES = 1500
NUM_SIGNALS = 1000

def generate_earthquake_signal():
    t = np.linspace(0,1, SAMPLES)
    freq = np.random.uniform(5,15) # frequency in Hz
    envelope = np.exp(-((t-0.5) **2) / 0.01)  # Gaussian envelope
    signal = envelope * np.sin(2*np.pi*freq*t) # sine wave modulated by envelope
    noise = np.random.normal(0, 0.1, SAMPLES)
    return signal + noise

def generate_noise_signal():
    t = np.linspace(0,1, SAMPLES)
    base = np.sin(2 * np.pi * np.random.uniform(0.1, 1) * t)
    noise = np.random.normal(0, 0.3, SAMPLES)
    return base + noise

X = []
y = []

for _ in range(NUM_SIGNALS // 2):
    X.append(generate_earthquake_signal())
    y.append(1)  # Earthquake signal label

for _ in range(NUM_SIGNALS // 2):
    X.append(generate_noise_signal())
    y.append(0)

X = np.array(X)
y = np.array(y)

indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

np.save("data/X_signals.npy", X)
np.save("data/y_labels.npy", y)

plt.figure()

for i in range(5):
    plt.plot(X[i], label = f"label: {y[i]}")

plt.title("Sample Synthetic Signals")
plt.legend()
plt.show()