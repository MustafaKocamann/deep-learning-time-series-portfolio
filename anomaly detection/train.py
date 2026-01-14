import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

X_train = np.load("X_train.npy")
print(f"Training dataset loaded. {X_train.shape}")

timesteps = X_train.shape[1]
input_dim = X_train.shape[2]

# Input layer
inputs = Input(shape=(timesteps, input_dim))

# Encoder
encoded = LSTM(64, activation="relu", return_sequences = False)(inputs)
latent = RepeatVector(timesteps)(encoded)

# Decoder
decoded = LSTM(64, activation="relu", return_sequences=True)(latent)
outputs = TimeDistributed(Dense(1))(decoded)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.summary()

early_stop = EarlyStopping(monitor = "loss", patience = 5, restore_best_weights = True)

history = autoencoder.fit(
    X_train, X_train,
    epochs = 50,
    batch_size = 32,
    shuffle = True,
    callbacks = [early_stop]
)

plt.figure()
plt.plot(history.history["loss"], label = "Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Process")
plt.legend()
plt.grid(True)
plt.show()

autoencoder.save("lstm_autoencoder.h5")
print("Model was succesfully saved")