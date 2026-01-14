import numpy as np
import os 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

DATA_DIR = "data"

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

model = Sequential()

# CNN Architecture
model.add(Conv1D(filters = 16, kernel_size = 5, activation = "relu", input_shape = (1500, 1)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2))

model.add(Conv1D(filters = 32, kernel_size = 5, activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2))

model.add(Conv1D(filters = 64, kernel_size = 3, activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2))

model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation = "sigmoid"))

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

model.summary()

early_stop = EarlyStopping(monitor = "val_loss", patience = 5, restore_best_weights = True)

## Model Training
history = model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs = 5,
    batch_size = 32,
    callbacks = [early_stop]
)

## Visualization
plt.figure()
plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label = "Train Accuracy")
plt.plot(history.history["val_accuracy"], label = "Test Accuracy")
plt.plot("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

## Save the model
model.save("cnn_model.h5")
print("Model saved as cnn_model.h5")
