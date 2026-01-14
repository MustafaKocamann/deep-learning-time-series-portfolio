import torch
import torch.nn as nn
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from train import LSTMModel

# load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# convert from numpy array format to pytorch tensor format
X_test = torch.tensor(X_test, dtype = torch.float32)
y_test = torch.tensor(y_test, dtype = torch.float32).view(-1,1)

# load scaler object for normalization
scaler = joblib.load("scaler.pkl")

#  Model Loading and Prediction ---
input_size = X_test.shape[2]
hidden_size = 64
num_layers = 2
dropout_rate = 0.2
output_size = 1

# create model
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate)
model.load_state_dict(torch.load("lstm_model.pt"))
model.eval() # set model to evaluation mode

# make predictions
with torch.no_grad():
    predictions = model(X_test)

# convert predictions to numpy array
y_pred = predictions.numpy()
y_true = y_test.numpy()

# reverse normalization
dummy_pred = np.zeros((len(y_pred), scaler.n_features_in_))

# --- image_34e1ab.png: Inverse Scaling and Results ---
dummy_pred[:, 0] = y_pred[:, 0]

dummy_true = np.zeros((len(y_true), scaler.n_features_in_))
dummy_true[:, 0] = y_true[:, 0]

inv_y_pred = scaler.inverse_transform(dummy_pred)[:, 0]
inv_y_true = scaler.inverse_transform(dummy_true)[:, 0]

# actual vs prediction comparison
plt.figure()
plt.plot(inv_y_true, label = "Actual NO2", color = "blue")
plt.plot(inv_y_pred, label = "Predicted NO2", color = "red", alpha = 0.5)
plt.title("Actual vs Predicted NO2")
plt.xlabel("Time Step")
plt.ylabel("NO2")
plt.legend()
plt.show()

mae = mean_absolute_error(inv_y_true, inv_y_pred)
mse = mean_squared_error(inv_y_true, inv_y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")