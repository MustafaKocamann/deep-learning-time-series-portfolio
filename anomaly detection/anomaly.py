import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt

model = load_model("lstm_autoencoder.h5", compile = False)

scaler = joblib.load("scaler.save")

df = pd.read_csv("sentetik_dizi.csv")
values = df["value"].values.reshape(-1,1)

values_scaled = scaler.transform(values)

def creating_sliding_windows(data, window_size):
    X = []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:+i + window_size])
    return np.array(X)

WINDOW_SIZE = 10

X_all = creating_sliding_windows(values_scaled, WINDOW_SIZE)
X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))

x_pred = model.predict(X_all)

mse_list = np.mean(np.square(X_all - x_pred), axis = (1,2))

threshold = np.percentile(mse_list, 95)
print(f"Eşik Değer:{threshold}")

anomalies = mse_list > threshold

df_results = df.iloc[WINDOW_SIZE - 1:].copy()
df_results["reconstruction_error"] = mse_list
df_results["anommaly"] = anomalies

df_results.to_csv("anomaly_results", index = False)
print(f"Anomally results was succesfully saved.")

plt.figure()
plt.plot(df_results["timestamp"], df_results["value"], label = "data")

plt.scatter(df_results[df_results["anommaly"]]["timestamp"],
            df_results[df_results["anommaly"]]["value"],
            color = "red", label = "Anomaly", s = 20)

plt.title("Time Series and Detected Anomalies")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.show()