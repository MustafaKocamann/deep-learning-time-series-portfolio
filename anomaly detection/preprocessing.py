import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("sentetik_dizi.csv")

values = df["value"].values.reshape(-1,1)

scaler = MinMaxScaler()

values_scaled = scaler.fit_transform(values)

joblib.dump(scaler, "scaler.save")

def creating_sliding_windows(data, window_size):

    X = []
    for i in range(len(data)- window_size + 1):
        X.append(data[i:i + window_size])
        return np.array(X)
    
WINDOW_SIZE = 10
X = creating_sliding_windows(values_scaled, WINDOW_SIZE)

X = X.reshape((X.shape[0], X.shape[1], 1))

np.save("X_train.npy", X)