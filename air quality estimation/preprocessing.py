import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("AirQualityUCI.csv", sep = ";", decimal = ",", encoding = "latin")

df.dropna(axis = 1, how = "all", inplace = True)

df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format = "%d/%m/%Y %H.%M.%S", errors = "coerce")
df.dropna(subset = ["datetime"], inplace = True)

df.drop(["Date", "Time"], axis = 1, inplace = True)
df.set_index("datetime", inplace = True)

df.replace(-200, np.nan, inplace = True)
df.interpolate(method = "time", inplace = True)

df["hour"] = df.index.hour
df["month"] = df.index.month

selected_columns = ["NO2(GT)", "T", "RH", "AH", "CO(GT)", "hour", "month"]
df = df[selected_columns]
df.dropna(inplace = True)

print(df.head())

## normalization 
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

joblib.dump(scaler, "scaler.pkl")

def create_sequences(data, window_size):
    X = []
    y = []

    for i in range(len(data) - window_size):
        seq_x = data[i:i + window_size]
        seq_y = data[i + window_size][0]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)

WINDOW_SIZE = 72
X, y = create_sequences(scaled, WINDOW_SIZE)

print(X.shape)
print(y.shape)

# train-test-split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle = False
)  

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# save data
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
