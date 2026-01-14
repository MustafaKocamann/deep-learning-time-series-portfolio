import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.utils import to_categorical

X_train = np.load("X_train_raw.npy")
X_test = np.load("X_test_raw.npy")
y_train = np.load("y_train_raw.npy")
y_test = np.load("y_test_raw.npy")

num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes = num_classes)
y_test_cat = to_categorical(y_test, num_classes = num_classes)

scalers = {}
X_train_scaled = np.zeros_like(X_train)
X_test_scaled = np.zeros_like(X_test)

for i in range(X_train.shape[2]):
    scaler = StandardScaler()
    X_train_scaled[:,:,i] = scaler.fit_transform(X_train[:,:,i])
    X_test_scaled[:,:,i] = scaler.transform(X_test[:,:,i])
    scalers[i] = scaler

np.save("X_train.npy", X_train_scaled)
np.save("X_test.npy", X_test_scaled)
np.save("y_train.npy", y_train_cat)
np.save("y_test.npy", y_test_cat)

joblib.dump(scalers , "scalers.pkl")
print("Data and scaler was succesfully saved.")