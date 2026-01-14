import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

DATA_DIR = "data"
MODEL_PATH = "cnn_model.h5"

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

X_test = X_test[..., np.newaxis]

model = load_model(MODEL_PATH)
print("Model loaded successfully.")

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

## Metrics Display
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")    
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
cm = confusion_matrix(y_test, y_pred)

## Confusion Matrix Visualization
plt.figure()
sns.heatmap(cm, annot = True, fmt="d", cmap = "Blues", xticklabels=["Noise", "Earthquake"], yticklabels=["Noise", "Earthquake"])
plt.xlabel("Predicted Value")
plt.ylabel("True Value")
plt.title("Confusion Matrix")
plt.show()