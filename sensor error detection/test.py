import torch
import torch.nn as nn
from model import LSTMFCN
from preprocessing import test_loader, num_features
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score 
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Load the trained model
model = LSTMFCN(input_size=num_features).to(device) 
model.load_state_dict(torch.load("lstmfcn_secom.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Classification Report: \n {classification_report(all_labels, all_preds)}")

cm = confusion_matrix(all_labels, all_preds)
print(f"Confusion Matrix: \n {cm}")

## Confusion Matrix Visualization
plt.figure()
sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues", xticklabels= ["Normal", "Error"], yticklabels=["Normal", "Error"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.tight_layout()
plt.show()