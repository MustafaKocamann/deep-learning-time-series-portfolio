import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import test_loader
from transformer_model import TransformerClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model yükleme
model = TransformerClassifier()

model.load_state_dict(torch.load("transformer_fraud_model.pth", map_location=device))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

print("Test işlemi başlıyor (Grafikler devre dışı)...")

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


print("Sınıflandırma Raporu:")
print(classification_report(all_labels, all_preds))




