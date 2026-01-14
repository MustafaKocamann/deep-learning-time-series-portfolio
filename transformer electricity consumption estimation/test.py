import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer import TransformerRegressor
from preprocessing import test_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔧 DÜZELTME: seq_length KALDIRILDI
model = TransformerRegressor().to(device)
model.load_state_dict(torch.load("transformer_energy_model.pth", map_location=device))
model.eval()

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = nn.MSELoss()

all_preds = []
all_targets = []
test_loss = 0.0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        output = model(batch_X)
        loss = criterion(output, batch_y)
        test_loss += loss.item()

        all_preds.extend(output.cpu().numpy().flatten())
        all_targets.extend(batch_y.cpu().numpy().flatten())

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.6f}")

plt.figure(figsize=(10, 5))
plt.plot(all_targets, label="Actual Value", color="blue")
plt.plot(all_preds, label="Prediction", color="red")
plt.title("Energy Consumption Estimation with Transformer")
plt.xlabel("Time Step")
plt.ylabel("Scaled Energy Consumption")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
