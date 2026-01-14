import torch
import torch.nn as nn
from model import LSTMFCN
from preprocessing import train_loader, test_loader, num_features
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Model
model = LSTMFCN(input_size=num_features).to(device)

## Loss Function
criterion = nn.CrossEntropyLoss()

## Optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

## Training
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    avg_loss = train_loss / total
    accuracy = correct / total

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f} Accuracy: {accuracy:.4f}")

torch.save(model.state_dict(), "lstmfcn_secom.pth")
print("Model saved to lstmfcn_secom.pth")