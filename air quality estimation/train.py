import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# load preprocessed data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

# normalize data to tensors
X_train = torch.tensor(X_train, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.float32)

# Model hyperparameters
input_size = X_train.shape[2] 
hidden_size = 64
num_layers = 2
dropout = 0.2
ourtput_size = 1
learning_rate = 0.001
num_epochs = 100


# Model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
model = LSTMModel(input_size, hidden_size, num_layers, ourtput_size, dropout)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

## training loop
loss_list = []

for epoch in range(num_epochs):

    model.train() 
    output = model(X_train) 
    loss = criterion(output, y_train) 

    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 

    loss_list.append(loss.item()) 

    if (epoch + 1 ) % 10 == 0:
        print(f"Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item()}")

# loss visualization
plt.figure()
plt.plot(loss_list)
plt.title("Training Loss Curve (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# save the model 
torch.save(model.state_dict(), "lstm_model.pt")
print("Model saved.")