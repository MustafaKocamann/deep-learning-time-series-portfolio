import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMFCN(nn.Module):
    def __init__(self, input_size = 590, lstm_hidden_size = 128, num_classes = 2):
        super(LSTMFCN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=lstm_hidden_size, batch_first=True)

        # feature extraction layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=8, padding=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(lstm_hidden_size + 64, num_classes)

    def forward(self, x):
        # Reshape: [batch, features] -> [batch, 1, features]
        x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)
        lstm_feature = lstm_out[:, -1, :]

        # For CNN: [batch, 1, features] -> [batch, features] -> [batch, 1, features] for Conv1d
        cnn_input = x.squeeze(1).unsqueeze(1)

        # CNN Layers
        x_cnn = F.relu(self.conv1(cnn_input))
        x_cnn = F.relu(self.conv2(x_cnn))
        x_cnn = F.relu(self.conv3(x_cnn))

        # Global Average Pooling
        x_cnn = self.gap(x_cnn).squeeze(2)

        # Concatenate LSTM and CNN features
        combined = torch.cat((lstm_feature, x_cnn), dim = 1)

        out = self.fc(combined)
        return out