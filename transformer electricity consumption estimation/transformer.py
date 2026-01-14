import torch
import torch.nn as nn

class TransformerRegressor(nn.Module):
    def __init__(self, d_model=64, n_head=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

   
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (batch, seq_len, 1)
        """
        x = self.input_proj(x)        
        x = self.transformer(x)      
        last_step = x[:, -1, :]       
        out = self.fc_out(last_step) 
        return out

    


