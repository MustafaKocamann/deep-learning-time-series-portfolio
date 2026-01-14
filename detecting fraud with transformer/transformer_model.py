import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    
    def __init__(self, input_dim = 5, model_dim = 64, num_heads = 4, num_layers = 2, num_classes = 3, dropout = 0.1):
        super(TransformerClassifier, self).__init__()

        self.input_project = nn.Linear(input_dim, model_dim)

        self.positional_embedding = nn.Parameter(torch.randn(1, 20, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model= model_dim,
            nhead = num_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers= num_layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(), # DÜZELTME 1: Buraya () ekledim.
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.input_project(x)
        x = x + self.positional_embedding

        x = self.transformer_encoder(x)

        x = x.transpose(1,2)
        x = self.pooling(x).squeeze(-1)

        out = self.classifier(x)

        return out
    
if __name__ == "__main__":
    dummy_input = torch.randn(32, 20, 5)

    # DÜZELTME 2: Sınıfı parantez () ile başlattık.
    model = TransformerClassifier() 

    output = model(dummy_input)

    print("Çıktı boyutu:", output.shape)