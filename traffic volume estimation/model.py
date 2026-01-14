import torch
import torch.nn as nn

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUNet, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU Katmanı
        # batch_first=True: Girdi formatı (Batch, Seq, Feature) olur.
        # dropout: Ezberlemeyi önlemek için nöronların bir kısmını kapatır.
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully Connected (FC) Katmanı
        # GRU'dan çıkan sonucu tek bir sayıya (Trafik Hacmi) dönüştürür.
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # Başlangıç hidden state'i (otomatik sıfır tensörü oluşturur ama biz açıkça belirtelim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # GRU İleri Besleme
        # out shape: (batch_size, seq_len, hidden_size)
        out, _ = self.gru(x, h0)
        
        # Sadece son zaman adımındaki çıktıyı alıyoruz (Many-to-One)
        last_out = out[:, -1, :]
        
        # Tahmin üret
        prediction = self.fc(last_out)
        
        return prediction