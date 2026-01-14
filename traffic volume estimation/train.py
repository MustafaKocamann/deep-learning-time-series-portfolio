import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Kendi modüllerimizi çağırıyoruz (Senior Yapı)
import config
from model import GRUNet

def train_model():
    print(f"--- Model Eğitimi Başlıyor ({config.DEVICE}) ---")
    
    # 1. VERİ YÜKLEME (Numpy dosyalarından)
    print("Veriler yükleniyor...")
    X_train_full = np.load(config.X_TRAIN_PATH)
    y_train_full = np.load(config.Y_TRAIN_PATH)
    
    # 2. VALIDATION SPLIT (Eğitim setinin %10'unu doğrulama için ayırıyoruz)
    # Model bu veriyi eğitim sırasında ASLA görmeyecek, sadece test edilecek.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, shuffle=False
    )
    
    # Tensor Dönüşümü ve Cihaza (GPU/CPU) Gönderme
    train_data = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).to(config.DEVICE),
        torch.tensor(y_train, dtype=torch.float32).to(config.DEVICE)
    )
    val_data = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).to(config.DEVICE),
        torch.tensor(y_val, dtype=torch.float32).to(config.DEVICE)
    )
    
    # DataLoader (Batch'ler halinde veri besleme)
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 3. MODEL KURULUMU (Config'den parametreleri alarak)
    model = GRUNet(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=config.OUTPUT_SIZE,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # 4. EĞİTİM DÖNGÜSÜ (TRAINING LOOP)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0  # Early Stopping sayacı
    
    print("Eğitim döngüsü başladı...")
    
    for epoch in range(config.NUM_EPOCHS):
        # --- A. TRAINING PHASE ---
        model.train() # Modeli eğitim moduna al (Dropout aktif)
        batch_losses = []
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()           # Gradyanları sıfırla
            outputs = model(X_batch)        # Tahmin yap
            loss = criterion(outputs, y_batch) # Hatayı hesapla
            loss.backward()                 # Geri yayılım (Backprop)
            optimizer.step()                # Ağırlıkları güncelle
            
            batch_losses.append(loss.item())
            
        avg_train_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(avg_train_loss)
        
        # --- B. VALIDATION PHASE ---
        model.eval() # Modeli değerlendirme moduna al (Dropout pasif)
        val_batch_losses = []
        
        with torch.no_grad(): # Gradyan hesaplama (Hız ve hafıza tasarrufu)
            for X_v, y_v in val_loader:
                out_v = model(X_v)
                v_loss = criterion(out_v, y_v)
                val_batch_losses.append(v_loss.item())
                
        avg_val_loss = sum(val_batch_losses) / len(val_batch_losses)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] "
              f"Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        # --- C. EARLY STOPPING VE CHECKPOINT ---
        # Eğer doğrulama hatası düşüyorsa modeli kaydet
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            # print("  -> En iyi model kaydedildi!") # İstersen açabilirsin
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\nERKEN DURDURMA (Early Stopping)! {config.PATIENCE} epoch boyunca iyileşme olmadı.")
                break
                
    print(f"\nEğitim Tamamlandı. En iyi Validation Loss: {best_val_loss:.5f}")
    
    # 5. GÖRSELLEŞTİRME
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss", linestyle="--")
    plt.title("Eğitim ve Doğrulama Hata Grafiği")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_model()