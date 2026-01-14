import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# SENIOR YAPI: Ayarları ve Modeli merkezden çağırıyoruz
import config
from model import GRUNet

def test_model():
    print(f"--- Test Aşaması Başlıyor ({config.DEVICE}) ---")

    # 1. VERİ YÜKLEME
    # Preprocessing aşamasında ayırdığımız Test setini yüklüyoruz
    X_test = np.load(config.X_TEST_PATH)
    y_test = np.load(config.Y_TEST_PATH)

    # Tensor Dönüşümü
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(config.DEVICE)

    # 2. MODEL KURULUMU VE YÜKLEME
    # Parametreleri config'den alıyoruz (Hata ihtimali sıfırlanıyor)
    model = GRUNet(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=config.OUTPUT_SIZE,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    # En iyi ağırlıkları yüklüyoruz
    print(f"Model yükleniyor: {config.BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    
    model.eval() # Değerlendirme modu (Dropout kapanır)

    # 3. TAHMİN (INFERENCE)
    print("Tahminler yapılıyor...")
    with torch.no_grad():
        predictions = model(X_test_tensor)

    # GPU'daysa CPU'ya alıp numpy'a çeviriyoruz
    predictions = predictions.cpu().numpy()

    # 4. GERİ DÖNÜŞÜM (INVERSE SCALING)
    # Scaler'ı yükleyip veriyi 0-1 arasından gerçek trafik değerlerine çeviriyoruz
    try:
        scaler_y = joblib.load(config.SCALER_Y_PATH)
        predictions_original = scaler_y.inverse_transform(predictions)
        y_test_original = scaler_y.inverse_transform(y_test)
    except Exception as e:
        print(f"UYARI: Scaler yüklenemedi! ({e})")
        print("Sonuçlar ölçekli (0-1 arası) gösterilecek.")
        predictions_original = predictions
        y_test_original = y_test

    # 5. METRİKLER (PERFORMANS ÖLÇÜMÜ)
    rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
    mae = mean_absolute_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)

    print("\n--- Test Sonuçları ---")
    print(f"RMSE (Kök Ortalama Kare Hata): {rmse:.2f}")
    print(f"MAE  (Ortalama Mutlak Hata)  : {mae:.2f}")
    print(f"R2 Score (Açıklayıcılık)     : {r2:.4f}")

    # 6. GÖRSELLEŞTİRME
    # Tüm veriyi çizdirmek yerine son 200 saati çizdirelim ki detaylar görünsün
    plot_limit = 200 
    
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_original[:plot_limit], label="Gerçek Değerler (Actual)", color="#1f77b4", linewidth=2)
    plt.plot(predictions_original[:plot_limit], label="Model Tahmini (Prediction)", color="#ff7f0e", linewidth=2, linestyle='--')
    
    plt.title(f"Trafik Hacmi Tahmini (İlk {plot_limit} Saat) | R2: {r2:.2f}")
    plt.xlabel("Zaman (Saat)")
    plt.ylabel("Trafik Hacmi")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    test_model()

# Mevcut kodların altına ekle:

# --- API AYARLARI (OPENWEATHERMAP) ---
API_KEY = "BURAYA_KENDI_API_KEYINI_YAZ"  # <-- Senin Key'in
CITY = "Minneapolis" # Model bu şehrin verisiyle eğitildiği için en mantıklısı bu
# Minneapolis Koordinatları (Hava durumu tahmini için daha hassas)
LAT = 44.9778
LON = -93.2650