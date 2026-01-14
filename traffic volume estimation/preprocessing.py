import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import config  # OLUŞTURDUĞUMUZ AYAR DOSYASINI ÇAĞIRIYORUZ

def preprocess_data():
    print("--- Ön İşleme Başlıyor ---")
    
    # 1. Veri Yükleme (Config'den dosya yolunu alıyor)
    df = pd.read_csv(config.RAW_DATA_PATH)
    
    # 2. Tarih ve Index İşlemleri
    df["date_time"] = pd.to_datetime(df["date_time"])
    df.set_index("date_time", inplace=True)
    
    # !!! KRİTİK DÜZELTME: Duplicate Temizliği !!!
    # Bu satır olmazsa model hatalı veriyi öğrenir.
    initial_count = len(df)
    df = df[~df.index.duplicated(keep='first')]
    print(f"Mükerrer kayıtlar temizlendi. Silinen satır: {initial_count - len(df)}")
    
    # Eksik saatleri doldurma (Resampling) - Zaman serisi sürekliliği için
    # Veri setinde atlanan saatler varsa onları dolduruyoruz
    df = df.asfreq('H', method='ffill')

    # 3. Feature Engineering (Özellik Türetme)
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    # Not: 'day' sütununu sildim çünkü features listesinde kullanmıyorduk, gereksiz yer kaplamasın.

    # Kullanılacak Özellikler
    features = ["temp", "rain_1h", "snow_1h", "clouds_all", "hour", "dayofweek", "month"]
    target = "traffic_volume"
    
    # Eksik veri kontrolü (dropna)
    df = df[features + [target]].dropna()
    print(f"İşlenecek Veri Boyutu: {df.shape}")

    # 4. Ölçeklendirme (Scaling)
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()

    # Feature'ları ve Hedefi ayrı ayrı ölçekliyoruz
    X_scaled = scaler_X.fit_transform(df[features])
    y_scaled = scaler_Y.fit_transform(df[[target]])

    # Scaler'ları kaydet (Config'den dosya yollarını alıyor)
    joblib.dump(scaler_X, config.SCALER_X_PATH)
    joblib.dump(scaler_Y, config.SCALER_Y_PATH)
    print("Scaler dosyaları kaydedildi.")

    # 5. Zaman Serisi Dizileri Oluşturma (Sliding Window)
    X_seq, y_seq = [], []
    seq_length = config.SEQ_LEN  # Config dosyasından 24 değerini alıyor
    
    for i in range(len(X_scaled) - seq_length):
        X_seq.append(X_scaled[i : i + seq_length])
        y_seq.append(y_scaled[i + seq_length])
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"Oluşturulan Dizi Boyutu (Samples, Seq_Len, Features): {X_seq.shape}")

    # 6. Train-Test Ayrımı (%80 - %20)
    split_idx = int(0.8 * len(X_seq))

    X_train = X_seq[:split_idx]
    y_train = y_seq[:split_idx]
    X_test = X_seq[split_idx:]
    y_test = y_seq[split_idx:]

    # 7. Kaydetme (Config'den dosya yollarını alıyor)
    np.save(config.X_TRAIN_PATH, X_train)
    np.save(config.Y_TRAIN_PATH, y_train)
    np.save(config.X_TEST_PATH, X_test)
    np.save(config.Y_TEST_PATH, y_test)

    print("--- Ön İşleme Tamamlandı ve Dosyalar Kaydedildi ---")

if __name__ == "__main__":
    preprocess_data()