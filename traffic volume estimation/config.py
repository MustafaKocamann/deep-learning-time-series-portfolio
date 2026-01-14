import torch

# --- DOSYA YOLLARI ---
RAW_DATA_PATH = "Metro_Interstate_Traffic_Volume.csv"
PROCESSED_DATA_PATH = "processed_data.csv"
SCALER_X_PATH = "scaler_X.save"
SCALER_Y_PATH = "scaler_y.save"
MODEL_SAVE_PATH = "gru_model.pth"
BEST_MODEL_PATH = "best_gru_model.pth"

# Numpy Dosyaları
X_TRAIN_PATH = "X_train.npy"
Y_TRAIN_PATH = "y_train.npy"
X_TEST_PATH = "X_test.npy"
Y_TEST_PATH = "y_test.npy"

# --- MODEL HİPERPARAMETRELERİ ---
INPUT_SIZE = 7
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1
DROPOUT = 0.2

# --- EĞİTİM AYARLARI ---
SEQ_LEN = 24
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10

# --- DONANIM AYARI ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')