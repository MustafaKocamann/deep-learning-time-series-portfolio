"""
Problem tanımı:
Giyilebilir cihazlarda ( akıllı telefon vb.) sensör(ivmeölçer, jiroskop vb.) verisi üretilmektedir.
Amaç: Inception ile farklı aktiviteleri sınıflandırma
Aktivitiler: yürüme, ayakta bekleme, uzanma, merdiven çıkma

Dataset:
UCI 
Human Activity Recognition Using Smartphones

- 30 farklı bireyden alınmış segmentlenmiş sensör verisi
- 128 zaman adımlı pencere ile örneklenmiş 9 kanal ((x,y,z) çarpı (body_acc, body_gyro, total_acc))


Yapılabilecekler:
- Train datadaki activitydeki unique değerlere bakmak
- train_data['Activity'].value_counts().sort_values().plot(kind = 'bar', color = 'pink') görselleştirme
- Heatmap
- Data provided by each user x='subject',hue='Activity görselleştirme
- No of Datapoints per Activity görselleştirmesi
- featurlardaki virgül,boşluk, - işaretlerin kaldırılması
"""

import numpy as np
import pandas as pd
import os

def load_inertial_signals(folder_path, subset = "train"):
    """
    folder_path: "UCI HAR Dataset/train/Inertial Signals
    subset: "train" veya "test"
    """
    signal_names = [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
        "total_acc_x",
        "total_acc_y",
        "total_acc_z"
    ]

    signals_data = []
    for signal in signal_names:
        filename = f"{signal}_{subset}.txt"
        filepath = os.path.join(folder_path, filename)
        data = np.loadtxt(filepath)
        signals_data.append(data)

    stacked = np.stack(signals_data, axis = 0).transpose(1,2,0)
    return stacked

X_train = load_inertial_signals("UCI HAR Dataset/train/Inertial Signals", subset = "train")
y_train = np.loadtxt("UCI HAR Dataset/train/y_train.txt").astype(int) -1
print(X_train.shape)
print(y_train.shape)

X_test = load_inertial_signals("UCI HAR Dataset/test/Inertial Signals", subset = "test")
y_test = np.loadtxt("UCI HAR Dataset/test/y_test.txt").astype(int) -1
print(X_test.shape)
print(y_test.shape)

np.save("X_train_raw.npy", X_train)
np.save("X_test_raw.npy", X_test)
np.save("y_train_raw.npy", y_train)
np.save("y_test_raw.npy", y_test)