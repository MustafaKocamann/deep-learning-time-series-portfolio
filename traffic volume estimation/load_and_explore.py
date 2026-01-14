import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. VERİ YÜKLEME
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

def check_df(df, head=5):
    print("##################### Shape #####################")
    print(df.shape)
    print("##################### Types #####################")
    print(df.dtypes)
    print("##################### Head #####################")
    print(df.head(head))
    print("##################### Tail #####################")
    print(df.tail(head))
    print("##################### NA #####################")
    print(df.isnull().sum())
    print("##################### Quantiles #####################")
    print(df.describe().T)
    print(df.info()) 
    print(df.duplicated().sum())

check_df(df)

# Tarih formatı ve Index
df["date_time"] = pd.to_datetime(df["date_time"])
df.set_index("date_time", inplace=True)

# Duplicate Temizliği
df = df[~df.index.duplicated(keep='first')]

print(df.head())
print("Data Range:", df.index.min(), "-", df.index.max())

# --- MEVCUT GÖRSELLEŞTİRMELER ---

plt.figure(figsize=(10, 5))
plt.plot(df["traffic_volume"], label="Traffic Volume", color="steelblue")
plt.xlabel("Date")
plt.ylabel("Traffic Volume")
plt.title("1. Genel Bakış: Traffic Volume Over Time")
plt.legend()
plt.show()

df["hour"] = df.index.hour
hourly_avg = df.groupby("hour")["traffic_volume"].mean()

plt.figure(figsize=(10, 5))
sns.barplot(x=hourly_avg.index, y=hourly_avg.values, palette="viridis", hue=hourly_avg.index, legend=False)
plt.title("2. Saatlik Analiz: Average Traffic Volume by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Traffic Volume")
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df["traffic_volume"], kde=True, color="darkblue", bins=50)
plt.title("Traffic Volume Distribution")
plt.xlabel("Traffic Volume")
plt.ylabel("Frequency")
plt.show()

fig, ax = plt.subplots(figsize=(10,5))
last = 1000
sns.lineplot(x=df.index[-last:], y=df['traffic_volume'].values[-last:])
plt.title(f'Traffic V/S Time | Last {last} samples')
plt.show()

print(df["holiday"].value_counts())

fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(df['holiday'][df['holiday']!='Work Day'])
plt.title('Holiday Distribution')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,5))
mask = df.holiday != 'Work Day'

sns.boxplot(
    x=df['traffic_volume'][mask].values, 
    y=df['holiday'][mask].values,
    palette="Set2",
    hue=df['holiday'][mask].values, 
    legend=False
)
plt.title('Traffic Volume on Holidays')
plt.show()

# Yeni Featurelar
df['month'] = df.index.month
df['day_of_month'] = df.index.day
df['day_of_week'] = df.index.dayofweek

plt.figure(figsize=(14, 6))
sns.violinplot(x='month', y='traffic_volume', data=df, palette="coolwarm", hue='month', legend=False)
plt.title('Traffic Volume by Months (Aylık Dağılım)')
plt.xlabel('Month')
plt.ylabel('Traffic Volume')
plt.show()

plt.figure(figsize=(14, 6))
sns.violinplot(x='day_of_week', y='traffic_volume', data=df, palette="Wistia", hue='day_of_week', legend=False)
plt.title('Traffic Volume by Day of Week (0:Mon - 6:Sun)')
plt.xlabel('Day of Week')
plt.ylabel('Traffic Volume')
plt.show()

daily_avg_values = df.groupby('day_of_month')['traffic_volume'].mean()

plt.figure(figsize=(12, 6))
plt.plot(daily_avg_values.index, daily_avg_values.values, linewidth=3, color='#008fd5')
plt.title('Mean Traffic Volume V/S Day of Month (Ayın Günleri Ortalaması)')
plt.xlabel('Day of Month (1-31)')
plt.ylabel('Average Traffic Volume')
plt.grid(True, alpha=0.3)
plt.show()

# (Weather Main)
plt.figure(figsize=(12, 6))
sns.countplot(
    x='weather_main', 
    data=df, 
    palette="Set2", 
    hue='weather_main', 
    legend=False,
    order=df['weather_main'].value_counts().index 
)
plt.title('Weather Main Distribution')
plt.xticks(rotation=45)
plt.show()

# (Weather Description)
plt.figure(figsize=(12, 10)) 
sns.countplot(
    y='weather_description', 
    data=df, 
    palette="viridis", 
    hue='weather_description', 
    legend=False,
    order=df['weather_description'].value_counts().index
)
plt.title('Weather Description Distribution')
plt.show()


# 1. ZAMAN SERİSİ AYRIŞTIRMASI (DECOMPOSITION) 

df_filled = df.asfreq('H', method='ffill') 
subset = df_filled['traffic_volume'].tail(720) # Son 1 ay

decomposition = seasonal_decompose(subset, model='additive', period=24) 

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
decomposition.observed.plot(ax=ax1, color='blue')
ax1.set_title('Zaman Serisi Ayrıştırması (Son 1 Ay)')
ax1.set_ylabel('Observed')

decomposition.trend.plot(ax=ax2, color='orange')
ax2.set_ylabel('Trend')

decomposition.seasonal.plot(ax=ax3, color='green')
ax3.set_ylabel('Seasonality')

decomposition.resid.plot(ax=ax4, color='red')
ax4.set_ylabel('Residuals')
plt.tight_layout()
plt.show()

# 2. OTOKORELASYON (ACF/PACF)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
plot_acf(df['traffic_volume'].dropna(), lags=48, ax=ax1)
ax1.set_title("Autocorrelation (ACF)")
plot_pacf(df['traffic_volume'].dropna(), lags=48, ax=ax2)
ax2.set_title("Partial Autocorrelation (PACF)")
plt.tight_layout()
plt.show()

# 3. HEATMAP

pivot_table = df.pivot_table(values="traffic_volume", index="day_of_week", columns="hour", aggfunc="mean")
plt.figure(figsize=(14, 6))
sns.heatmap(pivot_table, cmap="coolwarm", annot=False)
plt.title("Isı Haritası: Gün ve Saat Kırılımı")
plt.show()

# 4. WEEKLY RESAMPLING
weekly_trend = df['traffic_volume'].resample('W').mean()
plt.figure(figsize=(10, 6))
weekly_trend.plot(color='purple', linewidth=2)
plt.title("Haftalık Ortalama Trend")
plt.grid(True, alpha=0.3)
plt.show()