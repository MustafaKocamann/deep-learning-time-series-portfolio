import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("AirQualityUCI.csv", sep = ";", decimal = ",", encoding = "latin")
print(df.head())

## Remove empty columns 
df.dropna(axis = 1, how = "all", inplace = True)

df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format = "%d/%m/%Y %H.%M.%S", errors = "coerce")
print(df.head())

## remove date error rows 
df.dropna(subset = ["datetime"], inplace = True)

## remove original date and time columns
df.drop(["Date", "Time"], axis = 1, inplace = True)
df.set_index("datetime", inplace = True)
print(df.head())

## error sensor values to NaN
df.replace(-200, np.nan, inplace = True)

## interpolate missing values
df.interpolate(method = "time", inplace = True)

## feature engineering 
df["hour"] = df.index.hour
df["month"] = df.index.month
print(df.head())

## input and target variable
selected_columns = ["NO2(GT)", "T", "RH", "AH", "CO(GT)", "hour", "month"]
df = df[selected_columns]
print(df.head())

## missing values check
print(f"Eksik Değer:{df.isnull().sum()}")

## corelation matrix
plt.figure()
sns.heatmap(df.corr(), annot = True, cmap = "YlGnBu", fmt = ".2f")
plt.title("Correlation Matrix")
plt.show()