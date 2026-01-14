import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("secom.data", sep = r"\s+", header=None)
print(df.head())

labels = pd.read_csv("secom_labels.data", sep = r"\s+", header = None)

df["label"] = labels[0].values

print(df.shape)

print(f"Class distrubution:\n{df['label'].value_counts()}")

missing_per_column = df.isnull().sum()
total_missing = missing_per_column.sum()
print(f"Total missing values: {total_missing}")

missing_ratio = (missing_per_column / len(df)) * 100
missing_ratio = missing_ratio[missing_ratio > 0].sort_values(ascending=False)
print("Missing value ratio per column")

## visualization of missing data

plt.figure()
sns.barplot(x = missing_ratio.index, y = missing_ratio.values)
plt.title("Columns with missing data (%)")
plt.xlabel("Feature Index")
plt.ylabel("Missing Data Ratio (%)")
plt.xticks([])
plt.tight_layout()
plt.show()

plt.figure()
sns.scatterplot(data = df, x = 0, y = 1, hue = "label", alpha = 0.5, palette="Set1")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.tight_layout()
plt.show()