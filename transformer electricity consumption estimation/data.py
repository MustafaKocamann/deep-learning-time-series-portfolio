import pandas as pd

df = pd.read_csv("household_power_consumption.txt",
                 sep=";",
                 low_memory=False,
                 na_values="?")

import pandas as pd

def check_df(df: pd.DataFrame, head: int = 5, tail: int = 5, quantiles=None) -> None:
    if quantiles is None:
        quantiles = [0.01, 0.05, 0.50, 0.95, 0.99]

    print("=" * 80)
    print("SHAPE")
    print(df.shape)

    print("=" * 80)
    print(f"HEAD ({head})")
    print(df.head(head))

    print("=" * 80)
    print(f"TAIL ({tail})")
    print(df.tail(tail))

    print("=" * 80)
    print("MISSING VALUES (isnull().sum())")
    na = df.isnull().sum()
    na = na[na > 0].sort_values(ascending=False)
    if na.empty:
        print("Eksik değer yok.")
    else:
        missing_df = pd.DataFrame({
            "missing_count": na,
            "missing_ratio": (na / len(df)).round(4)
        })
        print(missing_df)

    print("=" * 80)
    print("DESCRIBE")

    desc = df.describe(include="all", percentiles=quantiles).T
    print(desc)

    print("=" * 80)
    
check_df(df)

df.info()

df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S")

print(df.head())

df = df[["datetime", "Global_active_power"]].dropna()
df = df.set_index("datetime")


print(df.head())

df = df.resample("1h").mean()

df = df.ffill()

df.to_csv("cleaned_power_consumption.csv")
