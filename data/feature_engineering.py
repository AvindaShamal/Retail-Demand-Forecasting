import pandas as pd

df = pd.read_csv("data/raw/train.csv")
df["Date"] = pd.to_datetime(df["Date"])


def fill_missing_weeks(group):
    store_id = group["Store"].iloc[0]
    dept_id = group["Dept"].iloc[0]
    group = group.sort_values("Date")
    full_weeks = pd.date_range(
        start=group["Date"].min(),
        end=group["Date"].max(),
        freq="W-FRI",  # Walmart weeks end on Friday
    )
    group = group.set_index("Date").reindex(full_weeks)
    group.index.name = "Date"
    group["Weekly_Sales"] = group["Weekly_Sales"].fillna(0)
    group["Store"] = store_id
    group["Dept"] = dept_id
    return group.reset_index()


df = df.groupby(["Store", "Dept"], group_keys=False).apply(fill_missing_weeks)

LAGS = [1, 2, 3, 4]
for lag in LAGS:
    df[f"Sales_Lag_{lag}"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(lag)

ROLLING_WINDOWS = [4, 8]
for window in ROLLING_WINDOWS:
    df[f"Sales_Rolling_Mean_{window}"] = (
        df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1).rolling(window).mean()
    )

df["IsHoliday"] = df["IsHoliday"].fillna(False).astype(int)
df["is_pre_holiday"] = df.groupby("Store")["IsHoliday"].shift(-1).fillna(0).astype(int)
df["is_post_holiday"] = df.groupby("Store")["IsHoliday"].shift(1).fillna(0).astype(int)
df["Store"] = df["Store"].astype(int)
df["Dept"] = df["Dept"].astype(int)
df = df.dropna().reset_index(drop=True)

df.to_csv("data/processed/feature_engineered_data.csv", index=False)
