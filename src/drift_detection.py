import pandas as pd


def detect_drift(
    train_df: pd.DataFrame, new_df: pd.DataFrame, threshold: float = 0.2
) -> list:

    drift_columns = []
    for column in train_df.columns:
        if column == "Weekly_Sales":
            continue

        train_mean = train_df[column].mean()
        new_mean = new_df[column].mean()

        diff = abs(train_mean - new_mean) / (abs(train_mean) + 1e-6)
        if diff > threshold:
            drift_columns.append(column)

    return drift_columns
