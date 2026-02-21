from src.config import DATE_COLUMN, SPLIT_DATE
from src.config import TARGET, FEATURE_EXCLUDE


def time_based_split(df):
    df = df.sort_values(DATE_COLUMN)

    train_df = df[df[DATE_COLUMN] < SPLIT_DATE]
    val_df = df[df[DATE_COLUMN] >= SPLIT_DATE]

    return train_df, val_df


def split_features_target(train_df, val_df):
    FEATURES = [col for col in train_df.columns if col not in FEATURE_EXCLUDE]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_val = val_df[FEATURES]
    y_val = val_df[TARGET]

    return X_train, y_train, X_val, y_val
