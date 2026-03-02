def time_based_split(df, config: dict) -> tuple:
    df = df.sort_values(config["data"]["date_column"])

    train_df = df[df[config["data"]["date_column"]] < config["data"]["split_date"]]
    val_df = df[df[config["data"]["date_column"]] >= config["data"]["split_date"]]

    return train_df, val_df


def split_features_target(train_df, val_df, config: dict) -> tuple:
    FEATURES = [
        col for col in train_df.columns if col not in config["data"]["feature_exclude"]
    ]

    X_train = train_df[FEATURES]
    y_train = train_df[config["data"]["target"]]
    X_val = val_df[FEATURES]
    y_val = val_df[config["data"]["target"]]

    return X_train, y_train, X_val, y_val
