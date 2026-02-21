import pandas as pd

from src.data_processing import time_based_split, split_features_target
from src.train import save_best_params, save_model, train_model
from src.evaluate import evaluate_model


def main():
    df = pd.read_csv("data/processed/feature_engineered_data.csv", parse_dates=["Date"])

    train_df, val_df = time_based_split(df)
    X_train, y_train, X_val, y_val = split_features_target(train_df, val_df)

    model, best_params = train_model(X_train, y_train, X_val, y_val)
    save_model(model)
    save_best_params(best_params)

    rmse, mae = evaluate_model(model, X_val, y_val)
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE: {mae:.4f}")


if __name__ == "__main__":
    main()
