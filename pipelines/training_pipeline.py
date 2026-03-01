import pandas as pd
import yaml

from src.data_processing import time_based_split, split_features_target
from src.train import parameter_tuning, train_and_log, save_model, save_best_params
from src.s3_utils import upload_to_s3


def run_training_pipeline(config_path: str) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["processed_data_path"]
    df = pd.read_csv(data_path, parse_dates=["Date"])
    train_df, val_df = time_based_split(df, test_size=config["data"]["test_size"])
    X_train, y_train, X_val, y_val = split_features_target(train_df, val_df)

    best_params = parameter_tuning(X_train, y_train, X_val, y_val)
    model, mae, rmse = train_and_log(X_train, y_train, X_val, y_val, best_params)
    save_model(model, model_path=config["model"]["output_path"])
    save_best_params(best_params, params_path=config["model"]["params_output_path"])

    print(f"Training completed. MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    upload_to_s3(data_path, config["aws"]["s3_bucket"], "processed_data.csv")
