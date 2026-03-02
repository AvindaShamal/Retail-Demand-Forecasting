import pandas as pd
import yaml

from src.data_processing import time_based_split, split_features_target
from src.train import parameter_tuning, train_and_log, save_model, save_best_params
from src.s3_utils import download_from_s3, upload_to_s3
from data.feature_engineering import run_feature_engineering


def run_training_pipeline(config_path: str) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # transforming raw data and loading to S3
    bucket = config["aws"]["s3_bucket"]
    raw_key = config["aws"]["raw_data_key"]
    download_from_s3(bucket, raw_key, local_path="data/raw_data.csv")
    df = pd.read_csv("data/raw_data.csv", parse_dates=["Date"])
    df = run_feature_engineering(df)
    data_path = config["data"]["processed_data_path"]
    df.to_csv(data_path, index=False)
    
    # training and logging with MLflow
    df = pd.read_csv(data_path, parse_dates=["Date"])
    train_df, val_df = time_based_split(df)
    X_train, y_train, X_val, y_val = split_features_target(train_df, val_df)

    best_params = parameter_tuning(X_train, y_train, X_val, y_val)
    model, mae, rmse = train_and_log(X_train, y_train, X_val, y_val, best_params)
    save_model(model, model_path=config["model"]["output_path"])
    save_best_params(best_params, params_path=config["model"]["params_output_path"])

    print(f"Training completed. MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    upload_to_s3(data_path, config["aws"]["s3_bucket"], "processed_data.csv")
