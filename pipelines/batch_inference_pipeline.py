import logging
import mlflow
import mlflow.pyfunc as pyfunc
import os
import pandas as pd
import yaml

from src.s3_utils import upload_to_s3


def load_model_from_registry(
    model_name: str, model_version: str = "Staging", tracking_uri: str | None = None
) -> pyfunc.PyFuncModel:
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    if tracking_uri is None:
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"]
        )
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{model_name}/{model_version}"
    model = pyfunc.load_model(model_uri)
    return model


def get_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def run_batch_inference(config_path: str) -> None:
    logger = get_logger()
    logger.info("Starting batch inference pipeline.")

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    # Load model from MLflow registry
    model = load_model_from_registry(
        model_name=config["model"]["registry_name"], tracking_uri=tracking_uri
    )

    # Load new data for inference
    inference_data_path = config["data"]["inference_data_path"]
    df = pd.read_csv(inference_data_path, parse_dates=["Date"])

    predictions = model.predict(df)
    df["Forecasted_Demand"] = predictions

    # Save predictions to CSV
    output_path = config["output"]["predictions_output_path"]
    df.to_csv(output_path, index=False)
    upload_to_s3(
        output_path, config["aws"]["s3_bucket"], "predictions/batch_predictions.csv"
    )

    print(f"Batch inference completed. Predictions saved to {output_path}")
