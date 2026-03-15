import mlflow.pyfunc as pyfunc
import yaml
import logging
import pandas as pd

from src.s3_utils import upload_to_s3


def load_model_from_registry(
    model_name: str, model_version: str = "Staging"
) -> pyfunc.PyFuncModel:
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

    # Load model from MLflow registry
    model = load_model_from_registry(
        model_name=config["model"]["registry_name"],
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
