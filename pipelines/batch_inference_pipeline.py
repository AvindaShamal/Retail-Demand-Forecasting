import logging
import mlflow
import mlflow.pyfunc as pyfunc
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

from src.mlflow_utils import get_tracking_uri, resolve_tracking_uri
from src.s3_utils import upload_to_s3


def _get_fallback_model_uris(model_name: str, preferred_version: str) -> list[str]:
    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{model_name}'")

    fallback_uris = []
    seen_versions = {preferred_version}
    for version_info in sorted(
        versions, key=lambda item: int(item.version), reverse=True
    ):
        version = str(version_info.version)
        if version in seen_versions or version_info.status != "READY":
            continue
        fallback_uris.append(f"models:/{model_name}/{version}")
        seen_versions.add(version)

    return fallback_uris


def load_model_from_registry(
    model_name: str, model_version: str = "production", tracking_uri: str | None = None
) -> pyfunc.PyFuncModel:
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    if tracking_uri is None:
        tracking_uri = get_tracking_uri(config["mlflow"]["tracking_uri"])
    else:
        tracking_uri = resolve_tracking_uri(tracking_uri)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    candidate_model_uris = [f"models:/{model_name}/{model_version}"]
    candidate_model_uris.extend(_get_fallback_model_uris(model_name, model_version))

    last_error = None
    for model_uri in candidate_model_uris:
        try:
            return pyfunc.load_model(model_uri)
        except Exception as exc:
            last_error = exc
            logging.warning("Failed to load model from %s: %s", model_uri, exc)

    raise RuntimeError(
        f"Unable to load model '{model_name}'. Tried: {candidate_model_uris}"
    ) from last_error


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
    tracking_uri = get_tracking_uri(config["mlflow"]["tracking_uri"])
    # Load model from MLflow registry
    model = load_model_from_registry(
        model_name=config["model"]["registry_name"], tracking_uri=tracking_uri
    )

    # Load new data for inference
    inference_data_path = config["data"]["processed_data_path"]
    df = pd.read_csv(inference_data_path, parse_dates=["Date"])

    # Select only feature columns (exclude Date and target column)
    feature_columns = [
        col for col in df.columns if col not in config["data"]["feature_exclude"]
    ]

    predictions = model.predict(df[feature_columns])
    df["Forecasted_Demand"] = predictions

    # Save predictions to CSV
    output_path = config["output"]["predictions_output_path"]
    df.to_csv(output_path, index=False)
    upload_to_s3(
        output_path, config["aws"]["s3_bucket"], "predictions/batch_predictions.csv"
    )

    print(f"Batch inference completed. Predictions saved to {output_path}")
