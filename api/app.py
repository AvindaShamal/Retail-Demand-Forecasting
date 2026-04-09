from fastapi import FastAPI, HTTPException
import pandas as pd
import yaml

from pipelines.batch_inference_pipeline import load_model_from_registry
from src.mlflow_utils import get_tracking_uri

app = FastAPI()


@app.on_event("startup")
def load_model() -> None:
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    tracking_uri = get_tracking_uri(config["mlflow"]["tracking_uri"])

    try:
        app.state.model = load_model_from_registry(
            model_name=config["model"]["registry_name"],
            model_version="Production",
            tracking_uri=tracking_uri,
        )
        app.state.tracking_uri = tracking_uri
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the production model from MLflow. "
            f"Resolved tracking URI: {tracking_uri}. "
            "If MLflow is running on the host machine, make sure it is reachable from Docker."
        ) from exc


@app.get("/health")
def health():
    return {"status": "ok", "tracking_uri": getattr(app.state, "tracking_uri", None)}


@app.get("/predict")
def predict(store: int, dept: int, week: str):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    input_data = pd.DataFrame(
        {
            "Store": [store],
            "Dept": [dept],
            "Date": [pd.to_datetime(week)],
        }
    )
    prediction = model.predict(input_data)
    return {"predicted_sales": float(prediction[0])}
