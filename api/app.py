from fastapi import FastAPI
import yaml

from pipelines.batch_inference_pipeline import load_model_from_registry

app = FastAPI()
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)
model = load_model_from_registry(
    model_name=config["model"]["registry_name"], model_version="Production"
)


@app.get("/predict")
def predict(store: int, dept: int, week: str):
    import pandas as pd

    input_data = pd.DataFrame(
        {
            "Store": [store],
            "Dept": [dept],
            "Date": [pd.to_datetime(week)],
        }
    )
    prediction = model.predict(input_data)
    return {"predicted_sales": float(prediction[0])}
