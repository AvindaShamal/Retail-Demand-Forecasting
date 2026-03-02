import json
import os
import mlflow
import yaml

import joblib
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from src.evaluate import plot_feature_importance


def parameter_tuning(X_train, y_train, X_val, y_val, config_path: str = "configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    tscv = TimeSeriesSplit(n_splits=config["model"]["n_cv_splits"])
    base_model = xgb.XGBRegressor(**config["model"]["model_base_params"])
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=config["model"]["param_grid"],
        cv=tscv,
        scoring=config["model"]["grid_search_settings"].get("scoring", "neg_root_mean_squared_error"),
        n_jobs=config["model"]["grid_search_settings"].get("n_jobs", -1),
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    cv_results = grid_search.cv_results_
    for params, mean_score in zip(
        cv_results["params"],
        cv_results["mean_test_score"],
    ):
        rmse = -mean_score
        mae = mean_absolute_error(y_val, grid_search.predict(X_val))
        print(f"Params: {params} -> CV RMSE: {rmse:.4f}, CV MAE: {mae:.4f}")

    best_params = grid_search.best_params_
    print(f"\nBest CV parameters: {best_params}")
    print(f"Best CV RMSE: {-grid_search.best_score_:.4f}")

    tuned_params = {**config["model"]["model_base_params"], **best_params}
    tuned_params["early_stopping_rounds"] = config["model"]["early_stopping_rounds"]

    return tuned_params


def train_and_log(X_train, y_train, X_val, y_val, params):
    with mlflow.start_run():
        mlflow.log_params(params)

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = root_mean_squared_error(y_val, preds)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        plot_feature_importance(model)
        mlflow.log_artifact("feature_importance.png")

        mlflow.xgboost.log_model(
            model, name="xgb_model", registered_model_name=config["model"]["registry_name"]
        )

    return model, mae, rmse


def save_model(model, model_path="models/xgb_model.pkl"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)


def save_best_params(best_params, params_path="models/best_params.json"):
    directory = os.path.dirname(params_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(params_path, "w", encoding="utf-8") as fp:
        json.dump(best_params, fp, indent=2, ensure_ascii=True)
