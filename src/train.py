import json
import os

import joblib
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.config import (
    EARLY_STOPPING_ROUNDS,
    GRID_SEARCH_SETTINGS,
    MODEL_BASE_PARAMS,
    PARAM_GRID,
)


def train_model(X_train, y_train, X_val, y_val):
    base_model = xgb.XGBRegressor(**MODEL_BASE_PARAMS)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        cv=GRID_SEARCH_SETTINGS.get("cv", 3),
        scoring=GRID_SEARCH_SETTINGS.get("scoring", "neg_root_mean_squared_error"),
        n_jobs=GRID_SEARCH_SETTINGS.get("n_jobs", -1),
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    tuned_params = {**MODEL_BASE_PARAMS, **best_params}
    tuned_params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS

    model = xgb.XGBRegressor(**tuned_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_rmse = root_mean_squared_error(y_val, model.predict(X_val))
    print(f"Best parameters: {best_params}")
    print(f"Validation RMSE: {val_rmse:.4f}")

    return model, best_params


def save_model(model, model_path="models/xgb_model.pkl"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)


def save_best_params(best_params, params_path="models/best_params.json"):
    directory = os.path.dirname(params_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(params_path, "w", encoding="utf-8") as fp:
        json.dump(best_params, fp, indent=2, ensure_ascii=True)
