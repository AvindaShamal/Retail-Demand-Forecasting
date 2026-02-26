import json
import os

import joblib
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.config import (
    EARLY_STOPPING_ROUNDS,
    GRID_SEARCH_SETTINGS,
    MODEL_BASE_PARAMS,
    N_CV_SPLITS,
    PARAM_GRID,
)


def train_model(X_train, y_train, X_val, y_val):
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    base_model = xgb.XGBRegressor(**MODEL_BASE_PARAMS)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        cv=tscv,
        scoring=GRID_SEARCH_SETTINGS.get("scoring", "neg_root_mean_squared_error"),
        n_jobs=GRID_SEARCH_SETTINGS.get("n_jobs", -1),
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    cv_results = grid_search.cv_results_
    for params, mean_score, std_score in zip(
        cv_results["params"],
        cv_results["mean_test_score"],
        cv_results["std_test_score"],
    ):
        rmse = -mean_score
        mse = rmse**2
        print(f"Params: {params} -> CV RMSE: {rmse:.4f}, CV MSE: {mse:.4f}")

    best_params = grid_search.best_params_
    print(f"\nBest CV parameters: {best_params}")
    print(f"Best CV RMSE: {-grid_search.best_score_:.4f}")

    tuned_params = {**MODEL_BASE_PARAMS, **best_params}
    tuned_params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS

    model = xgb.XGBRegressor(**tuned_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

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
