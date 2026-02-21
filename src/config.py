TARGET = "Weekly_Sales"
DATE_COLUMN = "Date"

SPLIT_DATE = "2012-01-01"

FEATURE_EXCLUDE = ["Date", TARGET]

PARAM_GRID = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 300, 500, 700],
}

MODEL_BASE_PARAMS = {
    "objective": "reg:squarederror",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "rmse",
    "verbosity": 0,
}

EARLY_STOPPING_ROUNDS = 10

GRID_SEARCH_SETTINGS = {
    "cv": 3,
    "scoring": "neg_root_mean_squared_error",
    "n_jobs": -1,
}
