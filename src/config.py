TARGET = "Weekly_Sales"
DATE_COLUMN = "Date"

SPLIT_DATE = "2012-01-01"

FEATURE_EXCLUDE = ["Date", TARGET]

PARAM_GRID = {
    "max_depth": [6, 8],
    "learning_rate": [0.01],
    "n_estimators": [300, 500, 700],
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

N_CV_SPLITS = 5

GRID_SEARCH_SETTINGS = {
    "scoring": "neg_root_mean_squared_error",
    "n_jobs": -1,
}
