from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    mae = mean_absolute_error(y_val, predictions)
    return rmse, mae
