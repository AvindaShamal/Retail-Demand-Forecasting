from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb


def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    mse = mean_squared_error(y_val, predictions)
    mae = mean_absolute_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)
    return rmse, mse, mae, r2


def plot_feature_importance(model, save_path="feature_importance.png"):
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=15, importance_type="gain")
    plt.savefig(save_path)
    plt.close()
