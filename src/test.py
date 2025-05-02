import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from model.model_factory import LinearRegressionModel, RandomForestModel, XGBoostModel, CatBoostModel

from preprocessing.generate_preprocessor import generate_preprocessing
from utils.config_utils import load_config

data_path = '/Users/sonjeehyung/Documents/Project/football-score-prediction/data/cleaned_data.csv'
config_path = '/Users/sonjeehyung/Documents/Project/football-score-prediction/config/config.yaml'

df = pd.read_csv(data_path)
config = load_config(config_path)

X_train, y_train, X_val, y_val, X_test, y_test = generate_preprocessing(df, config)

lr_model = LinearRegressionModel()
rf_model = RandomForestModel()
xgb_model = XGBoostModel()
cat_model = CatBoostModel()

models = {
    'LinearRegression': lr_model,
    'RandomForest': rf_model,
    'XGBoost': xgb_model,
    'CatBoost': cat_model
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    results.append({
        'model': name,
        'rmse': rmse,
        'mse': mse,
        'mae': mae
        })

print(results)