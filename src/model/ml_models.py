from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from utils.logger import logger


def get_model(config: dict):
    """
    """
    model_name = config['model']['name']
    params = config['model']['params'][model_name]

    if model_name == 'linear_regression':
        model = LinearRegression(**params)
    elif model_name == 'random_forest':
        model = MultiOutputRegressor(RandomForestRegressor(**params))
    elif model_name == 'xgboost':
        model = MultiOutputRegressor(XGBRegressor(**params))
    elif model_name == 'catboost':
        model = MultiOutputRegressor(CatBoostRegressor(**params))
    else:
        logger.error(f'Unknown model model_name : {model_name}')

    return model