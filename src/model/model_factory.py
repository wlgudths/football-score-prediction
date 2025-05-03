from model.linear_regression import LinearRegressionModel
from model.random_foreset import RandomForestModel
from model.xgboost_model import XGBoostModel
from model.catboost_model import CatBoostModel
from utils.logger import logger


def get_model(name: str, params: dict = None):
    """
    """
    model_classes = {
        'linear_regression': LinearRegressionModel,
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'catboost': CatBoostModel
        }

    try:
        model_class = model_classes[name]

        return model_class(**params) if params else model_class()

    except KeyError:
        logger.error(f"Unknown model name: '{name}'. Available options: {list(model_classes.keys())}")