from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from model.base_model import BaseModel


class CatBoostModel(BaseModel):
    def __init__(self, params=None):
        base = CatBoostRegressor(verbose=0, **(params or {}))
        self.model = MultiOutputRegressor(base)

    def fit(self, X_train, y_train):
        return self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)