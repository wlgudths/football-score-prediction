from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from model.base_model import BaseModel


class XGBoostModel(BaseModel):
    def __init__(self, params=None):
        base = XGBRegressor(**(params or {}))
        self.model = MultiOutputRegressor(base)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)