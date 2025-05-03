from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from model.base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, params=None):
        base = RandomForestRegressor(**(params or {}))
        self.model = MultiOutputRegressor(base)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)