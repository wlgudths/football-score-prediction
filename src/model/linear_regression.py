from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from model.base_model import BaseModel


class LinearRegressionModel(BaseModel):
    def __init__(self, params=None):
        base = LinearRegression(**(params or {}))
        self.model = MultiOutputRegressor(base)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)