import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils.split_dataset import split_by_season, split_X_y
from sklearn.metrics import mean_squared_error, mean_absolute_error

data_path = '/Users/sonjeehyung/Documents/Project/football-score-prediction/data/cleaned_data.csv'
df = pd.read_csv(data_path)
df['attendance'] = df['attendance'].fillna(0)

all_teams = pd.concat([df['home'], df['away']]).unique()

team_encoder = LabelEncoder()
team_encoder.fit(all_teams)

df['home'] = team_encoder.transform(df['home'])
df['away'] = team_encoder.transform(df['away'])

train_df, val_df, test_df = split_by_season(df=df)

target_cols = ['home_score', 'away_score']
drop_cols = ['wk', 'day', 'date', 'venue', 'referee', 'match report', 'season']

X_train, y_train = split_X_y(train_df, target_cols, drop_cols)
X_val, y_val = split_X_y(val_df, target_cols, drop_cols)
X_test, y_test = split_X_y(test_df, target_cols, drop_cols)

numerical_cols = X_train.select_dtypes(include=['int', 'float']).columns

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_val[numerical_cols] = scaler.fit_transform(X_val[numerical_cols])
X_test[numerical_cols] = scaler.fit_transform(X_test[numerical_cols])

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