import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score


def evaluate_df(df: pd.DataFrame, n_features: int = None) -> pd.DataFrame:
    metrics = {}

    for target in ['home', 'away']:
        y_true = df[f'true_{target}_score'].values
        y_pred = df[f'pred_{target}_score'].values

        metrics[f'{target}_MAE'] = mean_absolute_error(y_true, y_pred)
        metrics[f'{target}_MSE'] = mean_squared_error(y_true, y_pred)
        metrics[f'{target}_RMSE'] = root_mean_squared_error(y_true, y_pred)
        metrics[f'{target}_R2'] = r2_score(y_true, y_pred)

        if n_features is not None:
            metrics[f'{target}_AdjR2'] = adjusted_r2_score(y_true, y_pred, n_features)

        metrics[f'{target}_SmoothL1'] = F.smooth_l1_loss(
                torch.tensor(y_pred, dtype=torch.float32),
                torch.tensor(y_true, dtype=torch.float32),
                reduction='mean'
            ).item()

    y_true_all = df[['true_home_score', 'true_away_score']].values.flatten()
    y_pred_all = df[['pred_home_score', 'pred_away_score']].values.flatten()

    metrics['avg_MAE'] = mean_absolute_error(y_true_all, y_pred_all)
    metrics['avg_MSE'] = mean_squared_error(y_true_all, y_pred_all)
    metrics['avg_RMSE'] = root_mean_squared_error(y_true_all, y_pred_all)
    metrics['avg_R2'] = r2_score(y_true_all, y_pred_all)

    metrics['avg_SmoothL1'] = F.smooth_l1_loss(
            torch.tensor(y_pred_all, dtype=torch.float32),
            torch.tensor(y_true_all, dtype=torch.float32),
            reduction='mean'
        ).item()
    
    return pd.DataFrame([metrics])


def adjusted_r2_score(y_true, y_pred, n_features: int) -> float:
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
