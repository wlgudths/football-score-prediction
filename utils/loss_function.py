import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from utils.logger import logger

def get_loss_function_dl(config: dict):
    '''
    '''
    name = config['loss']['name']

    if name == 'mse':
        return nn.MSELoss()
    elif name == 'mae':
        return nn.L1Loss()
    elif name == 'smooth_l1':
        return nn.SmoothL1Loss()
    else:
        logger.error(f"Unknown loss function (DL): {name}")
        

def get_loss_function_ml(config: dict):
    name = config['loss']['name']

    if name == 'mae':
        return lambda y_true, y_pred: mean_absolute_error(y_true, y_pred)
    elif name == 'mse':
        return lambda y_true, y_pred: mean_squared_error(y_true, y_pred)
    elif name == 'rmse':
        return lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred)
    elif name == 'smooth_l1':
        return lambda y_true, y_pred: smooth_l1(y_true, y_pred)
    else:
        raise ValueError(f"[Loss] Unknown ML loss function: {name}")

def smooth_l1(y_true, y_pred, beta=1.0):
    diff = np.abs(y_true - y_pred)
    return np.mean(np.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta))

