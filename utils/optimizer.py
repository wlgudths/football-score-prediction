import torch
from torch import optim
from utils.logger import logger


def get_optimizer(model: torch.nn.Module, config: dict):
    '''
    '''
    name = config['optimizer']['name']
    params = config['optimizer'][name]

    if name == 'adam':
        return optim.Adam(model.parameters(), **params)
    elif name == 'adamw':
        return optim.AdamW(model.parameters(), **params)
    else:
        logger.error(f'Unknown Optimizer: {name}')
