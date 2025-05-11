from torch import optim
from utils.logger import logger


def get_scheduler(optimizer: optim.Optimizer, config: dict):
    '''
    '''
    name = config['scheduler']['name']
    params = config['scheduler'][name]

    if name == 'step_lr':
        return optim.lr_scheduler.StepLR(optimizer, **params)
    elif name == 'cosine_annealing':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    elif name == 'reduce_lr_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    else:
        logger.error(f'Unknown Scheduler: {name}')