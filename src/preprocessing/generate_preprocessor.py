import pandas as pd
import torch
from torch.utils.data import TensorDataset
from src.preprocessing.preprocessor import Preprocessor
from utils.encode_team import encode_team_by_ordinal
from utils.split_dataset import split_by_season, split_x_y

def generate_preprocessing(config: dict):
    '''
    '''
    df_path = config['save']['cleaned_data']
    df = pd.read_csv(df_path)

    target_cols = config['preprocessing']['split']['target']
    drop_cols = config['preprocessing']['split']['drop_cols']
    mode = config['preprocessing']['mode']

    decoders = {}
    team_encoded_df, decoders = encode_team_by_ordinal(df, decoders)

    train_data, val_data, test_data = split_by_season(team_encoded_df)

    train_preprocessor = Preprocessor(train_data, config, decoders)
    train_data, decoders = train_preprocessor.process_features('train')

    val_preprocessor = Preprocessor(val_data, config, decoders)
    val_data, _ = val_preprocessor.process_features('val')

    test_preprocessor = Preprocessor(test_data, config, decoders)
    test_data, _ = test_preprocessor.process_features('test')

    X_train, y_train = split_x_y(train_data, target_cols, drop_cols)
    X_val, y_val = split_x_y(val_data, target_cols, drop_cols)
    X_test, y_test = split_x_y(test_data, target_cols, drop_cols)

    test_df = test_preprocessor.decode(test_data, cols=['home', 'away'])

    if mode == 'ml':
        return X_train, y_train, X_val, y_val, X_test, y_test, test_df, decoders
    
    elif mode == 'dl':
        return (
            _to_tensor_dataset(X_train, y_train),
            _to_tensor_dataset(X_val, y_val),
            _to_tensor_dataset(X_test, y_test),
            test_df,
            decoders
        )
    
    else:
        raise ValueError(f'Unknown preprocessing mode: {mode}')


def _to_tensor_dataset(X, y):
    '''
    '''
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    
    return TensorDataset(X_tensor, y_tensor)