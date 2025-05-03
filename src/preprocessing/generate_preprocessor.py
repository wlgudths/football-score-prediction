import pandas as pd
from preprocessing.preprocessor import Preprocessor
from utils.encode_team import encode_team_by_ordinal
from utils.split_dataset import split_by_season, split_x_y

def generate_preprocessing(df: pd.DataFrame, config: dict):
    '''
    '''
    target_cols = config['preprocessing']['split']['target']
    drop_cols = config['preprocessing']['split']['drop_cols']

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

    return X_train, y_train, X_val, y_val, X_test, y_test