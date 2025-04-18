import pandas as pd
from utils.save_utils import save_to_csv

def split_by_season(df: pd.DataFrame,
                    train_until: int = 2020,
                    val_year: int = 2021,
                    test_from: int = 2022,
                    save: bool = False,
                    cfg: dict = None) -> pd.DataFrame:
    """
    Summary
    """
    train_df = df[df['season'] <= train_until]
    val_df = df[df['season'] == val_year]
    test_df = df[df['season'] >= test_from]

    if save:
        save_to_csv(df=train_df, config=cfg, file_name='train.csv', save_key='train')
        save_to_csv(df=val_df, config=cfg, file_name='val.csv', save_key='val')
        save_to_csv(df=test_df, config=cfg, file_name='test.csv', save_key='test')
    
    return train_df, val_df, test_df


def split_X_y(df: pd.DataFrame, target_cols: list[str], drop_cols: list[str] = []) -> pd.DataFrame:
    """
    """
    X = df.drop(columns=target_cols + drop_cols, errors='ignore')
    y = df[target_cols]
    return X, y