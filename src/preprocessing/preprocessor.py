import pandas as pd


class Preprocessor():
    """
    """
    def __init__(self, config, df):
        self.config = config
        self.df = df
    
    def impute_missing_value(self) -> pd.DataFrame:
        """
        """
        imputation_cfg = self.config['preprocessing'].get('imputation', {})
        col = imputation_cfg['col']
        group = imputation_cfg['group']
        strategy = imputation_cfg['strategy']

        if strategy == 'zero':
            self.df[col] = self.df[col].fillna(0)
        elif strategy == 'mean':
            self.df[col] = self.df.groupby(group)[col].transform(lambda x: x.fillna(x.mean()))
        elif strategy == 'median':
            self.df[col] = self.df.groupby(group)[col].transform(lambda x: x.fillna(x.median()))
        else:
            raise

        return self.df
    
    
    
            



