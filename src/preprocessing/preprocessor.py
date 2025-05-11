import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler


class Preprocessor():
    """
    """
    def __init__(self, df: pd.DataFrame, config: dict, decoders: dict):
        self.config = config
        self.data = df
        self.decoders = decoders

    def remove_outliers(self, col: str) -> pd.DataFrame:
        """
        """
        k = abs(kurtosis(self.data[col].dropna()))
        
        if k > 3:
            robust_scaler = RobustScaler()

            self.data.loc[:, [col]] = robust_scaler.fit_transform(self.data[[col]])
            self.decoders[col] = {"outliers" : robust_scaler}
        
        return self.data

    def impute_missing_value(self) -> pd.DataFrame:
        """
        """
        imputation_cfg = self.config['preprocessing']['imputation']
        cols = imputation_cfg['col']
        group = imputation_cfg['group']
        strategy = imputation_cfg['strategy']

        for col in cols:
            if strategy == 'zero':
                self.data.loc[:, col] = self.data[col].fillna(0)
            elif strategy == 'mean':
                self.data.loc[:, col] = self.data.groupby(group)[col].transform(lambda x: x.fillna(x.mean()))
            elif strategy == 'median':
                self.data.loc[:, col] = self.data.groupby(group)[col].transform(lambda x: x.fillna(x.median()))
            else:
                raise
                    
        return self.data
    
    def numerical_features(self, col: str, method: str):
        """
        수치형 변수를 정규화하는 함수.
        - train: 스케일러 학습 및 저장
        - val/test: 저장된 스케일러로 변환
        """
        self.data = self.data.copy()
        self.data[col] = self.data[col].astype(float)

        if method == 'train':
            skewness = abs(skew(self.data[col].dropna()))

            if skewness >= 1:
                transformer = PowerTransformer(method='yeo-johnson', standardize=True)
            else:
                transformer = StandardScaler()

            transformed = transformer.fit_transform(self.data[[col]])
            self.data.loc[:, col] = transformed

            if col in self.decoders and "outliers" in self.decoders[col]:
                outlier_scaler = self.decoders[col]["outliers"]
                self.decoders[col] = {"encoder": [outlier_scaler, transformer]}
            else:
                self.decoders[col] = {"encoder": transformer}

        else:  # val/test
            if col not in self.decoders:
                raise ValueError(f"No encoder found for column '{col}' in decoders.")

            encoder = self.decoders[col]["encoder"]

            if isinstance(encoder, list):
                outlier_scaler, scaler = encoder
                scaled = outlier_scaler.transform(self.data[[col]])
                self.data.loc[:, col] = scaler.transform(pd.DataFrame(scaled, columns=[col]))
            else:
                self.data.loc[:, col] = encoder.transform(self.data[[col]])

    def decode(self, df: pd.DataFrame, cols: list, round_decimals: int=2) -> pd.DataFrame:
        """
        """
        for col in cols:
            if col not in self.decoders:
                continue
        
            decoder_info = self.decoders[col]
            encoder = decoder_info.get('encoder', None)

            if isinstance(encoder, dict):
                df[col] = df[col].map(encoder)
    
            elif isinstance(encoder, list):
                outlier_scaler, scaler = encoder
                inv = scaler.inverse_transform(df[[col]])
                inv = outlier_scaler.inverse_transform(inv)
                df[[col]] = np.round(inv, round_decimals)

            elif encoder is not None:
                df.loc[:, [col]] = np.round(encoder.inverse_transform(df[[col]]), round_decimals)
        
        return df
    
    def process_features(self, method: str):
        """
        """
        columns = list(self.data.columns)
        skip_cols = (
            self.config['preprocessing']['categorical']['cols'] +
            self.config['preprocessing']['split']['target'] +
            self.config['preprocessing']['split']['drop_cols']
            )

        for col in columns:
            if col in skip_cols:
                continue
            
            elif col in self.config['preprocessing']['imputation']['col']:
                self.impute_missing_value()
                self.remove_outliers(col)
                self.numerical_features(col, method)

            else:
                self.remove_outliers(col)
                self.numerical_features(col, method)

        return self.data, self.decoders