import pandas as pd
import numpy as np
import pandas as pd

def clean_market_value_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    €10.69m, €1.01b 등의 문자열을 float 숫자 (단위: 유로)로 변환합니다.
    단위: b = 1,000,000,000 / m = 1,000,000 / k = 1,000
    """
    def convert(value):
        if not isinstance(value, str):
            return np.nan

        val = value.lower().replace('€', '').replace(',', '').strip()

        if val == '–' or val == '' or val == 'n/a':
            return np.nan

        try:
            if val.endswith('b'):
                return float(val[:-1]) * 1_000_000_000
            elif val.endswith('m'):
                return float(val[:-1]) * 1_000_000
            elif val.endswith('k'):
                return float(val[:-1]) * 1_000
            else:
                return float(val)  # 혹시 단위 없는 경우
        except ValueError:
            return np.nan

    df[column] = df[column].apply(convert)
    
    return df
