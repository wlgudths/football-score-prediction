import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def clean_market_value_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    시장 가치 컬럼을 float 숫자 (유로)로 변환합니다.
    문자열 예: '€10.69m', '€1.01bn', '€800k'
    단위: bn = 1,000,000,000 / m = 1,000,000 / k = 1,000
    """
    def convert(value):
        if not isinstance(value, str):
            return np.nan

        val = value.lower().replace('€', '').replace(',', '').strip()

        if val in ['–', '-', '', 'n/a', 'nan']:
            return np.nan

        try:
            if val.endswith('bn'):
                return float(val[:-2]) * 1_000_000_000
            elif val.endswith('m'):
                return float(val[:-1]) * 1_000_000
            elif val.endswith('k'):
                return float(val[:-1]) * 1_000
            else:
                return float(val)  # 단위가 없는 경우
        except ValueError:
            return np.nan

    df[column] = df[column].apply(convert)
    return df