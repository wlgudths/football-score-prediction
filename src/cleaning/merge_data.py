from typing import Dict, List, Optional, Union
import pandas as pd


def merge_df(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    how: str = 'left',
    left_on: Optional[List[str]] = None,
    right_on: Optional[List[str]] = None,
    rename_cols: Optional[Dict[str, str]] = None,
    drop_col: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
    """
    Summary:
        두 데이터프레임을 병합하고, 선택적으로 컬럼명을 변경하거나 삭제합니다.

    Args:
        left_df (pd.DataFrame): 병합 기준이 되는 왼쪽 데이터프레임
        right_df (pd.DataFrame): 병합할 오른쪽 데이터프레임
        how (str): 병합 방식 (기본값: 'left')
        left_on (List[str], optional): 왼쪽 DataFrame에서 병합할 기준 컬럼
        right_on (List[str], optional): 오른쪽 DataFrame에서 병합할 기준 컬럼
        rename_cols (Dict[str, str], optional): 병합 후 컬럼명 변경 딕셔너리
        drop_col (str or List[str], optional): 병합 후 제거할 컬럼명

    Returns:
        pd.DataFrame: 병합 및 컬럼 처리된 결과 데이터프레임
    """
    merged = left_df.merge(
        right_df,
        how=how,
        left_on=left_on,
        right_on=right_on
        )

    if rename_cols:
        merged.rename(columns=rename_cols, inplace=True)
    
    if drop_col:
        merged.drop(columns=drop_col, inplace=True)

    return merged