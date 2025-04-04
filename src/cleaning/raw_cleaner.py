import pandas as pd
from typing import Optional
from utils.logger import logger

class RawDataCleaner:
    def __init__(self, df, df_name):
        self.df = df.copy()
        self.df_name = df_name
    
    def __repr__(self):
        return (
            f"DataCleaner Summary\n"
            f"-------------------\n"
            f"Date     : {self.df_name}\n"
            f"Shape    : {self.df.shape}\n"
            f"Columns  : {list(self.df.columns)}\n\n"
            f"Preview  :\n{self.df.head(2)}\n"
            )
    
    def get(self):
        """
        Summary: 최종적으로 가공된 데이터프레임 반환
        """
        return self.df
 
    def drop_columns_by_nan_ratio(
            self,
            threshold: float = 0.3,
            columns: Optional[list] = None
            )-> 'RawDataCleaner':
        """
        Summary: 결측치 비율이 threshold 이상인 컬럼들을 자동 삭제

        Args:
            threshold (float): 컬럼 제거 기준 비율 (0 ~ 1 사이)
            columns (list, Optional): 특정 컬럼만 대상으로 지정할 수 있음 (기본값 : None -> 전체 컬럼)

        Returns:
            self: 체이닝 가능한 객체 반환
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f'[{self.df_name}] : threshold는 0 이상 1 이하의 값이어야 합니다.')
        
        target_cols = columns if columns is not None else self.df.columns
        nan_ratio = self.df[target_cols].isnull().mean()
        drop_cols = nan_ratio[nan_ratio >= threshold].index.to_list()

        if drop_cols:
            logger.info(f'[{self.df_name}] 결측 비율 : {threshold * 100:.0f}% 이상인 컬럼을 삭제했습니다.\n'
                        f'[{self.df_name}] 삭제 컬럼 : {drop_cols}')
            self.df.drop(columns=drop_cols, inplace=True)
        else:
            logger.info(f'[{self.df_name}] 결측 비율 : {threshold * 100:.0f}% 이상인 컬럼이 없습니다.')

        return self
    
    def drop_group_row(
            self, idx: int,
            reset_idx: bool = True,
            condition_column: str = None
            ) -> 'RawDataCleaner':
        """
        Summary:
            그룹 기준으로 특정 위치에 있는 행을 삭제하는 함수

        Args:
            idx (int): 삭제할 행의 위치. -1은 마지막, 0은 첫 번째 행을 의미
            reset_idx (bool): 삭제 후 인덱스를 초기화할지 여부 (기본값: True)
            condition_column (str, optional): 그룹 기준 열 이름. 지정되지 않으면 전체 데이터 기준으로 삭제됨

        Returns:
            self: 체이닝 가능한 객체 반환
        """ 
        if idx is not None and not isinstance(idx, int):
            raise TypeError(f'[{self.df_name}] idx는 int 타입만 허용됩니다.')

        if condition_column is not None:
            grouped = self.df.groupby(condition_column)
            try:
                drop_idx = grouped.nth(idx).index
                self.df = self.df.drop(index=drop_idx)
            except Exception as e:
                raise ValueError(f'[{self.df_name}] groupby 삭제 처리 중 오류 발생 : {e}')
        
        else:
            if idx == -1:
                self.df = self.df.iloc[:-1]
            else:
                self.df = self.df.drop(index=idx)
        
        if reset_idx:
            self.df = self.df.reset_index(drop=True)
        
        return self
        
    def split_into_columns(
            self,
            column: str,
            separator: str = '–',
            new_columns: list = None,
            drop_original : bool = True
            ) -> "RawDataCleaner":
        """
        Summary: 하나의 문자열 열을 구분자를 기준으로 분리히여 새로운 열들로 확장합니다.

        Args:
            column (str): 분리할 열 이름
            separator (str): 구분자 (기본값: '–')
            new_columns (list): 새로 생성될 열 이름 리스트
            drop_original (bool): 원본 열 삭제 여부
        
        Returns:
            self : 체이닝 가능한 self 객체 반환
        """
        split = self.df[column].astype(str).str.strip()
        split_columns = split.str.split(separator, expand=True)

        if new_columns is not None:
            if len(new_columns) != split_columns.shape[1]:
                raise ValueError(
                    f'[{self.df_name}] new_columns의 길이 {len(new_columns)}가 분리된 열 개수 {split_columns.shape[1]}와 일치해야합니다.'
                )
            split_columns.columns = new_columns
        
        self.df = pd.concat([self.df, split_columns], axis=1)

        if drop_original:
            self.df.drop(columns=[column], inplace=True)

        return self
        
    def apply_lower_all_str(
            self,
            include_columns: bool = False
            ) -> 'RawDataCleaner':
        """
        Summary: 문자열 타입 셀 전체에 대해 strip + lower 처리하며,
                 선택적으로 컬럼 이름도 소문자로 처리할 수 있습니다.
        Args:
            include_columns (bool): 컬럼명도 lower 적용할지 여부 (기본값: False)
        
        Returns:
            self: 체이닝 가능한 self 객체 반환
        """
        self.df = self.df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)

        if include_columns:
            self.df.columns = [col.strip().lower() for col in self.df.columns]
        
        return self