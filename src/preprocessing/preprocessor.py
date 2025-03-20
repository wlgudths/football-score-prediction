import pandas as pd

class Preprocessor:
    def __init__(self, df):
        self.df = df.copy()
    
    def drop_columns(self, columns):
        """
        특정 컬럼 삭제
        """
        self.df.drop(columns=columns, axis=1, inplace=True)
        
        return self.df
    
    def drop_rows(self, idx, reset_idx=True, condition_column=None):
        """
        특정 행 삭제
        """
        if idx is not None and not isinstance(idx, (int, list)):
            raise TypeError("idx는 int 또는 list 타입만 허용됩니다.")
        
        if condition_column is not None:
            if idx == -1:
                drop_idx = self.df.groupby(condition_column).nth(idx).index
                self.df = self.df.drop(index=drop_idx)
            else:
                drop_idx = self.df.groupby(condition_column).nth(idx).as_index
                self.df = self.df.drop(index=drop_idx)
        
        elif idx is not None:
            if idx == -1:
                self.df = self.df.iloc[:-1]
            else:
                self.df = self.df.drop(index=idx)
        
        if reset_idx:
            self.df = self.df.reset_index(drop=True)
        
        return self.df
        

    def split_column(self, column, separator='–', new_columns=None, expand=True, drop_original=True):
        """
        Summary: 특정 열을 구분자를 기준으로 나누어 새로운 열 생성

        Args:
            column (str): 분리할 열 이름
            separator (str): 구분자 (기본값: '–')
            new_columns (list): 새로 생성될 열 이름 리스트
            expand (bool): True -> 새로운 열로 나눔 / False -> 리스트로 반환
            drop_original (bool): 원본 열 삭제 여부
        
        Returns:
            self.df: 변환된 데이터프레임
        """
        split = self.df[column].copy()
        
        if '–' in split.values:
            split = split.str.replace('–', separator, regex=True).str.strip()
        
        split_columns = split.str.split(separator, expand=expand)

        if new_columns is not None:
            if len(new_columns) != split_columns.shape[1]:
                raise ValueError(f"new_columns의 길이 {len(new_columns)}가 분리된 열 개수 {split_columns.shape[1]}와 일치해야 합니다.")
            
            split_columns.columns = new_columns
            
        self.df = pd.concat([self.df, split_columns], axis=1)

        if drop_original:
            self.drop_columns([column])
        
        return self.df

    def map_and_replace(self, df, key_column, value_column, target_column):
        """
        """
        keys = sorted(set(df[key_column].str.lower()))
        values = sorted(set(self.df[value_column].str.lower()))
        
        if len(keys) != len(values):
            raise ValueError(f"keys의 길이 {len(keys)}가 values의 길이 {len(values)}와 일치해야 합니다.")
        else:
            mapping_dict = dict(zip(keys, values))
        
        df[target_column] = df[target_column].str.lower().replace(mapping_dict)

        return self.df
        
    def extract_date_features(self, column, year=False, month=False, day=False, day_of_week=False):
        """
        날짜
        """
        self.df[column] = pd.to_datetime(self.df[column], errors="coerce")

        if year:
            self.df['year'] = self.df[column].dt.year
        
        if month:
            self.df['month'] = self.df[column].dt.month
        
        if month:
            self.df['day'] = self.df[column].dt.month
        
        if month:
            self.df['dayofweek'] = self.df[column].dt.dayofweek
        
        return self.df

    def convert_str_case(self, columns, case="lower"):
        """
        특정 열에 대해 문자열 case 변환 적용
        """
        if case == "lower":
            self.df[columns] = self.df[columns].apply(lambda x: x.str.strip().str.lower())
        elif case == "upper":
            self.df[columns] = self.df[columns].apply(lambda x: x.str.strip().str.upper())
        else:
            raise ValueError("case는 'lower' 또는 'upper'만 허용됩니다.")
        
        return self.df

    
    def rename_columns(self, rename_map, drop_missing=False):
        """
        Summary: 컬럼명 변경

        Args:
            rename_map (dict): 변경할 컬럼 이름 매핑 딕셔너리
            drop_missing (bool): 매핑되지 않은 기존 열 삭제 여부 (기본값: False)
        
        Returns:
            self.df: 변환된 데이터프레임
        """
        self.df.rename(columns=rename_map, inplace=True, errors='ignore')

        if drop_missing:
            valid_columns = list(rename_map.values())
            self.df = self.df[valid_columns]
        
        return self.df