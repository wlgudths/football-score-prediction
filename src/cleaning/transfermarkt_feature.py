import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from utils.logger import logger

def clean_market_value(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Summary: 여러 시장 가치 컬럼들을 float 유로 단위로 정제합니다.

    Args:
        df (pd.DataFrame) : 입력 데이터프레임
        columns (list[str]): 변환할 컬럼명 리스트
    
    Returns:
        pd.DataFrame: 정제된 데이터프레임
    """
    for col in columns:
        df[col] = df[col].apply(_convert_market_value)
    
    return df

def _convert_market_value(value: str) -> float:
    """
    Summary:
        시장 가치 문자열을 float(유로 단위)로 변환하는 내부 함수입니다.
        단위가 붙은 문자열(예: '€10.5m', '€800k', '€1.2bn')을 숫자로 변환합니다.

    Args:
        value (str): 변환할 시장 가치 문자열 (예: '€10.5m', '€800k', '€1.2bn')

    Returns:
        float: 변환된 숫자 (유로 단위). 변환할 수 없는 경우 NaN 반환
    """
    if not isinstance(value, str):
        return np.nan
    
    val = value.lower().replace('€', '').replace(',', '').strip()
    
    if val in {'-', '', 'n/a', 'nan'}:
        return np.nan
    
    try:
        if val.endswith('bn'):
            return float(val[:-2]) * 1_000_000_000
        elif val.endswith('m'):
            return float(val[:-1]) * 1_000_000
        elif val.endswith('k'):
            return float(val[:-1]) * 1_000
        else:
            return float(val)
    
    except ValueError:
        return np.nan

def fuzzy_match_teams(source, target, start_threshold=80, step=20):
    """
    Summary :
        threshold를 낮춰가며 fuzzy match를 시도하고,
        unmatched가 없거나, threshold가 0이되면 종료.
    Args:    
        source: 기준이 되는 팀명 리스트 (예: A DataFrame의 팀명)
        target: 비교 대상이 되는 팀명 리스트 (예: B DataFrame의 팀명)
        threshold: 유사도 기준 (0~100). 높을수록 정확히 일치하는 경우만 매칭됨
        step: 
    
    Returns: dict 형태의 매핑 딕셔너리
    """
    current_threshold = start_threshold
    remaining_sources = source
    available_targets = target
    final_mapping = {}

    while current_threshold >= 0 and remaining_sources:
        new_mapping = {}
        
        for team in remaining_sources:
            match, score, _ = process.extractOne(team, available_targets, scorer=fuzz.ratio)
            if score >= current_threshold:
                new_mapping[team] = match
            
        final_mapping.update(new_mapping)
        remaining_sources -= set(new_mapping.keys())
        available_targets -= set(new_mapping.values())

        logger.info(
            f"[fuzzy_match] Threshold: {current_threshold} | 매핑 완료: {len(new_mapping)} | 남은 항목: {len(remaining_sources)}"
        )

        current_threshold -= step
    
    if not remaining_sources:
        logger.info('[fuzzy_match] 모든 항목이 성공적으로 매핑되었습니다.')
        return final_mapping, None
    
    else:
        logger.warning(f'[fuzzy_match] {sorted(remaining_sources)} 항목이 매핑되지 않았습니다.')
        return final_mapping, remaining_sources