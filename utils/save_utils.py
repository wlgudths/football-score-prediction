import os
import pandas as pd
from logger import logger


def save_to_csv(df: pd.DataFrame, file_name: str, save_dir: str ='./data') -> None:
    """
    Summary: 데이터프레임을 csv파일로 저장하는 함수

    Args:
        df (pd.DataFrame): 저장할 데이터프레임
        file_name (str): 저장할 파일 이름 (예 : 'output.csv')
        save_dir (str): 저장할 디렉토리 경로 (기본값: './data')
    
    Returns:
        None
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name)
        df.to_csv(file_path, index=False)
        logger.info(f'CSV 파일 저장 완료: {file_path}')
    
    except Exception as e:
        logger.error(f'CSV 저장 실패 : {file_name} -> {e}')