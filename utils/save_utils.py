import os
from typing import Dict
import pandas as pd
from utils.config_utils import save_config
from utils.logger import logger

def save_to_csv(df: pd.DataFrame, config: Dict, file_name: str, save_key: str) -> None:
    """
    Summary: 데이터프레임을 csv파일로 저장하는 함수

    Args:
        df (pd.DataFrame): 저장할 데이터프레임
        config (Dict): yaml 로드된 설정 딕셔너리
        file_name (str): 저장할 파일 이름 (예 : 'output.csv')
        save_key (str): config['save'] 내부에서 업데이트할 key 이름

    Returns:
        None
    """
    try:
        save_dir = config['save'].get('data_path', './data')
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, file_name)
        df.to_csv(file_path, index=False)

        config_path = config['save'].get('config_path')
        
        config['save'][save_key] = file_path
        save_config(config=config, config_path=config_path)
            
        logger.info(f'CSV 파일 저장 완료: {file_path}')

    except Exception as e:
        logger.error(f'CSV 저장 실패 : {file_name} -> {e}')