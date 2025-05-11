import os
from ruamel.yaml import YAML
from typing import Any, Dict
from utils.logger import logger


def load_config(save_path: str) -> Dict:
    """
    Summary: yaml 설정파일을 로드하는 함수

    Args:
        save_path (str): 설정 파일(.yaml)의 경로
    
    Returns:
        Dict: 로드된 설정 딕셔너리
    """
    yaml = get_yaml_parser()

    try:
        with open(save_path, 'r') as f:
            config = yaml.load(f)
        
        logger.info(f'config 로드 완료')
    
    except Exception as e:
        logger.error(f'config 로드 실패 : {e}')
    
    return config


def update_config(config: dict, key: str, value: Any, save_path: str=None) -> None:
    """
    """
    try:
        config[key] = value
        logger.info(f"config 업데이트 (in-memory): {key} = {value}")
    
        if save_path:
            save_config(config, save_path)
    
    except Exception as e:
        logger.error(f'config 업데이트 실패 : {e}')
    
    return config
    

def save_config(config: dict, save_path: str) -> None:
    """
    """
    yaml = get_yaml_parser()
    
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        logger.info(f'config 파일을 성공적으로 저장했습니다. 경로: {save_path}')
    
    except Exception as e:
        logger.error(f'config 저장 중 오류가 발생했습니다. 오류 내용: {e}')



def get_yaml_parser() -> YAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml