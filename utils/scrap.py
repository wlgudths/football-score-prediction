import os
import time
import random
import pandas as pd
import requests
from io import StringIO
from utils.logger import logger


def fbref_scraper(start_year: int, end_year: int, stats: bool = False, recent: bool = False) -> pd.DataFrame:
    """
    Summary: FBRef에서 EPL 시즌별 경기 데이터 및 팀 스탯을 스크랩하는 함수

    Args:
        start_year (int): 시작 시즌 연도 (예: 2015)
        end_year (int): 종료 시즌 연도 (예: 2023)
        stats (bool): 경기 데이터가 아닌 팀 스탯 데이터 스크랩
        recent (bool): 최근 시즌 데이터를 포함할지 여부 (기본값: False)
    
    Returns:
        pd.DataFrame: 스크랩한 데이터프레임
    """
    all_data = []    
    idx = 2 if stats else 0

    for n in range(start_year, end_year + 1):
        season = f"{n}-{n + 1}"
        url = generate_fbref_url(season, stats, recent)
        
        try:
            df = pd.read_html(url)
            df = pd.DataFrame(df[idx])
            df.dropna(how='all', inplace=True)
            df['season'] = n
            all_data.append(df)
            
            logger.info(f'fbref: {season} 시즌 데이터 로드 완료')
            time.sleep(random.randint(5, 10))

        except Exception as e:
            logger.error(f'fbref: {season} 시즌 데이터 오류 발생 : {e}')

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        logger.warning('fbref: 데이터를 불러오지 못했습니다.')
        return None


def transfermarkt_scraper(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Summary: Transfermarkt에서 시즌별 팀 스쿼드 데이터를 스크랩하는 함수

    Args:
        start_year (int): 시작 시즌 연도 (예: 2015)
        end_year (int): 종료 시즌 연도 (예: 2023)
    
    Returns:
        pd.DataFrame: 스크랩한 데이터프레임
    """

    all_data = []
    idx = 1

    for n in range(start_year, end_year + 1):
        season = f"{n}-{n + 1}"
        url  = generate_transfermarkt_url(season)

        try:
            df = pd.read_html(url)
            df = pd.DataFrame(df[idx])
            df['season'] = n
            all_data.append(df)
            
            logger.info(f'transfermarkt: {season} 시즌 데이터 로드 완료')
            time.sleep(random.randint(5, 10))
        
        except Exception as e:
            logger.error(f"transfermarkt : {season} 시즌 데이터 오류 발생 : {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        logger.warning('transfermarkt: 데이터를 불러오지 못했습니다.')
        return None


def generate_fbref_url(season: str, stats: bool = False, recent: bool = False) -> str:
    """
    Summary: Fbref 매치 데이터 또는 팀 스탯 URL 생성 함수
    
    Args:
        season(str): 시즌 문자열 (예: '2023-2024')
        stats (bool): 스탯 URL 여부
        recent(bool): 최신 시즌 URL 여부
    
    Returns:
        str: 생성된 URL
    """
    if recent:
        if stats:
            return 'https://fbref.com/en/comps/9/Premier-League-Stats'
        else:
            return 'https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures'
        
    if stats:
        return f'https://fbref.com/en/comps/9/{season}/{season}-Premier-League-Stats'
    else:
        return f'https://fbref.com/en/comps/9/{season}/schedule/{season}-Premier-League-Scores-and-Fixtures'


def generate_transfermarkt_url(season: int) -> StringIO:
    """
    Summary: Transfermarkt 시즌별 스쿼드 URL 생성
    
    Args:
        season (int): 시즌 연도 (예: 2023)
    
    Returns:
        StringIO: HTML 문자열을 담은 StringIO 객체
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Referer": "https://www.transfermarkt.com/",
    }

    url = f'https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1/plus/?saison_id={season}'    
    response = requests.get(url, headers=headers)

    return StringIO(response.text)