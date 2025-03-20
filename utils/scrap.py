import os
import random
import time
import pandas as pd
import requests
from io import StringIO


def fbref_scraper(start_year, end_year, stats=False, recent=False):
    """
    Summary: FBRef에서 EPL 시즌별 경기 데이터 및 팀 스탯을 스크랩하는 함수

    Args:
        start_year (int): 시작 시즌 연도 (예: 2015)
        end_year (int): 종료 시즌 연도 (예: 2023)
        stats (bool): 경기 데이터가 아닌 팀 스탯 데이터 스크랩
        recent (bool): 최근 시즌 데이터를 포함할지 여부 (기본값: False)
    
    Returns:
        pd.DataFrame: 스크랩한 EPL 경기 결과 및 일정 데이터프레임
    """
    all_data = []
    
    if stats:
        idx = 2
    else:
        idx = 0
    
    for n in range(start_year, end_year + 1):
        season = f"{n}-{n + 1}"
        url = generate_fbref_url(season, stats, recent)
        
        try:
            df = pd.read_html(url)
            df = pd.DataFrame(df[idx])
            
            df.dropna(how='all', inplace=True)
            df['Season'] = n
            
            all_data.append(df)
            print(f"fbref : {season} 시즌 데이터 로드 완료")
            
            temp = random.randint(5, 10)
            time.sleep(temp)

        except Exception as e:
            print(f"fbref : {season} 시즌 데이터 오류 발생: {e}")
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df
    else:
        print("데이터가 없습니다.")
        return None


def transfermarkt_scraper(start_year, end_year):
    """
    """

    all_data = []
    idx = 1

    for n in range(start_year, end_year + 1):
        season = n
        url  = generate_transfermarkt_url(season)

        try:
            df = pd.read_html(url)
            df = pd.DataFrame(df[idx])
            df['Season'] = n
            all_data.append(df)
            print(f"transfermarkt : {season} 시즌 데이터 로드 완료")

            temp = random.randint(5, 10)
            time.sleep(temp)
        except Exception as e:
            print(f"transfermarkt : {season} 시즌 데이터 오류 발생: {e}")
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df
    else:
        print("데이터가 없습니다.")
        return None



def generate_fbref_url(season, stats=False, recent=False):
    """
    Summary: 시즌별 경기 결과 및 일정 URL 생성 함수
    
    Args:
        season(str): 시즌 문자열 (예: "2023-2024")
        stats (bool): transfermarkt 데이터 (기본값: False)
        recent(bool): 최근 시즌 데이터 여부 (기본값: False)
    
    Returns:
        str: 생성된 URL 문자열
    """
    if recent:
        if stats:
            return "https://fbref.com/en/comps/9/Premier-League-Stats"
        else:
            return "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
        
    if stats:
        return f"https://fbref.com/en/comps/9/{season}/{season}-Premier-League-Stats"
    else:
        return f"https://fbref.com/en/comps/9/{season}/schedule/{season}-Premier-League-Scores-and-Fixtures"


def generate_transfermarkt_url(season):
    """
    Summary: 시즌별 경기 결과 및 일정 URL 생성 함수
    
    Args:
        season(str): 시즌 문자열 (예: 2023)
        stats (bool): transfermarkt 데이터 (기본값: False)
        recent(bool): 최근 시즌 데이터 여부 (기본값: False)
    
    Returns:
        str: 생성된 URL 문자열
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Referer": "https://www.transfermarkt.com/",
    }

    url = f'https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1/plus/?saison_id={season}'
    
    response = requests.get(url, headers=headers)
    html = StringIO(response.text)
    
    return html

    
def save_to_csv(df, file_name, save_dir='./data'):
    """
    Summary: 데이터프레임을 csv파일로 저장하는 함수

    Args:
        df (pd.DataFrame): 저장할 데이터프레임
        file_name (str): 저장할 파일 이름
        save_dir (str): 저장할 디렉토리 경로 (기본값: './data')
    
    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file_name)
    df.to_csv(file_path, index=False)
    print(f"해당 경로에 {file_path} 저장 완료")


if __name__ == "__main__":
    # 2004-2005 시즌부터 2023-2024 시즌
    start_year = 2004
    end_year = 2023
    
    save_fbref_dir = "./data/fbref/original"
    save_transfermarkt_dir = "./data/transfermarkt/original"
    
    fixture_file = "fbref_fixture.csv"
    fixture_df = fbref_scraper(start_year, end_year, stats=False, recent=False)

    if fixture_df is not None:
        save_to_csv(fixture_df, fixture_file, save_fbref_dir)
    
    # stats_file = "transfermarkt_stats.csv"
    # stats_df = transfermarkt_scraper(start_year, end_year)

    # if stats_df is not None:
    #     save_to_csv(stats_df, stats_file, save_transfermarkt_dir)

    
