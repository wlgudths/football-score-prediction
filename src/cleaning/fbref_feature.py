import pandas as pd
from typing import Dict, List, Tuple


def update_season_stats(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Summary:
        각 시즌별로 팀 통계를 누적 계산하고,
        랭킹 및 각 경기 시점 기준의 팀 통계를 추가한 새로운 데이터프레임 생성
    
    Args:
        df (pd.DataFrame): 경기 데이터 프레임
        column (str): 시즌 구분에 사용할 컬럼명
    
    Returns:
        pd.DataFrame: 누적 통계 및 순위가 포함된 확장된 경기 데이터프레임
    """
    all_data = []
    seasons = df[column].unique()

    for season in seasons:
        season_df = df.loc[df[column] == season].copy()
        stats = _init_stats(season_df, season)

        for _, row in season_df.iterrows():
            home = row["home"]
            away = row["away"]
            home_score = int(row['home_score'])
            away_score = int(row['away_score'])

            home_points, away_points = _calculate_points(home_score, away_score)

            # 홈 팀 통계 업데이트
            stats[home]["points"] += home_points
            stats[home]["goals_for"] += home_score
            stats[home]["goals_against"] += away_score
            stats[home]["goal_diff"] += (home_score - away_score)
            
            # 원정 팀 통계 업데이트
            stats[away]["points"] += away_points
            stats[away]["goals_for"] += away_score
            stats[away]["goals_against"] += home_score
            stats[away]["goal_diff"] += (away_score - home_score)
            
            # 시점별 누적 통계 추가
            season_df.loc[row.name, 'home_points'] = stats[home]['points']
            season_df.loc[row.name, 'home_goals'] = stats[home]['goals_for']
            season_df.loc[row.name, 'home_goals_against'] = stats[home]['goals_against']
            season_df.loc[row.name, 'home_goal_diff'] = stats[home]['goal_diff']
            
            season_df.loc[row.name, 'away_points'] = stats[away]['points']
            season_df.loc[row.name, 'away_goals'] = stats[away]['goals_for']
            season_df.loc[row.name, 'away_goals_against'] = stats[away]['goals_against']
            season_df.loc[row.name, 'away_goal_diff'] = stats[away]['goal_diff']

        # 순위 계산 및 추가
        rank_df = _calculate_rank(stats)
        season_df['home_rank'] = season_df['home'].map(rank_df.set_index('index')['rank'])
        season_df['away_rank'] = season_df['away'].map(rank_df.set_index('index')['rank'])

        all_data.append(season_df)

    updated_df = pd.concat(all_data, ignore_index=True)

    return updated_df

def _init_stats(df: pd.DataFrame, season: int) -> Dict:
    """
    Summary: 주어진 시즌에 참가한 모든 팀의 초기 통계 딕셔너리 생성

    Args:
        df (pd.DataFrame): 경기 데이터프레임
        season (int): 해당 시즌
    
    Returns:
        dict: 팀별 초기화된 통계 딕셔너리
    """
    teams = _get_season_teams(df, season)
    stats = {
        team : {
            'points' : 0,
            'goals_for' : 0,
            'goals_against' : 0,
            'goal_diff' : 0
            } for team in teams
        }
    
    return stats

def _get_season_teams(df: pd.DataFrame, season: int) -> list[str]:
    """
    Summary: 주어진 시즌에 홈/원정으로 모두 등장한 팀 목록 추출

    Args:
        df (pd.DataFrame): 경기 데이터프레임
        season (int): 해당 시즌
    
    Returns:
        List[str]: 해당 시즌에 참가한 팀 이름 리스트
    """
    home = df.loc[df['season'] == season, 'home'].unique()
    away = df.loc[df['season'] == season, 'away'].unique()
    teams = list(set(home) & set(away))
    
    return teams

def _calculate_points(home_score: int, away_score: int) -> Tuple[int, int]:
    """
    Summary: 홈/원정 점수를 기반으로 승/무/패 결과에 따른 승점 반환

    Args:
        home_score (int): 홈 팀 득점
        away_score (int): 원정 팀 득점
    
    Returns:
        Tuple[int, int]: (홈 팀 승점, 원정 팀 승점)
    """
    if home_score > away_score:
        return 3, 0
    elif home_score < away_score:
        return 0, 3
    else:
        return 1, 1

def _calculate_rank(stats: Dict) -> pd.DataFrame:
    """
    Summary: 팀별 누적 통계로 순위를 계산합니다.

    Args:
        stats (dict): 팀별 누적 통계 딕셔너리
    
    Returns:
        pd.DataFrame: 순위가 포함된 정렬된 팀 통계 데이터프레임
    """
    team_stats = pd.DataFrame.from_dict(stats, orient='index').reset_index()

    team_stats = team_stats.sort_values(
        by=['points', 'goal_diff', 'goals_for'],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    team_stats['rank'] = team_stats.index + 1

    return team_stats