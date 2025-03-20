import pandas as pd

def calculate_points(home_score, away_score):
    if home_score > away_score:
        return 3, 0
    elif home_score < away_score:
        return 0, 3
    else:
        return 1, 1

def calculate_rank(stats):
    """
    """
    team_stats = pd.DataFrame.from_dict(stats, orient='index').reset_index()

    team_stats = team_stats.sort_values(
        by=['points', 'goal_diff', 'goals_for'],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    team_stats['rank'] = team_stats.index + 1

    return team_stats

def get_season_teams(df, season):
    """
    해당 시즌에 참가했던 팀 목록 반환
    """
    home = df.loc[df['season'] == season, 'home'].unique()
    away = df.loc[df['season'] == season, 'away'].unique()
    teams = list(set(home) & set(away))
    
    return teams

def init_stats(season):
    teams = get_season_teams(season)
    stats = {team : {'points' : 0, 'goals_for' : 0, 'goals_against' : 0, 'goal_diff' : 0} for team in teams}
    
    return stats

def update_season_stats(df, column):
    all_data = []
    seasons = df[column].unique()

    for season in seasons:
        season_df = df.loc[df[column] == season].copy()
        stats = init_stats(season_df, season)

        for _, row in season_df.iterrows():
            home = row["home"]
            away = row["away"]
            home_score = row['home_score']
            away_score = row['away_score']

            home_points, away_points = calculate_points(home_score, away_score)

            stats[home]["points"] += home_points
            stats[home]["goals_for"] += home_score
            stats[home]["goals_against"] += away_score
            stats[home]["goal_diff"] += (home_score - away_score)

            stats[away]["points"] += away_points
            stats[away]["goals_for"] += away_score
            stats[away]["goals_against"] += home_score
            stats[away]["goal_diff"] += (away_score - home_score)

            season_df.loc[row.name, 'home_points'] = stats[home]['points']
            season_df.loc[row.name, 'home_goals'] = stats[home]['goals_for']
            season_df.loc[row.name, 'home_goals_against'] = stats[home]['goals_against']
            season_df.loc[row.name, 'home_goal_diff'] = stats[home]['goal_diff']
            
            season_df.loc[row.name, 'away_points'] = stats[away]['points']
            season_df.loc[row.name, 'away_goals'] = stats[away]['goals_for']
            season_df.loc[row.name, 'away_goals_against'] = stats[away]['goals_against']
            season_df.loc[row.name, 'away_goal_diff'] = stats[away]['goal_diff']

        rank_df = calculate_rank(stats)
        season_df['home_rank'] = season_df['home'].map(rank_df.set_index('index')['rank'])
        season_df['away_rank'] = season_df['away'].map(rank_df.set_index('index')['rank'])

        all_data.append(season_df)

    updated_df = pd.concat(all_data, ignore_index=True)

    return updated_df
