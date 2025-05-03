import pandas as pd

def encode_team_by_ordinal(df: pd.DataFrame, decoders: dict):
    """
    """
    home_ranks = df[df['wk'] == 38][['season', 'home', 'home_rank']].rename(columns={'home': 'team', 'home_rank': 'rank'})
    away_ranks = df[df['wk'] == 38][['season', 'away', 'away_rank']].rename(columns={'away': 'team', 'away_rank': 'rank'})

    all_ranks = pd.concat([home_ranks, away_ranks])
    avg_rank_dict = all_ranks.groupby('team')['rank'].mean().to_dict()
    sorted_teams = sorted(avg_rank_dict.items(), key=lambda x: x[1], reverse=True)
    
    ordinal_index_dict = {team: idx + 1 for idx, (team, _) in enumerate(sorted_teams)}
    max_rank = all_ranks['rank'].max()

    for k in avg_rank_dict:
        avg_rank_dict[k] = max_rank + 1 - avg_rank_dict[k]

    df['home'] = df['home'].map(ordinal_index_dict)
    df['away'] = df['away'].map(ordinal_index_dict)

    decoders['home'] = {'encoder': {v: k for k, v in ordinal_index_dict.items()}}
    decoders['away'] = {'encoder': {v: k for k, v in ordinal_index_dict.items()}}

    return df, decoders