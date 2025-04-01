import pandas as pd
from preprocessor import Preprocessor
from fbref_feature import update_season_stats
from transfermarkt_feature import clean_market_value_column

import sys
import os
# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.match_teams import fuzzy_match_teams
from utils.save_utils import save_to_csv

fbref_file_path = ''
tranfermarkt_file_path = ''

fbref_df = pd.read_csv(fbref_file_path)
tr_df = pd.read_csv(tranfermarkt_file_path)

fbref_preprocessor = Preprocessor(fbref_df)
tr_preprocessor = Preprocessor(tr_df)

fberf_drop_columns = ['Wk', 'Day', 'Time', 'Venue', 'Referee', 'Match Report', 'Notes', 'xG', 'xG.1']

fbref_df = fbref_preprocessor.drop_columns(fberf_drop_columns)

fbref_df = fbref_preprocessor.split_column(column='Score', new_columns=['home_score', 'away_score'])

fbref_df = fbref_preprocessor.convert_str_case(columns=["Home","Away"], case="lower")

fbref_df = fbref_preprocessor.extract_date_features(column="Date", year=True, month=True, day=True, day_of_week=True)
fbref_df = fbref_preprocessor.drop_columns("Date")

fbref_df.columns = fbref_df.columns.str.lower()

fbref_df = update_season_stats(fbref_df, 'season')

tr_drop_columns = ['Club', 'Total market value']

tr_df = tr_preprocessor.drop_columns(tr_drop_columns)
tr_df = tr_preprocessor.drop_rows(-1, reset_idx=True, condition_column='Season')

tr_rename = ['team', 'squad', 'age', 'foreigners', 'market_value', 'total_market_value', 'season']
tr_df.columns = tr_rename

tr_df = tr_preprocessor.convert_str_case(columns=['team'], case='lower')
tr_teams = tr_df['team'].unique()
fbref_teams = fbref_df['home'].unique()

tr_rename_map = fuzzy_match_teams(tr_teams, fbref_teams, 45)
# 수동으로 업데이트
tr_rename_map.update({
    'queens park rangers': 'qpr'
})

tr_df['team'] = tr_df['team'].replace(tr_rename_map)

tr_df = clean_market_value_column(tr_df, 'market_value')
tr_df = clean_market_value_column(tr_df, 'total_market_value')

# 홈팀 기준 머지
fbref_df = fbref_df.merge(
    tr_df,
    how='left',
    left_on=['home', 'season'],
    right_on=['team', 'season']
).rename(columns={
    'squad': 'home_squad',
    'age': 'home_age',
    'foreigners': 'home_foreigners',
    'market_value': 'home_market_value',
    'total_market_value': 'home_total_market_value'
}).drop(columns=['team'])  # team 컬럼은 중복되므로 삭제

# 어웨이팀 기준 머지
fbref_df = fbref_df.merge(
    tr_df,
    how='left',
    left_on=['away', 'season'],
    right_on=['team', 'season']
).rename(columns={
    'squad': 'away_squad',
    'age': 'away_age',
    'foreigners': 'away_foreigners',
    'market_value': 'away_market_value',
    'total_market_value': 'away_total_market_value'
}).drop(columns=['team'])  # team 컬럼은 중복되므로 삭제

save_dir = ''

save_to_csv(fbref_df, 'data.csv', save_dir=save_dir)