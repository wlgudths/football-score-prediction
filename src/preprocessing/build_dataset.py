import pandas as pd
from preprocessor import Preprocessor
from fbref_feature import update_season_stats


fbref_file_path = '/Users/sonjeehyung/Documents/Project/football-score-prediction/data/fbref/original/fbref_fixture.csv'
tranfermarkt_file_path = '/Users/sonjeehyung/Documents/Project/football-score-prediction/data/transfermarkt/original/transfermarkt_stats.csv'

fbref_df = pd.read_csv(fbref_file_path)
tr_df = pd.read_csv(tranfermarkt_file_path)

fbref_preprocessor = Preprocessor(fbref_df)
tr_df = Preprocessor(tr_df)

fberf_drop_columns = ['Wk', 'Day', 'Time', 'Venue', 'Referee', 'Match Report', 'Notes', 'xG', 'xG.1']

fbref_df = fbref_preprocessor.drop_columns(fberf_drop_columns)

fbref_df = fbref_preprocessor.split_column(column='Score', new_columns=['home_score', 'away_score'])

fbref_df = fbref_preprocessor.convert_str_case(columns=["Home","Away"], case="lower")

fbref_df = fbref_preprocessor.extract_date_features(column="Date", year=True, month=True, day=True, day_of_week=True)
fbref_df = fbref_preprocessor.drop_columns("Date")

fbref_df.columns = fbref_df.columns.str.lower()

fbref_df = update_season_stats(fbref_df, 'season')
