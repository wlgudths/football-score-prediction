import pandas as pd
from preprocessor import Preprocessor

tr_df = pd.read_csv('/Users/sonjeehyung/Documents/Project/football-score-prediction/data/transfermarkt/original/transfermarkt_stats.csv')
fbref_df = pd.read_csv('/Users/sonjeehyung/Documents/Project/football-score-prediction/data/fbref/original/fbref_fixture.csv')

print(tr_df.head())
print('------------------')
print(fbref_df.head())

tr_preprocessor = Preprocessor(tr_df)
tr_df = tr_preprocessor.drop_columns(['Club', 'Total market value'])

print(tr_df.head())
print(tr_df.isna().sum())

tr_df = tr_preprocessor.drop_rows(idx=-1, reset_idx=True, condition_column='Season')

tr_df.head()
print(tr_df.isna().sum())

rename_list = ['team', 'squad', 'age', 'Foreigners', 'market_value', 'total_market_value', 'season']

rename_map = {
    col : rename_col for col, rename_col in zip(tr_df.columns, rename_list)
}

tr_df = tr_preprocessor.rename_columns(rename_map=rename_map)

print(tr_df.head())