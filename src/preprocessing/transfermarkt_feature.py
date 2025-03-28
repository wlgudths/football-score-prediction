import pandas as pd
from preprocessor import Preprocessor

tr_file_path = '/Users/sonjeehyung/Documents/Project/football-score-prediction/data/transfermarkt/original/transfermarkt_stats.csv'
fbref_file_path = '/Users/sonjeehyung/Documents/Project/football-score-prediction/data/fbref/original/fbref_fixture.csv'


tr_df = pd.read_csv(tr_file_path)
fbref_df = pd.read_csv(fbref_file_path)

print(tr_df.head())

def convert_money(value):
    if isinstance(value, str):
        

