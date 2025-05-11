import pandas as pd

df = pd.read_csv('/Users/sonjeehyung/Documents/Project/football-score-prediction/data/cleaned_data.csv')

print(df['season'].value_counts().sort_index())