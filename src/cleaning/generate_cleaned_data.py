from typing import Dict
import pandas as pd
from src.cleaning.raw_cleaner import RawDataCleaner
from src.cleaning.fbref_feature import update_season_stats
from src.cleaning.transfermarkt_feature import clean_market_value, fuzzy_match_teams
from src.cleaning.merge_data import merge_df
from utils.logger import logger
from utils.save_utils import save_to_csv


def generate_cleaned_data(cfg: Dict) -> None:
    """
    Summary: 
        Fbref, Transfermarkt 원천 데이터를 정제하고 병합하여 학습용 데이터셋을 생성하는 함수.
        
    Args:
        cfg (Dict): config.yaml 설정파일
    
    Return: None
    """
    logger.info('[generate_cleaned_data] 데이터 정제 파이프라인 시작')
    
    fbref_cfg = cfg['cleaning'].get('fbref')
    tr_cfg = cfg['cleaning'].get('transfermarkt')
    merge_cfg = cfg['cleaning'].get('merge')
    
    fbref_raw_data = cfg['save'].get('fbref_raw_data')
    transfermarkt_raw_data = cfg['save'].get('transfermarkt_raw_data')
    
    fbref_df = pd.read_csv(fbref_raw_data)
    tr_df = pd.read_csv(transfermarkt_raw_data)

    fbref_cleaner = RawDataCleaner(fbref_df, 'Fbref')
    
    tr_df.drop(columns=tr_cfg['drop_columns'], inplace=True)
    tr_df.columns = tr_cfg['rename_columns']

    tr_cleaner = RawDataCleaner(tr_df, 'Transfermarkt')

    fbref_df = (
        fbref_cleaner
        .apply_lower_all_str(include_columns=True)
        .drop_columns_by_nan_ratio(threshold=fbref_cfg['nan_ratio'])
        .split_into_columns(column=fbref_cfg['split_column'],
                            separator=fbref_cfg['split_seperator'],
                            new_columns=fbref_cfg['split_new_columns'])
        .get()
        )
    
    fbref_df = update_season_stats(fbref_df, column=fbref_cfg['feature_column'])

    tr_df = (
        tr_cleaner
        .apply_lower_all_str(include_columns=True)
        .drop_group_row(idx=-1, condition_column=tr_cfg['condition_column'])
        .get()
    )

    tr_df = clean_market_value(tr_df, columns=tr_cfg['covert_columns'])
    
    fbref_teams = set(fbref_df[fbref_cfg['target_columns'][0]].unique()) | set(fbref_df[fbref_cfg['target_columns'][1]].unique())
    tr_teams = set(tr_df[tr_cfg['source_columns']])
    team_mapping = fuzzy_match_teams(source=tr_teams, target=fbref_teams, start_threshold=80, step=20)

    tr_df[tr_cfg['source_columns']] = tr_df[tr_cfg['source_columns']].replace(team_mapping)

    home_merged_df = merge_df(
        left_df=fbref_df,
        right_df=tr_df,
        how=merge_cfg['how'],
        left_on=merge_cfg['home_left_on'],
        right_on=merge_cfg['right_on'],
        rename_cols=merge_cfg['home_rename_cols'],
        drop_col=merge_cfg['drop_column']
    )

    away_merged_df = merge_df(
        left_df=home_merged_df,
        right_df=tr_df,
        how=merge_cfg['how'],
        left_on=merge_cfg['away_left_on'],
        right_on=merge_cfg['right_on'],
        rename_cols=merge_cfg['away_rename_cols'],
        drop_col=merge_cfg['drop_column']
    )

    save_to_csv(df=away_merged_df, config=cfg, file_name=merge_cfg['file_name'], save_key=merge_cfg['save_key'])

    logger.info('[generate_cleaned_data] 데이터 정제 파이프라인 종료')