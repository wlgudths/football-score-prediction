import argparse
from utils.config_utils import load_config, update_config
from utils.scrap import fbref_scraper, transfermarkt_scraper
from utils.save_utils import save_to_csv
from src.cleaning.generate_cleaned_data import generate_cleaned_data


def main():
    parser = argparse.ArgumentParser(description="Football Score Prediction")
    parser.add_argument('--config_path', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    update_config(config_path=args.config_path,
                  key='save',
                  value={**cfg['save'], 'config_path' : args.config_path},
                  save=True)
    
    if cfg['scrap'].get('enable') == True:
        fbref_cfg = cfg['scrap'].get('fbref')
        tr_cfg = cfg['scrap'].get('transfermarkt')
        
        fbref_raw_data = fbref_scraper(start_year=fbref_cfg['start_year'],
                                       end_year=fbref_cfg['end_year'],
                                       stats=fbref_cfg['stats'],
                                       recent=fbref_cfg['recent'])

        save_to_csv(df=fbref_raw_data, config=cfg, file_name=fbref_cfg['file_name'], save_key=fbref_cfg['save_key'])

        tr_raw_data = transfermarkt_scraper(start_year=tr_cfg['start_year'],
                                            end_year=tr_cfg['end_year'])
        
        save_to_csv(df=tr_raw_data, config=cfg, file_name=tr_cfg['file_name'], save_key=tr_cfg['save_key'])

    generate_cleaned_data(cfg)


if __name__ == '__main__':
    main()
