import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from utils.config_utils import load_config, update_config
from utils.scrap import fbref_scraper, transfermarkt_scraper
from utils.save_utils import save_to_csv
from src.cleaning.generate_cleaned_data import generate_cleaned_data
from src.preprocessing.generate_preprocessor import generate_preprocessing
from src.model.ml_models import get_model
from src.model.mlpregressor import MLPRegressor
from src.trainer.trainer import dl_train, ml_train
from src.trainer.test import dl_test, ml_test
from utils.loss_function import get_loss_function_ml, get_loss_function_dl
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler
from utils.generate_run_id import generate_run_id


def main():
    parser = argparse.ArgumentParser(description="Football Score Prediction")
    parser.add_argument('--config_path', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    config = load_config(args.config_path)

    run_id = generate_run_id()
    model_name = config['model']['name']
    output_path = os.path.join(config['save']['output_path'], f'{run_id}_{model_name}')
    os.makedirs(output_path, exist_ok=True)
    config_path = os.path.join(output_path, f'{run_id}_config.yaml')
    
    config = update_config(config,
                  key='save',
                  value={
                      **config['save'],
                      'output_path': output_path,
                      'config_path': config_path},
                  save_path=config_path)
    
    if config['scrap'].get('enable') == True:
        fbref_config = config['scrap'].get('fbref')
        tr_config = config['scrap'].get('transfermarkt')
        
        fbref_raw_data = fbref_scraper(start_year=fbref_config['start_year'],
                                       end_year=fbref_config['end_year'],
                                       stats=fbref_config['stats'],
                                       recent=fbref_config['recent'])

        save_to_csv(df=fbref_raw_data, config=config, file_name=fbref_config['file_name'], save_key=fbref_config['save_key'])

        tr_raw_data = transfermarkt_scraper(start_year=tr_config['start_year'],
                                            end_year=tr_config['end_year'])
        
        save_to_csv(df=tr_raw_data, config=config, file_name=tr_config['file_name'], save_key=tr_config['save_key'])

    if config['cleaning'].get('enable') == True:
        generate_cleaned_data(config)

    if config['preprocessing']['mode'] == 'ml':
        X_train, y_train, X_val, y_val, X_test, y_test, test_df, decoders = generate_preprocessing(config)
        model = get_model(config)
        criterion = get_loss_function_ml(config)
        
        if config['train']['test'] == True:
            model = ml_train(X_train, y_train, X_val, y_val, model, criterion, config)
            ml_test(model, X_test, y_test, test_df, config)
        else:
            ml_train(X_train, y_train, X_val, y_val, model, criterion, config)
        
    elif config['preprocessing']['mode'] == 'dl':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        train_dataset, val_dataset, test_dataset, test_df, decoders = generate_preprocessing(config)
        train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        
        model = MLPRegressor(input_dim=23).to(device)
        criterion = get_loss_function_dl(config)
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)

        if config['train']['test'] == True:
            model = dl_train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config)
            dl_test(model, test_loader, device, config, test_df)
        
        else:
            dl_train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config)

    else: #transformer
        print('FTtransformer 준비중')


    # 테스트까지


if __name__ == '__main__':
    main()
