import os
import torch
import pandas as pd
from tqdm import tqdm
from utils.metrics import evaluate_df
from utils.logger import logger

def dl_test(model, test_loader, device, config, test_df):
    '''
    '''
    logger.info('[Testing Started] 테스트를 시작합니다...')
    
    model.eval()
    model.to(device)

    preds_home = []
    preds_away = []
    gts_home = []
    gts_away = []

    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(tqdm(test_loader, desc='[Testing]')):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).cpu()

            preds_home.extend(outputs[:, 0].tolist())
            preds_away.extend(outputs[:, 1].tolist())
            gts_home.extend(batch_y[:, 0].cpu().tolist())
            gts_away.extend(batch_y[:, 1].cpu().tolist())

            if i == 0:
                n_features = batch_x.shape[1]
    
    test_df = test_df.reset_index(drop=True)

    df_result = pd.DataFrame({
        'home': test_df['home'],
        'away': test_df['away'],
        'true_home_score': gts_home,
        'true_away_score': gts_away,
        'pred_home_score': preds_home,
        'pred_away_score': preds_away
    })

    save_dir = config['save']['output_path']
    pred_dir = os.path.join(save_dir, 'result')
    os.makedirs(pred_dir, exist_ok=True)

    pred_path = os.path.join(pred_dir, 'prediction.csv')
    df_result.to_csv(pred_path, index=False, float_format='%.4f')
    
    logger.info(f'[Testing Completed] 예측 결과가 다음 위치에 저장되었습니다: {pred_path}')
    
    matrics_path = os.path.join(pred_dir, 'evaluation_matrics.csv')
    df_score = evaluate_df(df_result, n_features)
    df_score.to_csv(matrics_path, index=False, float_format='%.4f')

    logger.info(f'[Evaluation Completed] 결과가 다음 위치에 저장되었습니다: {matrics_path}')


def ml_test(model, X_test, y_test, test_df, config):
    '''
    '''
    preds = model.predict(X_test)

    df_result = pd.DataFrame({
        'home': test_df['home'],
        'away': test_df['away'],
        'true_home_score': y_test.iloc[:, 0],
        'true_away_score': y_test.iloc[:, 1],
        'pred_home_score': preds[:, 0],
        'pred_away_score': preds[:, 1]
    })

    save_dir = config['save']['output_path']
    pred_dir = os.path.join(save_dir, 'result')
    os.makedirs(pred_dir, exist_ok=True)

    pred_path = os.path.join(pred_dir, 'prediction.csv')
    df_result.to_csv(pred_path, index=False, float_format='%.4f')
    logger.info(f'[Testing Completed] 예측 결과가 다음 위치에 저장되었습니다: {pred_path}')

    matrics_path = os.path.join(pred_dir, 'evaluation_metrics.csv')
    df_score = evaluate_df(df_result, X_test.shape[1])
    df_score.to_csv(matrics_path, index=False, float_format='%.4f')
    logger.info(f'[Evaluation Completed] 결과가 다음 위치에 저장되었습니다: {matrics_path}')