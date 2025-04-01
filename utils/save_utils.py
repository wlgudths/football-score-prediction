import os

def save_to_csv(df, file_name, save_dir='./data'):
    """
    Summary: 데이터프레임을 csv파일로 저장하는 함수

    Args:
        df (pd.DataFrame): 저장할 데이터프레임
        file_name (str): 저장할 파일 이름
        save_dir (str): 저장할 디렉토리 경로 (기본값: './data')
    
    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file_name)
    df.to_csv(file_path, index=False)
    print(f"해당 경로에 {file_path} 저장 완료")