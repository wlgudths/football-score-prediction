from datetime import datetime
import pytz


def generate_run_id() -> str:
    '''
    '''
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    return now.strftime("%Y%m%d_%H%M%S")