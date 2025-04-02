import sys
import os
import logging
from datetime import datetime, timedelta, timezone

os.makedirs('logs', exist_ok=True)
kst = timezone(timedelta(hours=9))
timestamp = datetime.now(kst).strftime('%Y%m%d_%H%M%S')
LOG_FILE = f'logs/process_{timestamp}.log'

logger = logging.getLogger('project_logger')
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)

file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()


tee = TeeLogger(LOG_FILE)
sys.stdout = tee
sys.stderr = tee