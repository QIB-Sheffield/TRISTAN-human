import logging
from datetime import datetime
import os

def create_logger(subject_details):
    """Function to create a run log"""
    
    subject, visit, scan = subject_details

    # Create a directory for log files if it doesn't exist
    os.makedirs('log_files', exist_ok=True)

    # Create a logger
    log   = logging.getLogger(__name__)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f's{subject}_v{visit}_s{scan}_{current_time}.log'

    # Set log file configuration
    
    logging.basicConfig(filename=os.path.join('log_files',log_name), encoding='utf-8', level=logging.DEBUG)
    
    return log


if __name__ == '__main__':
    log = create_logger('test')
    log.info('This is a test message')

