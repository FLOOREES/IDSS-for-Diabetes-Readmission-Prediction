# src/utils/logging_config.py
import logging
import sys
import os
from typing import Optional

def setup_logging(log_level=logging.INFO, log_file: Optional[str] = None):
    """ Configures logging for the project. """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
             os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='a')) # Append mode

    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    logging.info("Logging configured.")