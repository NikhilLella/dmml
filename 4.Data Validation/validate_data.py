import pandas as pd
from pathlib import Path

import logging

from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the correct output directories based on the project root
LOG_DIR = PROJECT_ROOT / "metadata" / "log"
RAW_DATA_DIR = PROJECT_ROOT / "3.Rawdata"
KAGGLE_OUTPUT_DIR = RAW_DATA_DIR / "kaggle"
HF_OUTPUT_DIR = RAW_DATA_DIR / "huggingface"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# --- Setup Logging ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = LOG_DIR / f"validation_{timestamp}.log"

# Handlers
file_handler = logging.FileHandler(log_filename)
console_handler = logging.StreamHandler()

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Root logger
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

logging.info(f"Logging initialized. Project Root: {PROJECT_ROOT}")
logging.info(f"Log files will be saved to: {LOG_DIR}")
logging.info(f"Raw data will be saved to: {RAW_DATA_DIR}")

data=pd.read_csv(KAGGLE_OUTPUT_DIR/"WA_Fn-UseC_-Telco-Customer-Churn.csv")
logging.info(f"The shape of the data is {data.shape}")
logging.info(f"The columns  of the telecom customer churn data  is {data.columns}")
logging.info(f"Number of Duplicate rows of data found  {data.duplicated().sum()}")

