import os
import sys
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from datasets import load_dataset
import pandas as pd

# --- Dynamic Path Configuration ---
# This automatically finds the project root directory (dmml) from the script's location
# Path(__file__) is the path to this script (dmml/ingest/myPython.py)
# .parent is the directory of the script (dmml/ingest)
# .parent.parent is the project root (dmml)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the correct output directories based on the project root
LOG_DIR = PROJECT_ROOT / "metadata" / "log"
RAW_DATA_DIR = PROJECT_ROOT / "3.Rawdata"
KAGGLE_OUTPUT_DIR = RAW_DATA_DIR / "kaggle"
HF_OUTPUT_DIR = RAW_DATA_DIR / "huggingface"

# Create all necessary directories
LOG_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Setup Logging ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = LOG_DIR / f"ingestion_{timestamp}.log"

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

# --- Configuration ---
KAGGLE_DATASET_ID = "blastchar/telco-customer-churn"
HUGGINGFACE_DATASET_ID = "aai510-group1/telco-customer-churn"

def download_from_kaggle(dataset_id: str, path: str):
    """Download Kaggle dataset using API"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        logging.info(f"Starting Kaggle ingestion for {dataset_id}")
        command = [
            "kaggle", "datasets", "download",
            "-d", dataset_id,
            "-p", path,
            "--unzip"
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"Kaggle dataset {dataset_id} successfully downloaded to {path}/ at {timestamp}")
    except FileNotFoundError:
        logging.error("Kaggle CLI not found. Install via `pip install kaggle` and set up API token.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Kaggle download failed: {e.stderr}")

def download_from_huggingface(dataset_id: str, output_dir: Path):
    """Download HuggingFace dataset and save as CSV"""
    try:
        logging.info(f"Starting HuggingFace ingestion for {dataset_id}")
        dataset = load_dataset(dataset_id)
        df = dataset["train"].to_pandas()
        output_file = output_dir / "hf_churn.csv"
        df.to_csv(output_file, index=False)
        logging.info(f"HuggingFace dataset saved to {output_file}, rows={len(df)}")
        return df
    except Exception as e:
        logging.error(f"HuggingFace ingestion failed: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    download_from_kaggle(KAGGLE_DATASET_ID, KAGGLE_OUTPUT_DIR)
    download_from_huggingface(HUGGINGFACE_DATASET_ID, HF_OUTPUT_DIR)