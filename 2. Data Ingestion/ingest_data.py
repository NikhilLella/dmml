import os
import logging
import subprocess
from datasets import load_dataset
from datetime import datetime

# Create logs directory if it doesn’t exist
os.makedirs("logs", exist_ok=True)

# Create timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/ingestion_{timestamp}.log"

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

logging.info("Logging initialized with both file + console handlers")

# --- Config ---
KAGGLE_DATASET_ID = "blastchar/telco-customer-churn"
HUGGINGFACE_DATASET_ID = "aai510-group1/telco-customer-churn"

KAGGLE_OUTPUT_DIR = "raw_data/kaggle"
HF_OUTPUT_DIR = "raw_data/huggingface"

os.makedirs(KAGGLE_OUTPUT_DIR, exist_ok=True)
os.makedirs(HF_OUTPUT_DIR, exist_ok=True)


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


def download_from_huggingface(dataset_id: str, output_dir: str):
    """Download HuggingFace dataset and save as CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        logging.info(f"Starting HuggingFace ingestion for {dataset_id}")
        dataset = load_dataset(dataset_id)
        df = dataset["train"].to_pandas()
        output_file = os.path.join(output_dir, f"hf_churn.csv")
        df.to_csv(output_file, index=False)
        logging.info(f"HuggingFace dataset saved to {output_file}, rows={len(df)}")
        return df
    except Exception as e:
        logging.error(f"HuggingFace ingestion failed: {e}")
        return None


# --- Run ingestion ---
# --- Main Execution ---
if __name__ == "__main__":
     download_from_kaggle(KAGGLE_DATASET_ID, KAGGLE_OUTPUT_DIR)
     download_from_huggingface(HUGGINGFACE_DATASET_ID, HF_OUTPUT_DIR)

