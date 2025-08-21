import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from datetime import datetime

# -----------------------------
# Setup Logging
# -----------------------------

# Define the correct output directories based on the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "metadata" / "log"
KAGGLE_DATA_DIR =  PROJECT_ROOT / "3.Rawdata"/"kaggle"/"WA_Fn-UseC_-Telco-Customer-Churn.csv"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- Setup Logging ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = LOG_DIR / f"validation{timestamp}.log"

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



def validate_data(df: pd.DataFrame, dataset_name: str, output_dir="data_quality_reports"):
    """Validate dataframe and generate data quality report"""

    logging.info(f"Starting data validation for dataset: {dataset_name}")

    report_rows = []

    # 1. Missing values
    missing_values = df.isnull().sum()
    for col, count in missing_values.items():
        report_rows.append(["Missing Values", col, count])
    logging.info("Missing values check complete.")

    # 2. Duplicates
    duplicate_rows = int(df.duplicated().sum())
    report_rows.append(["Duplicate Rows", "all_columns", duplicate_rows])
    logging.info("Duplicate rows check complete.")

    # 3. Data types
    for col, dtype in df.dtypes.astype(str).items():
        report_rows.append(["Data Type", col, dtype])
    logging.info("Data types check complete.")

    # 4. Numeric ranges (check negative values)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if (df[col] < 0).any():
            report_rows.append(["Range Issue", col, "Contains negative values"])

    logging.info("Numeric range check complete.")

    # 5. Anomalies (z-score > 3)
    for col in numeric_columns:
        if df[col].std() != 0:  # avoid division by zero
            z_scores = (df[col] - df[col].mean()) / df[col].std()
            anomaly_count = int((np.abs(z_scores) > 3).sum())
            if anomaly_count > 0:
                logging.warning("Anomalies detected...... in the data")
            report_rows.append(["Anomalies", col, anomaly_count])

    logging.info("Anomaly detection complete.")

    # -----------------------------
    # Save report
    # -----------------------------
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/{dataset_name}_quality_report_{timestamp}.csv"

    report_df = pd.DataFrame(report_rows, columns=["Check", "Column", "Details"])
    report_df.to_csv(output_file, index=False)

    logging.info(f" Data Quality Report saved at {output_file}")

    return report_df


# -------------------------
# Example usage after ingestion
# -------------------------
if __name__ == "__main__":
    try:
        df = pd.read_csv(KAGGLE_DATA_DIR)
        report = validate_data(df, dataset_name="telco_churn")
        print(report.head())  # show first few rows of the report
    except Exception as e:
        logging.error(f"Error during validation: {e}")