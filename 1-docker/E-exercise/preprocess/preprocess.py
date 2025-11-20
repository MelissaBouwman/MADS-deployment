
import pandas as pd
from pandas import json_normalize
import json
from pathlib import Path
from loguru import logger
import re
import sys
import time
from datetime import datetime

#1. Configuration
# Set paths based on Docker volumes
DATA_DIR_RAW = Path("/app/data/raw")
DATA_DIR_PROCESSED = Path("/app/data/processed")
LOG_DIR = Path("/app/logs")

# Define input and output files
INPUT_FILE = DATA_DIR_RAW / "posts.json"
OUTPUT_FILE = DATA_DIR_PROCESSED / "posts.parquet"

#2. Logging Setup
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_DIR / "preprocess.log", rotation="1 MB", level="INFO")

logger.info("Preprocess script started...")

#3. Helper Functions
def bin_time(time):
    if time < datetime(2017, 12, 1):
        return 0
    elif time < datetime(2018, 1, 1):
        return 1
    elif time < datetime(2018, 8, 10):
        return 2
    elif time < datetime(2019, 8, 1):
        return 3
    else:
        return 4

def remove_url(text):
    return re.sub(r'^https?:\/\/.*[\r\n]*', '', text)

#4. Main Script Logic 
try:
    # Wait for the ingest service to create the file
    logger.info(f"Waiting for input file: {INPUT_FILE}...")
    while not INPUT_FILE.exists():
        logger.warning("Input file not found. Waiting 10 seconds for ingest service...")
        time.sleep(10)
    
    logger.info(f"Input file {INPUT_FILE} found. Starting preprocessing.")

    # --- Your notebook logic starts here ---
    with INPUT_FILE.open() as f:
        df = json_normalize(json.load(f)["posts"], sep="_")

    df["time"] = df["post_metadata_time"].apply(pd.to_datetime, unit="s")
    df["bintime"] = df["time"].apply(lambda x : bin_time(x))
    df["text"] = df["text"].apply(lambda x : str(x).replace("\n", " "))
    df["text"] = df["text"].apply(lambda x : remove_url(x))
    df["text"] = df["text"].apply(lambda x : x.lower())
    df['size'] = df['text'].apply(lambda x : len(str(x)))
    df = df[df["size"] > 50]
    df.reset_index(inplace=True, drop=True)
    
    logger.success("Data preprocessing and feature engineering complete.")
    
    # --- Your notebook logic ends here ---

    # Create processed directory and save file
    DATA_DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving preprocessed data to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE)
    
    logger.success(f"Data successfully saved to {OUTPUT_FILE}.")

except Exception as e:
    logger.critical(f"Preprocessing script failed: {e}")
    sys.exit(1) # Exit with an error code

logger.info("Preprocess script finished successfully.")