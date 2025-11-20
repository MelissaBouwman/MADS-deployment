import requests
from pathlib import Path
from loguru import logger
import re
import sys
import time

# --- 1. Configuration ---
# These paths correspond to the 'volumes' in your docker-compose.yml
# We write to /app/data, which will appear in ./data on your Mac
DATA_DIR = Path("/app/data/raw") # This is the path INSIDE the container
LOG_DIR = Path("/app/logs")

# --- 2. Logging Setup ---
# Ensure the log directory exists (Docker won't create it for us)
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.remove()
# Add logging to the terminal (so you see it live in 'docker compose up')
logger.add(sys.stderr, level="INFO") 
# Add logging to the log file in the shared volume
logger.add(LOG_DIR / "ingest.log", rotation="1 MB", level="INFO") 

logger.info("Ingest script started...")

# --- Your Download Function ---
def download(url, datafile: Path):
    """
    Downloads a file from a URL and saves it to 'datafile'.
    Includes robust error handling.
    """
    datadir = datafile.parent
    if not datadir.exists():
        logger.info(f"Creating directory {datadir}")
        datadir.mkdir(parents=True)

    if not datafile.exists():
        logger.info(f"Downloading {url} to {datafile}")
        try:
            response = requests.get(url, timeout=30) # 30 sec timeout
            response.raise_for_status()  # Will stop if status is 4xx or 5xx
            
            with datafile.open("wb") as f:
                f.write(response.content)
            logger.success(f"File {datafile.name} downloaded successfully.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed for {url}: {e}")
            raise # Re-raise the exception to stop the script
    else:
        logger.info(f"File {datafile} already exists, skipping download.")

# --- 3. Main Logic (Your notebook code, combined) ---
try:
    # --- Task 1: Download 'posts.json' ---
    url_posts = "https://raw.githubusercontent.com/jkingsman/JSON-QAnon/main/posts.json"
    datafile_posts = DATA_DIR / "posts.json" # /app/data/raw/posts.json
    download(url_posts, datafile_posts)

    # --- Task 2: Download 'Tanach' books ---
    books = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua",
        "Judges",
        "1%20Samuel",
        "2%20Samuel",
        "1%20Kings",
        "2%20Kings",
        "Isaiah",
        "Jeremiah",
        "Ezekiel",
        "Hosea",
        "Joel",
        "Amos",
        "Obadiah",
        "Jonah",
        "Micah",
        "Nahum",
        "Habakkuk",
        "Zephaniah",
        "Haggai",
        "Zechariah",
        "Malachi"
    ]

    datadir_tanach = DATA_DIR / "tanach" # /app/data/raw/tanach

    logger.info(f"Starting download of {len(books)} Tanach books...")
    for book in books:
        # --- THIS IS THE FIXED LINE ---
        url_book = f"https://www.tanach.us/Server.txt?{book}*&content=Accents"
        # --- END OF FIX ---
        
        filename = re.sub(r"%20", "_", book)
        datafile_book = datadir_tanach / Path(f"{filename}.txt")
        
        download(url_book, datafile_book)
        time.sleep(0.5) # Be kind to their server
    
    logger.success("All ingest tasks completed.")

except Exception as e:
    logger.critical(f"Ingest script failed: {e}")
    sys.exit(1) # Exit with an error code so Docker knows it failed

logger.info("Ingest script finished successfully.")