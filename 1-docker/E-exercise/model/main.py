import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from loguru import logger
import sys
import uvicorn
from pydantic import BaseModel
import time

# Import your TextClustering class from the *other* file
try:
    from model import TextClustering
except ImportError:
    logger.error("Could not import TextClustering from model.py. Ensure file exists.")
    sys.exit(1)

# --- 1. Configuration ---
DATA_DIR_PROCESSED = Path("/app/data/processed")
LOG_DIR = Path("/app/logs")
INPUT_FILE = DATA_DIR_PROCESSED / "posts.parquet"

# --- 2. Logging Setup ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_DIR / "model_api.log", rotation="1 MB", level="INFO")

logger.info("Model API service starting up...")

# --- 3. Load Model and Data (on startup) ---
# Wait for the preprocess service to create the file
while not INPUT_FILE.exists():
    logger.warning(f"Input file {INPUT_FILE} not found. Waiting 10 seconds for preprocess service...")
    time.sleep(10)

logger.info(f"Loading data from {INPUT_FILE}...")
try:
    df = pd.read_parquet(INPUT_FILE)
    logger.success(f"Successfully loaded data. {len(df)} rows.")
except Exception as e:
    logger.critical(f"Failed to load data: {e}")
    df = pd.DataFrame() # Create empty dataframe to prevent crash, but endpoints will fail

# Initialize your model one time
model_instance = TextClustering()
app = FastAPI()

# --- 4. Define API Request Body ---
# This lets us send parameters (k, batch, etc.) in a POST request
class ClusterRequest(BaseModel):
    k: int = 100
    batch: bool = True
    method: str = "PCA"

# --- 5. Define API Endpoints ---
@app.get("/health")
def health_check():
    """
    Health check endpoint required by docker-compose.yml
    """
    if df.empty:
        raise HTTPException(status_code=503, detail="Service is starting, data not yet loaded.")
    return {"status": "ok", "data_rows": len(df)}

@app.post("/run_clustering")
def run_clustering(request: ClusterRequest):
    """
    This endpoint runs the clustering algorithm.
    It takes 'k', 'batch', and 'method' as JSON input.
    """
    if df.empty:
        logger.error("Clustering failed: Dataframe is empty.")
        raise HTTPException(status_code=500, detail="Data not loaded. Check logs.")

    try:
        logger.info(f"Received clustering request: k={request.k}, method={request.method}")
        
        # Get text data from the loaded dataframe
        text_data = df["text"].astype(str).tolist()
        
        # Run the model (from your notebook)
        X = model_instance(
            text=text_data, 
            k=request.k, 
            batch=request.batch, 
            method=request.method
        )
        
        # Get the labels (from your notebook)
        labels = model_instance.get_labels(df)
        
        logger.success("Clustering complete.")
        
        # Return the results as JSON
        # We must convert numpy arrays to lists
        return {
            "message": "Clustering successful",
            "clusters": X.tolist(),
            "labels": labels.tolist()
        }
        
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # This part is for debugging and won't be used by the Dockerfile
    uvicorn.run(app, host="0.0.0.0", port=8000)