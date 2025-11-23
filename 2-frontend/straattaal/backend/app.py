"""
FastAPI Backend voor Dog Name Generator
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import torch
import uvicorn
import json

# Import slanggen model
from slanggen import models

app = FastAPI(
    title="Dog Name Generator API",
    description="Generate unique dog names using AI",
    version="1.0.0"
)

# Paden (relatief vanaf backend/)
MODEL_PATH = Path(__file__).parent.parent / "artefacts"
STATIC_PATH = Path(__file__).parent / "static"

# Global variables
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GenerateRequest(BaseModel):
    """Request model"""
    prompt: str = ""
    max_length: int = 15
    temperature: float = 1.0


class GenerateResponse(BaseModel):
    """Response model"""
    generated_text: str
    prompt: str


@app.on_event("startup")
async def load_model():
    """Load model at startup"""
    global model, tokenizer
    
    print("🚀 Starting Dog Name Generator API...")
    print(f"📁 Model path: {MODEL_PATH}")
    print(f"📁 Static path: {STATIC_PATH}")
    
    try:
        # Load tokenizer
        tokenizer_file = MODEL_PATH / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_file}")
        
        print("📦 Loading tokenizer...")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        print(f"✅ Tokenizer loaded! Vocab size: {tokenizer.get_vocab_size()}")
        
        # Load config from training
        config_file = MODEL_PATH / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")
        
        print("📦 Loading config...")
        with open(config_file, 'r') as f:
            training_config = json.load(f)
        
        # Use the EXACT model config from training
        modelconfig = {
            "vocab_size": tokenizer.get_vocab_size(),
            "embedding_dim": training_config["model"]["embedding_dim"],
            "hidden_dim": training_config["model"]["hidden_dim"],
            "num_layers": training_config["model"]["num_layers"]
        }
        
        print(f"📋 Model config: {modelconfig}")
        
        # Load model
        model_file = MODEL_PATH / "model.pth"
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")
        
        print("📦 Loading model...")
        
        model = models.SlangRNN(modelconfig)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded on {device}!")
        print("✅ API ready to generate dog names!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


@app.post("/generate", response_model=GenerateResponse)
async def generate_name(request: GenerateRequest):
    """Generate dog name"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Encode prompt
        if request.prompt:
            encoded = tokenizer.encode(request.prompt)
            input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(device)
        else:
            # Start with random token
            input_ids = torch.randint(
                0, tokenizer.get_vocab_size(), (1, 1), dtype=torch.long
            ).to(device)
        
        # Generate
        generated = []
        hidden = model.init_hidden(input_ids)
        
        with torch.no_grad():
            for _ in range(request.max_length):
                output, hidden = model(input_ids, hidden)
                
                # Apply temperature
                logits = output[0, -1] / request.temperature
                probs = torch.softmax(logits, dim=0)
                
                # Sample next token
                next_token = torch.multinomial(probs, 1)
                generated.append(next_token.item())
                
                # Update input
                input_ids = next_token.unsqueeze(0)
                
                # Stop at space or newline
                decoded = tokenizer.decode([next_token.item()])
                if decoded in [' ', '\n', '<', '>']:
                    break
        
        # Decode result
        generated_text = tokenizer.decode(generated)
        
        return GenerateResponse(
            generated_text=generated_text.strip(),
            prompt=request.prompt
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }


# Serve frontend
@app.get("/")
async def read_root():
    """Serve frontend HTML"""
    index_file = STATIC_PATH / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_file)


# Mount static files (if you have CSS/JS files)
if STATIC_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")


if __name__ == "__main__":
    print("🐕 Starting Dog Name Generator...")
    print("=" * 50)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=80,
        log_level="info"
    )