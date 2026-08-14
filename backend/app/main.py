import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import gc

from app.core.engine import QualityEngine

# Resolve paths relative to this file's location so the app works
# regardless of the working directory it's launched from.
BASE_DIR = Path(__file__).resolve().parent.parent  # points to backend/
STATIC_DIR = BASE_DIR.parent / "static"           # points to IMG/static/

# Prevent OpenCV from allocating too many threads and consuming memory
cv2.setNumThreads(1)

app = FastAPI(title="QualityEngine API", description="Image Quality Simulation and Upscaling Engine")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific domains in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = QualityEngine(model_path=str(BASE_DIR / "EDSR_x4.pb"))

VALID_TIERS = {
    "none", "camrip", "hdcam", "telesync", "ts", "hdts", "workprint", "wp", 
    "telecine", "tc", "screener", "scr", "vhsrip", "r5", "dvdrip", "pdtv", 
    "hdtv", "webrip", "hdrip", "web-dl", "yify", "bdscr", "brrip", "bdrip", 
    "remux", "uhd-remux"
}

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def read_root():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.post("/api/process-format")
async def process_format(
    file: UploadFile = File(...),
    format_tier: str = Form(...),
    resolution: str = Form("none")
):
    format_tier = format_tier.lower()
    if format_tier not in VALID_TIERS:
        raise HTTPException(status_code=400, detail=f"Invalid format_tier. Must be one of: {', '.join(VALID_TIERS)}")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    try:
        processed_image = engine.process(
            image, 
            format_tier, 
            resolution=resolution
        )
        
        _, encoded_img = cv2.imencode('.jpg', processed_image)
        
        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
    
    except Exception as e:
        # Fallback: just return the original image if anything goes fundamentally wrong
        _, encoded_img = cv2.imencode('.jpg', image)
        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
        
    finally:
        # Aggressive memory cleanup to prevent memory leaks on constrained environments like Render
        if 'contents' in locals(): del contents
        if 'nparr' in locals(): del nparr
        if 'image' in locals(): del image
        if 'processed_image' in locals(): del processed_image
        if 'encoded_img' in locals(): del encoded_img
        gc.collect()


