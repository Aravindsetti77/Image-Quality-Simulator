import io
import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
import gc

from app.core.engine import QualityEngine

# Prevent OpenCV from allocating too many threads and consuming memory
cv2.setNumThreads(1)

app = FastAPI(title="QualityEngine API", description="Image Quality Simulation and Upscaling Engine")
engine = QualityEngine()

VALID_TIERS = {
    "none", "camrip", "telesync", "ts", "telecine", "tc", "screener", "scr",
    "dvdrip", "hdtv", "webrip", "yify", "bdrip", "remux"
}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/api/process-format")
async def process_format(
    file: UploadFile = File(...),
    format_tier: str = Form(...),
    resolution: str = Form("none"),
    hdr: str = Form("false")
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
            resolution=resolution, 
            hdr=(hdr.lower() == "true")
        )
        
        _, encoded_img = cv2.imencode('.jpg', processed_image)
        
        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
        
    finally:
        # Aggressive memory cleanup to prevent memory leaks on constrained environments like Render
        if 'contents' in locals(): del contents
        if 'nparr' in locals(): del nparr
        if 'image' in locals(): del image
        if 'processed_image' in locals(): del processed_image
        if 'encoded_img' in locals(): del encoded_img
        gc.collect()


