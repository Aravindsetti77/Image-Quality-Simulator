import numpy as np
import cv2
from app.core.engine import QualityEngine

engine = QualityEngine()
image = np.zeros((1080, 1920, 3), dtype=np.uint8)

# Force the fallback to run by giving a bad model path
engine.model_path = "non_existent_model.pb"

try:
    res = engine.process(image, "remux", "2160", False)
    print("Fallback success, shape:", res.shape)
except Exception as e:
    print("Fallback error:", repr(e))
