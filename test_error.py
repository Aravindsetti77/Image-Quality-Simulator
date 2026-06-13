import numpy as np
import cv2
from app.core.engine import QualityEngine

engine = QualityEngine()
image = np.zeros((1080, 1920, 3), dtype=np.uint8)
try:
    res = engine.process(image, "remux", "2160", False)
    print("Success, shape:", res.shape)
except Exception as e:
    print("Error:", repr(e))
