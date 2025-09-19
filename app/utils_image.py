# app/utils_image.py
import io
import numpy as np
from PIL import Image

def bytes_to_rgb_ndarray(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img)
