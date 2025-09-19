# app/inference.py
import os
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/EfficientNetV2B0_Original_best.keras")
MODEL_KIND = None    # "torch" or "tf"
MODEL = None
CLASS_NAMES = os.getenv("CLASS_NAMES", "")  # optional: "cat,dog,car"
CLASS_NAMES = [c.strip() for c in CLASS_NAMES.split(",")] if CLASS_NAMES else None

# --- lazy imports (โหลดเมื่อจำเป็น) ---
def _try_import_torch():
    try:
        import torch
        return torch
    except Exception:
        return None

def _try_import_tf():
    try:
        import tensorflow as tf
        return tf
    except Exception:
        return None

def detect_model_kind(path: str):
    p = path.lower()
    if p.endswith((".pt", ".pth")):
        return "torch"
    if p.endswith((".h5", ".keras")) or "savedmodel" in p:
        return "tf"
    # เดา: ทดลองโหลดตามลำดับ
    if _try_import_torch() is not None:
        return "torch"
    return "tf"

def load_model():
    global MODEL_KIND, MODEL
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")
    MODEL_KIND = detect_model_kind(MODEL_PATH)

    if MODEL_KIND == "torch":
        torch = _try_import_torch()
        assert torch is not None, "PyTorch not installed. Add to requirements.txt"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        MODEL = torch.jit.load(MODEL_PATH, map_location=device) if MODEL_PATH.endswith(".pt") else torch.load(MODEL_PATH, map_location=device)
        MODEL.eval()
    else:
        tf = _try_import_tf()
        assert tf is not None, "TensorFlow not installed. Add to requirements.txt"
        
        # Try different loading strategies for version compatibility
        try:
            # First try: Standard loading
            if MODEL_PATH.endswith((".h5", ".keras")):
                MODEL = tf.keras.models.load_model(MODEL_PATH)
            else:
                MODEL = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e1:
            logger.warning(f"Standard loading failed: {e1}")
            try:
                # Second try: Load with custom objects and compile=False
                if MODEL_PATH.endswith((".h5", ".keras")):
                    MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
                else:
                    MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
                logger.info("Model loaded successfully with compile=False")
            except Exception as e2:
                logger.warning(f"Loading with compile=False failed: {e2}")
                try:
                    # Third try: Load with custom objects
                    custom_objects = {
                        'tf_keras': tf.keras,
                        'keras': tf.keras,
                        'Functional': tf.keras.Model
                    }
                    if MODEL_PATH.endswith((".h5", ".keras")):
                        MODEL = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
                    else:
                        MODEL = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
                    logger.info("Model loaded successfully with custom objects")
                except Exception as e3:
                    logger.warning(f"Loading with custom objects failed: {e3}")
                    # Fourth try: Use tf.saved_model.load for SavedModel format
                    if not MODEL_PATH.endswith((".h5", ".keras")):
                        try:
                            MODEL = tf.saved_model.load(MODEL_PATH)
                            logger.info("Model loaded successfully as SavedModel")
                        except Exception as e4:
                            logger.error(f"SavedModel loading failed: {e4}")
                            raise Exception(f"All loading methods failed. Last error: {e4}")
                    else:
                        raise Exception(f"All loading methods failed. Last error: {e3}")
    return MODEL_KIND, MODEL

def preprocess_image(img_array, target_size=(224, 224)):
    """
    img_array: RGB np.array (H,W,3), dtype uint8
    return: float32 (1, H, W, 3) scaled 0-1
    """
    import cv2
    h, w = target_size
    img = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def postprocess_logits(logits: np.ndarray):
    """
    รองรับทั้ง binary/multi-class
    return: dict(probabilities=[...], top_idx=int, top_label=str, top_conf=float)
    """
    probs = None
    if logits.ndim == 1:
        # (C,)
        exps = np.exp(logits - np.max(logits))
        probs = exps / exps.sum()
    else:
        # (1, C)
        logits = logits[0]
        exps = np.exp(logits - np.max(logits))
        probs = exps / exps.sum()
    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    top_label = CLASS_NAMES[top_idx] if CLASS_NAMES and top_idx < len(CLASS_NAMES) else str(top_idx)
    return {
        "probabilities": probs.tolist(),
        "top_idx": top_idx,
        "top_label": top_label,
        "top_conf": top_conf,
    }

def predict_from_ndarray(img_array) -> dict:
    global MODEL
    if MODEL is None:
        load_model()

    x = preprocess_image(img_array, target_size=(224, 224))

    if MODEL_KIND == "torch":
        import torch
        with torch.no_grad():
            x_t = torch.from_numpy(x).permute(0,3,1,2).contiguous()  # N,H,W,3 -> N,3,H,W
            logits = MODEL(x_t).cpu().numpy()
    else:
        # Handle different TensorFlow model types
        if hasattr(MODEL, 'predict'):
            # Standard Keras model
            logits = MODEL.predict(x, verbose=0)
        elif hasattr(MODEL, '__call__'):
            # SavedModel or functional model
            try:
                logits = MODEL(x, training=False).numpy()
            except:
                # Fallback for SavedModel
                logits = MODEL(x).numpy()
        else:
            raise ValueError("Unknown model type")

    return postprocess_logits(logits)
