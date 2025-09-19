# app/inference.py
import os
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "models/your_model_file.h5")
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
        if MODEL_PATH.endswith((".h5", ".keras")):
            MODEL = tf.keras.models.load_model(MODEL_PATH)
        else:
            # กรณี SavedModel บีบเป็น zip ให้แตกไฟล์เอง หรือชี้ไปโฟลเดอร์ SavedModel
            MODEL = tf.keras.models.load_model(MODEL_PATH)
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
        logits = MODEL(x, training=False).numpy() if hasattr(MODEL, "predict") is False else MODEL.predict(x)

    return postprocess_logits(logits)
