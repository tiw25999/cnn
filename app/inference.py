# app/inference.py
import os
import sys
import types
import logging
from functools import lru_cache
from typing import Dict, Any, Optional

# Ensure standalone Keras (if used) selects TensorFlow backend
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf

logger = logging.getLogger("app.inference")
logger.setLevel(logging.INFO)

# ---------- 2) Lambda ที่เข้ากันได้กับเมทาดาทาเก่า ----------
class CompatibleLambda(tf.keras.layers.Lambda):
    """
    Lambda ที่รองรับคีย์เวิร์ดจากโมเดล Keras เวอร์ชันเก่า
    เช่น function_type/module/output_shape_type/output_shape_module
    """
    def __init__(
        self,
        function=None,
        output_shape=None,
        mask=None,
        arguments=None,
        **kwargs,
    ):
        # ตัดคีย์ที่ Keras 3 ไม่รู้จักทิ้ง
        for k in ("function_type", "module", "output_shape_type", "output_shape_module"):
            kwargs.pop(k, None)
        
        # Handle preprocess_input function specially
        if function == "preprocess_input":
            function = tf.keras.applications.efficientnet_v2.preprocess_input
        elif isinstance(function, dict) and function.get('config') == 'preprocess_input':
            function = tf.keras.applications.efficientnet_v2.preprocess_input
            
        super().__init__(
            function=function,
            output_shape=output_shape,
            mask=mask,
            arguments=arguments,
            **kwargs,
        )

    # กันกรณี serialize อีกครั้ง
    def get_config(self):
        cfg = super().get_config()
        for k in ("function_type", "module", "output_shape_type", "output_shape_module"):
            cfg.pop(k, None)
        return cfg

# ลงทะเบียน/monkey-patch ให้แน่ใจว่า deserializer จะเจอ
tf.keras.layers.Lambda = CompatibleLambda
try:
    import keras as _ks
    _ks.layers.Lambda = CompatibleLambda
    logger.info("Patched keras.layers.Lambda -> CompatibleLambda")
except Exception:
    logger.info("Standalone keras not available at import-time; will import lazily")
logger.info("Patched tf.keras.layers.Lambda -> CompatibleLambda")

# คีย์ชื่อที่อาจปรากฏในไฟล์โมเดลหลายยุค
def _custom_objects() -> Dict[str, Any]:
    return {
        # Lambda aliases
        "Lambda": CompatibleLambda,
        "KerasLayers>Lambda": CompatibleLambda,
        "keras.layers.core.lambda_layer.Lambda": CompatibleLambda,
        "tf_keras.src.layers.core.lambda_layer.Lambda": CompatibleLambda,
        # Preprocessing functions
        "preprocess_input": tf.keras.applications.efficientnet_v2.preprocess_input,
        "function": lambda x: x,  # Fallback for function deserialization
        # ช่วย mapping อื่น ๆ ที่บางทีโมเดลฝังชื่อไว้
        "tf_keras": tf.keras,
        "keras": tf.keras,
        "Functional": tf.keras.Model,
        "Model": tf.keras.Model,
        "Sequential": tf.keras.Sequential,
    }

# ---------- 3) เส้นทางโมเดล ----------
MODEL_ENV_PATHS = [
    "MODEL_PATH",           # สามารถตั้ง env ได้
    "MODEL_FILE",           # เผื่อชื่ออื่น
]
DEFAULT_MODEL_PATHS = [
    "/app/models/EfficientNetV2B0_Original_best.keras",
    "models/EfficientNetV2B0_Original_best.keras",
    "EfficientNetV2B0_Original_best.keras",
]

_MODEL_PRIORITY = [
    "EfficientNetV2B0",
    "EfficientNetB0",
    "MobileNetV3Large",
    "MobileNetV2",
    "NASNetMobile",
]

def _scan_models_dir(base_dir: str = "models") -> Optional[str]:
    """ค้นหาไฟล์โมเดลที่เหมาะสมที่สุดในโฟลเดอร์ models
    เกณฑ์เลือก:
      1) ชื่อโมเดลตามลำดับ _MODEL_PRIORITY
      2) เลือกไฟล์ที่ลงท้ายด้วย _Original_best.keras ก่อน
      3) รองลงมา _Original.keras
      4) รองลงมา *_best.keras แล้วค่อย *.keras อื่น ๆ
      5) หากมีหลายไฟล์ตรงกัน ให้เลือกขนาดใหญ่สุด
    """
    try:
        if not os.path.isdir(base_dir):
            return None
        candidates = []  # (priority_idx, tier, size, fullpath)
        def tier_of(name: str) -> int:
            if name.endswith("_Original_best.keras"): return 0
            if name.endswith("_Original.keras"):      return 1
            if name.endswith("_best.keras"):          return 2
            if name.endswith(".keras"):               return 3
            return 9
        for fname in os.listdir(base_dir):
            if not fname.lower().endswith(".keras"):
                continue
            full = os.path.join(base_dir, fname)
            try:
                size = os.path.getsize(full)
            except Exception:
                size = 0
            pr_idx = len(_MODEL_PRIORITY)
            for i, pref in enumerate(_MODEL_PRIORITY):
                if fname.startswith(pref + "_"):
                    pr_idx = i
                    break
            candidates.append((pr_idx, tier_of(fname), -size, full))
        if not candidates:
            return None
        # sort by priority_idx asc, tier asc, size desc (via -size)
        candidates.sort()
        return candidates[0][3]
    except Exception:
        return None

def _resolve_model_path() -> str:
    for k in MODEL_ENV_PATHS:
        p = os.getenv(k)
        if p and os.path.exists(p):
            return p
    for p in DEFAULT_MODEL_PATHS:
        if os.path.exists(p):
            return p
    # scan models/ directory for best available
    scanned = _scan_models_dir("models")
    if scanned and os.path.exists(scanned):
        logger.info("Auto-selected model: %s", scanned)
        return scanned
    # ถ้าไม่เจอ ปล่อยพาธแรกไว้ให้ error ชัดเจน
    return DEFAULT_MODEL_PATHS[0]

# ---------- 4) โหลดโมเดลด้วย fallback หลายแบบ ----------
def _try_load_keras_api(path: str):
    # Keras 3 API
    try:
        from keras.saving import load_model as kload
        logger.info("Loading via keras.saving.load_model(...)")
        return kload(path, custom_objects=_custom_objects(), safe_mode=False)
    except Exception as e:
        raise RuntimeError(f"keras.saving.load_model not available: {e}")

def _try_load_tf_keras(path: str):
    logger.info("Loading via tf.keras.models.load_model(...)")
    return tf.keras.models.load_model(
        path,
        custom_objects=_custom_objects(),
        compile=False,
        safe_mode=False,
    )

def _try_load_keras_models(path: str):
    logger.info("Loading via keras.models.load_model(...)")
    # import ภายในฟังก์ชัน เพื่อลดปัญหา import-time
    import keras as ks
    return ks.models.load_model(
        path,
        custom_objects=_custom_objects(),
        compile=False,
        safe_mode=False,
    )

def _try_load_saved_model(path: str):
    logger.info("Loading via tf.saved_model.load(...)")
    return tf.saved_model.load(path)

@lru_cache(maxsize=1)
def load_model() -> Any:
    path = _resolve_model_path()
    logger.info("Resolved model path: %s", path)

    last_err: Optional[Exception] = None
    attempts = [
        ("tf.keras", _try_load_tf_keras),
        ("keras.models", _try_load_keras_models),
        ("keras.saving", _try_load_keras_api),
        ("tf.saved_model", _try_load_saved_model),
    ]

    for name, fn in attempts:
        try:
            model = fn(path)
            logger.info("Model loaded with method: %s", name)
            logger.info("Using model file: %s", path)
            return model
        except Exception as e:
            logger.warning("%s load failed: %s", name, e)
            last_err = e

    raise RuntimeError(f"All loading methods failed. Last error: {last_err}")

def get_current_model_info() -> Dict[str, Any]:
    """คืนข้อมูลโมเดลที่ใช้อยู่ปัจจุบัน"""
    try:
        path = _resolve_model_path()
        model_name = os.path.basename(path)
        file_size = os.path.getsize(path) if os.path.exists(path) else 0
        
        return {
            "model_path": path,
            "model_name": model_name,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "exists": os.path.exists(path)
        }
    except Exception as e:
        return {
            "error": str(e),
            "exists": False
        }

# ---------- 5) พรีโพรเซสและพยากรณ์ ----------
def _resize_and_scale(img: np.ndarray, size=(192, 192)) -> np.ndarray:
    """รับ ndarray (H,W,3) RGB/BGR/uint8 ก็ได้ -> float32 [0,1] 192x192"""
    x = img
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    # ถ้าเป็น BGR (จาก OpenCV) ให้สลับเป็น RGB แบบปลอดภัย
    if x.shape[-1] == 3 and x[..., 0].mean() > x[..., 2].mean():
        # heuristic เล็ก ๆ: ถ้าช่องแรกเฉลี่ยสว่างกว่าช่องแดงมาก ให้ลองสลับ BGR->RGB
        x = x[..., ::-1]
    # แปลงเป็นขาวดำก่อน แล้วคงรูปเป็น 3 แชนเนลเพื่อให้เข้ากับโมเดลที่รับ 3 แชนเนล
    if x.ndim == 3 and x.shape[-1] == 3:
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x = tf.image.rgb_to_grayscale(x)            # (H,W,1)
        x = tf.image.grayscale_to_rgb(x)            # (H,W,3)
        x = x.numpy()
    x = tf.image.resize(x, size, method="bilinear").numpy()
    x = x / 255.0
    return x

def predict_from_ndarray(img: np.ndarray) -> Dict[str, Any]:
    """
    รับภาพเป็น ndarray แล้วคืนค่า:
    {
        "ok": True,
        "probabilities": [..],
        "top1": {"index": int, "prob": float},
        "model_info": {"name": str, "path": str, "size_mb": float}
    }
    """
    try:
        model = load_model()
    except Exception as e:
        logger.error("Model load error: %s", e)
        return {"ok": False, "error": str(e)}

    try:
        # เก็บข้อมูลโมเดลที่ใช้
        model_info = get_current_model_info()
        
        x = _resize_and_scale(img, size=(192, 192))
        x = np.expand_dims(x, axis=0)  # (1,192,192,3)
        preds = model(x, training=False) if callable(model) else model.predict(x)
        preds = np.array(preds)
        if preds.ndim == 2:
            probs = preds[0]
        elif preds.ndim == 1:
            probs = preds
        else:
            probs = preds.reshape(-1)
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        return {
            "ok": True,
            "probabilities": [float(p) for p in probs.tolist()],
            "top1": {"index": top_idx, "prob": top_prob},
            "model_info": {
                "name": model_info.get("model_name", "Unknown"),
                "path": model_info.get("model_path", "Unknown"),
                "size_mb": model_info.get("file_size_mb", 0)
            }
        }
    except Exception as e:
        logger.exception("Predict error")
        return {"ok": False, "error": f"Predict failed: {e}"}
