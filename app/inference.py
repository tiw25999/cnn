# app/inference.py
import os
import sys
import types
import logging
from functools import lru_cache
from typing import Dict, Any, Optional

# ถ้าต้องการใช้ tf-keras legacy ให้ลองตั้ง env ไว้ตั้งแต่เริ่มโปรเซส (ไม่มีผลถ้านำเข้าไปแล้ว)
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
import keras

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
keras.layers.Lambda = CompatibleLambda
tf.keras.layers.Lambda = CompatibleLambda
logger.info("Patched tf.keras.layers.Lambda and keras.layers.Lambda -> CompatibleLambda")

# คีย์ชื่อที่อาจปรากฏในไฟล์โมเดลหลายยุค
def _custom_objects() -> Dict[str, Any]:
    return {
        # Lambda aliases
        "Lambda": CompatibleLambda,
        "KerasLayers>Lambda": CompatibleLambda,
        "keras.layers.core.lambda_layer.Lambda": CompatibleLambda,
        "tf_keras.src.layers.core.lambda_layer.Lambda": CompatibleLambda,
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

def _resolve_model_path() -> str:
    for k in MODEL_ENV_PATHS:
        p = os.getenv(k)
        if p and os.path.exists(p):
            return p
    for p in DEFAULT_MODEL_PATHS:
        if os.path.exists(p):
            return p
    # ถ้าไม่เจอ ปล่อยพาธแรกไว้ให้ error ชัดเจน
    return DEFAULT_MODEL_PATHS[0]

# ---------- 4) โหลดโมเดลด้วย fallback หลายแบบ ----------
def _try_load_keras_api(path: str):
    # Keras 3 API
    try:
        from keras.saving import load_model as kload
    except Exception:
        kload = None

    if kload:
        logger.info("Loading via keras.saving.load_model(...)")
        return kload(path, custom_objects=_custom_objects(), safe_mode=False)
    raise RuntimeError("keras.saving.load_model not available")

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
    return keras.models.load_model(
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
            return model
        except Exception as e:
            logger.warning("%s load failed: %s", name, e)
            last_err = e

    raise RuntimeError(f"All loading methods failed. Last error: {last_err}")

# ---------- 5) พรีโพรเซสและพยากรณ์ ----------
def _resize_and_scale(img: np.ndarray, size=(224, 224)) -> np.ndarray:
    """รับ ndarray (H,W,3) RGB/BGR/uint8 ก็ได้ -> float32 [0,1] 224x224"""
    x = img
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    # ถ้าเป็น BGR (จาก OpenCV) ให้สลับเป็น RGB แบบปลอดภัย
    if x.shape[-1] == 3 and x[..., 0].mean() > x[..., 2].mean():
        # heuristic เล็ก ๆ: ถ้าช่องแรกเฉลี่ยสว่างกว่าช่องแดงมาก ให้ลองสลับ BGR->RGB
        x = x[..., ::-1]
    x = tf.image.resize(x, size, method="bilinear").numpy()
    x = x / 255.0
    return x

def predict_from_ndarray(img: np.ndarray) -> Dict[str, Any]:
    """
    รับภาพเป็น ndarray แล้วคืนค่า:
    {
        "ok": True,
        "probabilities": [..],
        "top1": {"index": int, "prob": float}
    }
    """
    try:
        model = load_model()
    except Exception as e:
        logger.error("Model load error: %s", e)
        return {"ok": False, "error": str(e)}

    try:
        x = _resize_and_scale(img, size=(224, 224))
        x = np.expand_dims(x, axis=0)  # (1,224,224,3)
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
        }
    except Exception as e:
        logger.exception("Predict error")
        return {"ok": False, "error": f"Predict failed: {e}"}
