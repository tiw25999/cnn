# app/inference.py
import os
import sys
import types
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ใช้ TF backend กับ Keras 3
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

MODEL_PATH = os.getenv("MODEL_PATH", "models/EfficientNetV2B0_Original_best.keras")
CLASS_NAMES = os.getenv("CLASS_NAMES", "")  # เช่น "cat,dog,car"
CLASS_NAMES = [c.strip() for c in CLASS_NAMES.split(",")] if CLASS_NAMES else None

MODEL = None
MODEL_KIND = "tf"  # โค้ดนี้โฟกัสที่ TensorFlow/Keras


# ---------- Utilities ----------
def _try_import_tf():
    try:
        import tensorflow as tf
        return tf
    except Exception:
        return None


def _patch_tf_keras_imports():
    """
    โมเดลที่เซฟมาบางครั้ง serialize เส้นทางเป็น tf_keras.src.engine.functional.Functional
    ซึ่งไม่มีอยู่ในสภาพแวดล้อมปัจจุบัน ให้สร้าง mock module/class เพื่อให้ deserialize ผ่าน
    """
    tf = _try_import_tf()
    if tf is None:
        return

    mock_targets = [
        ("tf_keras.src.engine.functional", {"Functional": tf.keras.Model}),
        ("tf_keras.src.engine.base_layer", {"Layer": tf.keras.layers.Layer}),
        ("tf_keras.src.engine.input_layer", {"InputLayer": tf.keras.layers.InputLayer}),
        ("tf_keras.src.engine.training", {"Model": tf.keras.Model}),
        ("tf_keras.src.engine.network", {"Network": tf.keras.Model}),
        ("tf_keras.src.engine.sequential", {"Sequential": tf.keras.Sequential}),
    ]

    for module_name, attrs in mock_targets:
        if module_name not in sys.modules:
            m = types.ModuleType(module_name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[module_name] = m
            logger.info(f"Created mock module: {module_name}")


def _build_custom_objects(tf):
    """
    - สร้าง CompatibleLambda ที่รองรับคีย์เวิร์ดเก่าของ Keras2/TF2
    - แมปเลเยอร์/โมเดลยอดนิยม
    - ใส่ตัวชี้ไปยัง mock tf_keras.src.engine.functional.Functional
    """
    # แมปฟังก์ชัน Lambda ให้ตรงกับชื่อเลเยอร์ที่พบในล็อก:
    #   - noop_aug     -> identity
    #   - to_255       -> x * 255.0
    #   - preprocess   -> tf.keras.applications.efficientnet_v2.preprocess_input
    def _lambda_from_name(name):
        if name == "noop_aug":
            return (lambda x: x)
        if name == "to_255":
            return (lambda x: x * 255.0)
        if name == "preprocess":
            # ใช้เวอร์ชันของ TF ปัจจุบัน
            return tf.keras.applications.efficientnet_v2.preprocess_input
        # fallback: identity
        return (lambda x: x)

    class CompatibleLambda(tf.keras.layers.Lambda):
        """
        - ตัดคีย์ที่ Keras3 ไม่รู้จักทิ้ง
        - หาก function ถูก serialize เป็นบล็อบ/ลิสต์จาก __main__, ให้กำหนดใหม่จากชื่อเลเยอร์
        """
        def __init__(self, function=None, **kwargs):
            # ตัดคีย์เก่าๆ ที่ทำให้ "Unrecognized keyword arguments passed to Lambda"
            kwargs.pop("function_type", None)
            kwargs.pop("module", None)
            kwargs.pop("output_shape_type", None)
            kwargs.pop("output_shape_module", None)

            # ถ้า function เป็น list/blob จาก serializer เก่า ให้แทนด้วย identity จนกว่าจะตั้งใหม่
            if isinstance(function, (list, tuple, bytes)):
                function = None

            super().__init__(function=function, **kwargs)

        @classmethod
        def from_config(cls, config):
            # clone & ล้างคีย์ที่ไม่รู้จัก
            cfg = dict(config)
            for k in ("function_type", "module", "output_shape_type", "output_shape_module"):
                cfg.pop(k, None)

            # function จาก serializer เก่าอาจมาเป็น list/bytes -> ทิ้งก่อน
            func = cfg.get("function", None)
            if isinstance(func, (list, tuple, bytes)):
                func = None
            cfg["function"] = func

            # ถ้าไม่มีฟังก์ชัน ให้หาใหม่จากชื่อเลเยอร์
            name = cfg.get("name") or cfg.get("config", {}).get("name")
            if cfg.get("function", None) is None and name:
                cfg["function"] = _lambda_from_name(name)

            return super().from_config(cfg)

    custom = {
        # ให้ deserializer หา 'Lambda' แล้วมาใช้ CompatibleLambda
        "Lambda": CompatibleLambda,

        # ชื่อคลาสยอดนิยมต่างๆ
        "tf_keras": tf.keras,
        "keras": tf.keras,
        "Functional": tf.keras.Model,
        "Model": tf.keras.Model,
        "Sequential": tf.keras.Sequential,
        "Input": tf.keras.Input,

        # เลเยอร์ที่พบบ่อย
        "Dense": tf.keras.layers.Dense,
        "Conv2D": tf.keras.layers.Conv2D,
        "MaxPooling2D": tf.keras.layers.MaxPooling2D,
        "GlobalAveragePooling2D": tf.keras.layers.GlobalAveragePooling2D,
        "GlobalMaxPooling2D": tf.keras.layers.GlobalMaxPooling2D,
        "AveragePooling2D": tf.keras.layers.AveragePooling2D,
        "Dropout": tf.keras.layers.Dropout,
        "BatchNormalization": tf.keras.layers.BatchNormalization,
        "ReLU": tf.keras.layers.ReLU,
        "Softmax": tf.keras.layers.Softmax,
        "Activation": tf.keras.layers.Activation,
        "Flatten": tf.keras.layers.Flatten,
        "Reshape": tf.keras.layers.Reshape,
        "Add": tf.keras.layers.Add,
        "Concatenate": tf.keras.layers.Concatenate,
        "Multiply": tf.keras.layers.Multiply,
        "ZeroPadding2D": tf.keras.layers.ZeroPadding2D,

        # แอปพลิเคชัน EfficientNet V2 (เผื่อฐานโมเดล)
        "EfficientNetV2B0": tf.keras.applications.EfficientNetV2B0,
        "EfficientNetV2B1": tf.keras.applications.EfficientNetV2B1,
        "EfficientNetV2B2": tf.keras.applications.EfficientNetV2B2,
        "EfficientNetV2B3": tf.keras.applications.EfficientNetV2B3,
        "EfficientNetV2S": tf.keras.applications.EfficientNetV2S,
        "EfficientNetV2M": tf.keras.applications.EfficientNetV2M,
        "EfficientNetV2L": tf.keras.applications.EfficientNetV2L,
        "EfficientNetB0": tf.keras.applications.EfficientNetB0,
    }

    # ใส่ทางลัดไปยัง mock module/class ด้วย (บางครั้ง Keras จะค้นด้วยสตริงเต็ม)
    if "tf_keras.src.engine.functional" in sys.modules:
        custom["tf_keras.src.engine.functional"] = sys.modules["tf_keras.src.engine.functional"]
        custom["tf_keras.src.engine.functional.Functional"] = tf.keras.Model

    return custom


# ---------- Loader ----------
def load_model():
    global MODEL
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    # เตรียม mock modules ก่อน
    _patch_tf_keras_imports()

    tf = _try_import_tf()
    assert tf is not None, "TensorFlow not installed."

    custom_objects = _build_custom_objects(tf)

    # 1) ลอง standalone keras (Keras 3) ก่อน พร้อม custom objects และปิด safe_mode
    try:
        import keras as ks
        MODEL = ks.models.load_model(
            MODEL_PATH,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False,
        )
        logger.info("Model loaded via standalone Keras with custom_objects (safe_mode=False, compile=False)")
        return "tf", MODEL
    except Exception as e:
        logger.warning(f"Standalone Keras load failed: {e}")

    # 2) ใช้ tf.keras เป็นทางสำรอง
    last_err = None
    for attempt in [
        dict(args={}, note="tf.keras standard"),
        dict(args={"compile": False}, note="tf.keras compile=False"),
        dict(args={"compile": False, "custom_objects": custom_objects}, note="tf.keras custom_objects+compile=False"),
        dict(args={"compile": False, "custom_objects": custom_objects, "safe_mode": False}, note="tf.keras custom_objects+compile=False+safe_mode=False"),
    ]:
        try:
            MODEL = tf.keras.models.load_model(MODEL_PATH, **attempt["args"])
            logger.info(f"Model loaded via {attempt['note']}")
            return "tf", MODEL
        except Exception as e:
            logger.warning(f"{attempt['note']} failed: {e}")
            last_err = e

    raise RuntimeError(f"All loading methods failed. Last error: {last_err}")


# ---------- Prediction ----------
def _ensure_loaded():
    global MODEL
    if MODEL is None:
        load_model()


def _prepare_image_tf(img_bgr_or_rgb: np.ndarray, target_size=(192, 192)):
    """
    รับภาพเป็น np.uint8 [H,W,3] (BGR หรือ RGB ก็ได้)
    - แปลงเป็น RGB
    - resize
    - scale 0..1 (ขั้นตอนในโมเดลจะ *255 แล้ว preprocess อีกทีตามเลเยอร์ Lambda)
    """
    import cv2
    img = img_bgr_or_rgb
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # เดาว่าอินพุตอาจมาเป็น BGR จาก OpenCV -> แปลงเป็น RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0  # 0..1
    return img


def predict_image(img_bgr_or_rgb: np.ndarray):
    """
    คืนค่า: dict { ok, preds, top, logits/probs, class_names }
    """
    _ensure_loaded()
    tf = _try_import_tf()
    assert tf is not None, "TensorFlow not installed."

    x = _prepare_image_tf(img_bgr_or_rgb)
    x = np.expand_dims(x, axis=0)  # [1,H,W,3]

    # เปิด graph ใน tf เพื่อรันให้เร็วขึ้น
    preds = MODEL(x, training=False) if hasattr(MODEL, "__call__") else MODEL.predict(x)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    preds = np.asarray(preds)

    # แปลงเป็น probs
    if preds.ndim == 2 and preds.shape[0] == 1:
        vec = preds[0]
    else:
        vec = preds

    # softmax หากยังไม่ใช่ความน่าจะเป็น
    if np.any(vec < 0) or not np.allclose(vec.sum(), 1.0, atol=1e-3):
        e = np.exp(vec - np.max(vec))
        probs = e / (e.sum() + 1e-8)
    else:
        probs = vec

    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])

    result = {
        "ok": True,
        "probs": probs.tolist(),
        "top_index": top_idx,
        "top_prob": top_prob,
    }
    if CLASS_NAMES and 0 <= top_idx < len(CLASS_NAMES):
        result["top_class"] = CLASS_NAMES[top_idx]
        result["class_names"] = CLASS_NAMES
    return result
