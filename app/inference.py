# app/inference.py
import os
import sys
import types
import numpy as np
import logging

# ----- Logging -----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ให้ Standalone Keras ใช้ backend เป็น TF
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

MODEL_PATH = os.getenv("MODEL_PATH", "models/EfficientNetV2B0_Original_best.keras")
MODEL_KIND = None   # "torch" หรือ "tf"
MODEL = None
CLASS_NAMES = os.getenv("CLASS_NAMES", "")  # ตัวอย่าง: "cat,dog,car"
CLASS_NAMES = [c.strip() for c in CLASS_NAMES.split(",")] if CLASS_NAMES else None


# ===== Lazy import =====
def _try_import_torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def _try_import_tf_and_keras():
    """คืนค่า (tf, ks) ทั้งคู่ถ้าใช้ได้"""
    tf = None
    ks = None
    try:
        import tensorflow as _tf
        tf = _tf
    except Exception:
        pass
    try:
        import keras as _ks
        ks = _ks
    except Exception:
        # บางสภาพแวดล้อมไม่มี standalone keras
        pass
    return tf, ks


# ===== Patch: tf_keras internal modules ที่บางรุ่นไม่เจอ =====
def _patch_tf_keras_imports():
    mock_modules = [
        "tf_keras.src.engine.functional",
        "tf_keras.src.engine.base_layer",
        "tf_keras.src.engine.input_layer",
        "tf_keras.src.engine.training",
        "tf_keras.src.engine.network",
        "tf_keras.src.engine.sequential",
    ]
    for name in mock_modules:
        if name not in sys.modules:
            try:
                import tensorflow as tf
                m = types.ModuleType(name)
                # ใส่ class ที่มักอ้างอิง
                if "functional" in name:
                    m.Functional = tf.keras.Model
                if "base_layer" in name:
                    m.Layer = tf.keras.layers.Layer
                if "input_layer" in name:
                    m.InputLayer = tf.keras.layers.InputLayer
                if "training" in name:
                    m.Model = tf.keras.Model
                if "network" in name:
                    m.Network = tf.keras.Model
                if "sequential" in name:
                    m.Sequential = tf.keras.Sequential
                sys.modules[name] = m
                logger.info(f"Created mock module: {name}")
            except Exception as e:
                logger.warning(f"Could not create mock module {name}: {e}")


# ===== Patch: Lambda ให้ยอมรับคีย์ส่วนเกินจาก serialization เก่า =====
def _install_compatible_lambda(tf, ks):
    """
    Monkey-patch ให้ทั้ง tf.keras.layers.Lambda และ keras.layers.Lambda
    กลายเป็น CompatibleLambda ซึ่งจะ pop คีย์แปลกๆ ออกจาก kwargs ตอนสร้าง layer
    """
    if tf is None:
        return

    class CompatibleLambda(tf.keras.layers.Lambda):
        def __init__(self, function=None, **kwargs):
            # คีย์พิเศษจาก Keras รุ่น/วิธี serialize อื่น
            kwargs.pop("function_type", None)
            kwargs.pop("module", None)
            kwargs.pop("output_shape_type", None)
            kwargs.pop("output_shape_module", None)
            super().__init__(function=function, **kwargs)

    # แทนที่คลาสใน tf.keras
    try:
        tf.keras.layers.Lambda = CompatibleLambda  # type: ignore
        # ลงทะเบียนเป็น serializable ด้วย (ชื่อเดิม 'Lambda')
        try:
            # ทั้งสอง util ในบางเวอร์ชัน
            reg = getattr(tf.keras.utils, "register_keras_serializable", None)
            if callable(reg):
                reg(package="keras.layers", name="Lambda")(CompatibleLambda)
        except Exception:
            pass
        logger.info("Patched tf.keras.layers.Lambda -> CompatibleLambda")
    except Exception as e:
        logger.warning(f"Failed to patch tf.keras.layers.Lambda: {e}")

    # แทนที่คลาสใน standalone keras (ถ้ามี)
    if ks is not None:
        try:
            ks.layers.Lambda = CompatibleLambda  # type: ignore
            try:
                reg = getattr(ks.saving, "register_keras_serializable", None)
                if callable(reg):
                    reg(package="keras.layers", name="Lambda")(CompatibleLambda)
            except Exception:
                pass
            logger.info("Patched keras.layers.Lambda -> CompatibleLambda")
        except Exception as e:
            logger.warning(f"Failed to patch keras.layers.Lambda: {e}")


def _create_custom_objects(tf):
    """แมป class/ฟังก์ชันยอดฮิต + ใช้ Lambda ที่ patch แล้ว"""
    custom_objects = {
        # alias ที่บางโมเดล serialize มา
        "tf_keras": tf.keras,
        "keras": tf.keras,
        "Functional": tf.keras.Model,
        "Model": tf.keras.Model,
        "Sequential": tf.keras.Sequential,
        "Input": tf.keras.Input,
        # layers พื้นฐาน
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
        "Lambda": tf.keras.layers.Lambda,  # ถูก patch เป็น CompatibleLambda แล้ว
        "Add": tf.keras.layers.Add,
        "Concatenate": tf.keras.layers.Concatenate,
        "Multiply": tf.keras.layers.Multiply,
        "ZeroPadding2D": tf.keras.layers.ZeroPadding2D,
        # บางทีจะอ้าง path ภายใน
        "tf_keras.src.engine.functional": sys.modules.get("tf_keras.src.engine.functional"),
        "tf_keras.src.engine.functional.Functional": tf.keras.Model,
    }

    # EfficientNet / อื่นๆ (ถ้ามี)
    try:
        custom_objects.update({
            "EfficientNetV2B0": tf.keras.applications.EfficientNetV2B0,
            "EfficientNetV2B1": tf.keras.applications.EfficientNetV2B1,
            "EfficientNetV2B2": tf.keras.applications.EfficientNetV2B2,
            "EfficientNetV2B3": tf.keras.applications.EfficientNetV2B3,
            "EfficientNetV2S": tf.keras.applications.EfficientNetV2S,
            "EfficientNetV2M": tf.keras.applications.EfficientNetV2M,
            "EfficientNetV2L": tf.keras.applications.EfficientNetV2L,
            "EfficientNetB0": tf.keras.applications.EfficientNetB0,
            "EfficientNetB1": tf.keras.applications.EfficientNetB1,
            "EfficientNetB2": tf.keras.applications.EfficientNetB2,
            "EfficientNetB3": tf.keras.applications.EfficientNetB3,
            "EfficientNetB4": tf.keras.applications.EfficientNetB4,
            "EfficientNetB5": tf.keras.applications.EfficientNetB5,
            "EfficientNetB6": tf.keras.applications.EfficientNetB6,
            "EfficientNetB7": tf.keras.applications.EfficientNetB7,
            "ResNet50": tf.keras.applications.ResNet50,
            "ResNet50V2": tf.keras.applications.ResNet50V2,
        })
    except Exception:
        pass

    # preprocessing (ถ้ามี)
    try:
        custom_objects.update({
            "preprocess_input": tf.keras.applications.efficientnet_v2.preprocess_input,
            "decode_predictions": tf.keras.applications.efficientnet_v2.decode_predictions,
        })
    except Exception:
        pass

    return custom_objects


def detect_model_kind(path: str):
    p = path.lower()
    if p.endswith((".pt", ".pth")):
        return "torch"
    if p.endswith((".h5", ".keras")) or "savedmodel" in p:
        return "tf"
    # เดาให้ตาม lib ที่มี
    if _try_import_torch() is not None:
        return "torch"
    return "tf"


def load_model():
    """
    โหลดโมเดลด้วยลำดับ:
      1) (ถ้ามี) standalone keras + safe_mode=False (หลัง patch Lambda)
      2) tf.keras.load_model() มาตรฐาน
      3) tf.keras.load_model(custom_objects=..., compile=False, safe_mode=False)
      4) (ไม่ใช่ .h5/.keras) tf.saved_model.load()
      5) ความเข้ากันได้ TF v1 (สุดท้าย)
    """
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
        return MODEL_KIND, MODEL

    # TensorFlow / Keras branch
    tf, ks = _try_import_tf_and_keras()
    assert tf is not None, "TensorFlow not installed. Add to requirements.txt"

    # แพตช์โมดูลภายในก่อน
    _patch_tf_keras_imports()
    # แพตช์ Lambda ให้ยอมรับ kwargs แปลกๆ
    _install_compatible_lambda(tf, ks)

    # 1) Standalone Keras (ถ้ามี)
    if ks is not None:
        try:
            MODEL = ks.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
            logger.info("Model loaded via standalone Keras (safe_mode=False, compile=False)")
            return MODEL_KIND, MODEL
        except Exception as e:
            logger.warning(f"Standalone Keras load failed: {e}")

    # 2) tf.keras มาตรฐาน
    try:
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded via tf.keras standard loader")
        return MODEL_KIND, MODEL
    except Exception as e1:
        logger.warning(f"Standard loading failed: {e1}")

    # 3) tf.keras + custom_objects + compile=False + safe_mode=False
    try:
        custom_objects = _create_custom_objects(tf)
        MODEL = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False,  # สำคัญสำหรับ Lambda/function ที่ serialize มา
        )
        logger.info("Model loaded with custom_objects (CompatibleLambda) and safe_mode=False")
        return MODEL_KIND, MODEL
    except Exception as e2:
        logger.warning(f"Loading with custom objects failed: {e2}")

    # 4) SavedModel (สำหรับ path ที่ไม่ใช่ .h5/.keras)
    try:
        if not MODEL_PATH.endswith((".h5", ".keras")):
            MODEL = tf.saved_model.load(MODEL_PATH)
            logger.info("Model loaded as SavedModel")
            return MODEL_KIND, MODEL
    except Exception as e3:
        logger.error(f"SavedModel loading failed: {e3}")

    # 5) TF v1 compatibility (ไม้ตายชิ้นสุดท้าย)
    try:
        import tensorflow.compat.v1 as tf_v1
        tf_v1.disable_eager_execution()
        MODEL = tf_v1.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded with TensorFlow v1 compatibility")
        return MODEL_KIND, MODEL
    except Exception as e4:
        logger.error(f"TensorFlow v1 compatibility loading failed: {e4}")
        raise Exception(f"All loading methods failed. Last error: {e4}")


# ===== Pre/Post-process & Predict =====
def preprocess_image(img_array, target_size=(224, 224)):
    """
    img_array: RGB np.array (H,W,3), dtype uint8
    return: float32 (1,H,W,3) scaled 0-1
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
    if logits.ndim == 1:
        vec = logits
    else:
        vec = logits[0]
    exps = np.exp(vec - np.max(vec))
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
            x_t = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
            logits = MODEL(x_t).cpu().numpy()
    else:
        # Keras model ปกติ
        if hasattr(MODEL, "predict"):
            logits = MODEL.predict(x, verbose=0)
        elif hasattr(MODEL, "__call__"):
            # SavedModel / callable graph
            try:
                logits = MODEL(x, training=False).numpy()
            except Exception:
                logits = MODEL(x).numpy()
        else:
            raise ValueError("Unknown model type")

    return postprocess_logits(logits)
