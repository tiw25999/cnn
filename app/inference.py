# app/inference.py
import os
import numpy as np
import logging

# ----- logging -----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ใช้ TF backend เสมอ
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

MODEL_PATH = os.getenv("MODEL_PATH", "models/EfficientNetV2B0_Original_best.keras")
MODEL_KIND = None   # "torch" หรือ "tf"
MODEL = None

# ชื่อคลาส (ถ้ามี) ใส่มาใน env เป็น "a,b,c"
CLASS_NAMES = os.getenv("CLASS_NAMES", "")
CLASS_NAMES = [c.strip() for c in CLASS_NAMES.split(",")] if CLASS_NAMES else None


# -------- lazy import ----------
def _try_import_torch():
    try:
        import torch  # noqa
        return torch
    except Exception:
        return None


def _try_import_tf():
    try:
        import tensorflow as tf  # noqa
        return tf
    except Exception:
        return None


# ----------- patch / custom objects -------------
def _patch_tf_keras_imports():
    """Mock โมดูล tf_keras ภายในที่มักหายไปตอน deserialize."""
    import sys
    import types
    targets = [
        'tf_keras.src.engine.functional',
        'tf_keras.src.engine.base_layer',
        'tf_keras.src.engine.input_layer',
        'tf_keras.src.engine.training',
        'tf_keras.src.engine.network',
        'tf_keras.src.engine.sequential',
    ]
    for name in targets:
        if name in sys.modules:
            continue
        try:
            import tensorflow as tf
            m = types.ModuleType(name)
            # ใส่คลาสพอประมาณให้ import ผ่าน
            if 'functional' in name:
                m.Functional = tf.keras.Model
            if 'base_layer' in name:
                m.Layer = tf.keras.layers.Layer
            if 'input_layer' in name:
                m.InputLayer = tf.keras.layers.InputLayer
            if 'training' in name:
                m.Model = tf.keras.Model
            if 'network' in name:
                m.Network = tf.keras.Model
            if 'sequential' in name:
                m.Sequential = tf.keras.Sequential
            sys.modules[name] = m
            logger.info(f"Created mock module: {name}")
        except Exception as e:
            logger.warning(f"Could not create mock module {name}: {e}")


def _build_custom_objects(tf):
    """
    คืน dict ของ custom_objects และนิยาม CompatibleLambda
    ที่ 'กลืน' kwargs แปลกๆ ของ Lambda จากโมเดลเก่า
    """
    class CompatibleLambda(tf.keras.layers.Lambda):
        def __init__(self, function=None, **kwargs):
            # กรองคีย์ที่ Keras3/TF2.16 ไม่รู้จักออก
            kwargs.pop('function_type', None)
            kwargs.pop('module', None)
            kwargs.pop('output_shape_type', None)
            kwargs.pop('output_shape_module', None)
            # NOTE: บางไฟล์มี 'function': [<bytes>, None, None] ซึ่ง Keras จะ handle เอง
            super().__init__(function=function, **kwargs)

    # แมปของชั้น/โมเดลยอดนิยม + แทน Lambda ด้วย CompatibleLambda
    custom = {
        'tf_keras': tf.keras,
        'keras': tf.keras,
        'Functional': tf.keras.Model,
        'Model': tf.keras.Model,
        'Sequential': tf.keras.Sequential,
        'Input': tf.keras.Input,

        # layers หลัก ๆ
        'Dense': tf.keras.layers.Dense,
        'Conv2D': tf.keras.layers.Conv2D,
        'MaxPooling2D': tf.keras.layers.MaxPooling2D,
        'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
        'GlobalMaxPooling2D': tf.keras.layers.GlobalMaxPooling2D,
        'AveragePooling2D': tf.keras.layers.AveragePooling2D,
        'Dropout': tf.keras.layers.Dropout,
        'BatchNormalization': tf.keras.layers.BatchNormalization,
        'ReLU': tf.keras.layers.ReLU,
        'Softmax': tf.keras.layers.Softmax,
        'Activation': tf.keras.layers.Activation,
        'Flatten': tf.keras.layers.Flatten,
        'Reshape': tf.keras.layers.Reshape,
        'Add': tf.keras.layers.Add,
        'Concatenate': tf.keras.layers.Concatenate,
        'Multiply': tf.keras.layers.Multiply,
        'ZeroPadding2D': tf.keras.layers.ZeroPadding2D,

        # จุดสำคัญ
        'Lambda': CompatibleLambda,
    }

    # ให้ deserializer มองเห็น functional จาก tf_keras (ป้องกัน error functional)  :contentReference[oaicite:4]{index=4}
    try:
        import sys, types
        if 'tf_keras.src.engine.functional' not in sys.modules:
            fmod = types.ModuleType('tf_keras.src.engine.functional')
            fmod.Functional = tf.keras.Model
            sys.modules['tf_keras.src.engine.functional'] = fmod
        custom['tf_keras.src.engine.functional'] = sys.modules['tf_keras.src.engine.functional']
        custom['tf_keras.src.engine.functional.Functional'] = tf.keras.Model
    except Exception as e:
        logger.warning(f"Could not create tf_keras.src.engine.functional mock: {e}")

    # ใส่ชื่อโมเดล/พรีโพรเซสยอดนิยม (กันไว้)
    try:
        custom.update({
            'EfficientNetV2B0': tf.keras.applications.EfficientNetV2B0,
            'EfficientNetV2B1': tf.keras.applications.EfficientNetV2B1,
            'EfficientNetV2B2': tf.keras.applications.EfficientNetV2B2,
            'EfficientNetV2B3': tf.keras.applications.EfficientNetV2B3,
            'EfficientNetV2S': tf.keras.applications.EfficientNetV2S,
            'EfficientNetV2M': tf.keras.applications.EfficientNetV2M,
            'EfficientNetV2L': tf.keras.applications.EfficientNetV2L,
            'EfficientNetB0': tf.keras.applications.EfficientNetB0,
            'EfficientNetB1': tf.keras.applications.EfficientNetB1,
            'EfficientNetB2': tf.keras.applications.EfficientNetB2,
            'EfficientNetB3': tf.keras.applications.EfficientNetB3,
            'EfficientNetB4': tf.keras.applications.EfficientNetB4,
            'EfficientNetB5': tf.keras.applications.EfficientNetB5,
            'EfficientNetB6': tf.keras.applications.EfficientNetB6,
            'EfficientNetB7': tf.keras.applications.EfficientNetB7,
            # อื่น ๆ เผื่อมี
            'ResNet50': tf.keras.applications.ResNet50,
            'ResNet101': tf.keras.applications.ResNet101,
            'ResNet152': tf.keras.applications.ResNet152,
            'ResNet50V2': tf.keras.applications.ResNet50V2,
            'ResNet101V2': tf.keras.applications.ResNet101V2,
            'ResNet152V2': tf.keras.applications.ResNet152V2,
            'VGG16': tf.keras.applications.VGG16,
            'VGG19': tf.keras.applications.VGG19,
            'MobileNet': tf.keras.applications.MobileNet,
            'MobileNetV2': tf.keras.applications.MobileNetV2,
            'MobileNetV3Small': tf.keras.applications.MobileNetV3Small,
            'MobileNetV3Large': tf.keras.applications.MobileNetV3Large,
            'DenseNet121': tf.keras.applications.DenseNet121,
            'DenseNet169': tf.keras.applications.DenseNet169,
            'DenseNet201': tf.keras.applications.DenseNet201,
            'InceptionV3': tf.keras.applications.InceptionV3,
            'InceptionResNetV2': tf.keras.applications.InceptionResNetV2,
            'Xception': tf.keras.applications.Xception,
            'NASNetMobile': tf.keras.applications.NASNetMobile,
            'NASNetLarge': tf.keras.applications.NASNetLarge,
        })
    except Exception:
        pass

    # พรีโพรเซสของ EfficientNetV2 (ชื่อ "preprocess_input" โผล่ใน config log) :contentReference[oaicite:5]{index=5}
    try:
        custom.update({
            'preprocess_input': tf.keras.applications.efficientnet_v2.preprocess_input,
            'decode_predictions': tf.keras.applications.efficientnet_v2.decode_predictions,
        })
    except Exception:
        pass

    return custom


def detect_model_kind(path: str):
    p = path.lower()
    if p.endswith((".pt", ".pth")):
        return "torch"
    if p.endswith((".h5", ".keras")) or "savedmodel" in p:
        return "tf"
    # เดา: ถ้ามี torch ติดตั้ง เอา torch ก่อน
    if _try_import_torch() is not None:
        return "torch"
    return "tf"


# -------------- load ----------------
def load_model():
    global MODEL_KIND, MODEL
    if MODEL is not None:
        return MODEL_KIND, MODEL

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    MODEL_KIND = detect_model_kind(MODEL_PATH)

    if MODEL_KIND == "torch":
        torch = _try_import_torch()
        if torch is None:
            raise RuntimeError("Torch not available but model looks like torch.")
        MODEL = torch.jit.load(MODEL_PATH) if MODEL_PATH.endswith(".pt") else torch.load(MODEL_PATH, map_location="cpu")
        MODEL.eval()
        return MODEL_KIND, MODEL

    # TF / Keras
    tf = _try_import_tf()
    if tf is None:
        raise RuntimeError("TensorFlow not available.")

    _patch_tf_keras_imports()
    custom_objects = _build_custom_objects(tf)

    # 1) ลองโหลดปกติ (บางไฟล์ผ่าน)
    try:
        MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("Model loaded (standard).")
        return MODEL_KIND, MODEL
    except Exception as e1:
        logger.warning(f"Standard loading failed: {e1}")

    # 2) บังคับ custom_objects + safe_mode=False เพื่อให้ CompatibleLambda ทำงานจริง
    try:
        from contextlib import nullcontext
        scope = getattr(tf.keras.utils, "custom_object_scope", nullcontext)
        with scope(custom_objects):
            MODEL = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects=custom_objects,
                compile=False,
                safe_mode=False,  # สำคัญกับ .keras (Keras v3)
            )
        logger.info("Model loaded with custom objects (Lambda tolerant) + safe_mode=False.")
        return MODEL_KIND, MODEL
    except Exception as e2:
        logger.warning(f"Loading with custom objects failed: {e2}")

    # 3) ถ้าเป็นไดเรกทอรี/ SavedModel
    try:
        if not MODEL_PATH.endswith((".h5", ".keras")):
            MODEL = tf.saved_model.load(MODEL_PATH)
            logger.info("Model loaded as SavedModel.")
            return MODEL_KIND, MODEL
    except Exception as e3:
        logger.error(f"SavedModel loading failed: {e3}")

    # 4) ความพยายามสุดท้าย: TF v1 compat
    try:
        import tensorflow.compat.v1 as tf_v1
        tf_v1.disable_eager_execution()
        MODEL = tf_v1.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded with TF v1 compatibility.")
        return MODEL_KIND, MODEL
    except Exception as e4:
        logger.error(f"TF v1 compatibility loading failed: {e4}")
        raise Exception(f"All loading methods failed. Last error: {e4}")


# -------------- preprocess / predict --------------
def preprocess_image(img_array, target_size=(224, 224)):
    """
    img_array: RGB uint8 (H,W,3) -> float32 (1,H,W,3) scaled [0,1]
    """
    import cv2
    h, w = target_size
    img = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def postprocess_logits(logits: np.ndarray):
    """
    รองรับทั้ง binary/multi-class
    return: dict(probabilities=[...], top_idx, top_label, top_conf)
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
    """
    ให้ main.py เรียกใช้ได้ตรง ๆ
    """
    global MODEL_KIND, MODEL
    if MODEL is None:
        load_model()

    x = preprocess_image(img_array, target_size=(224, 224))

    if MODEL_KIND == "torch":
        import torch
        with torch.no_grad():
            x_t = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
            logits = MODEL(x_t).cpu().numpy()
    else:
        # Keras/SavedModel รูปแบบต่าง ๆ
        if hasattr(MODEL, "predict"):
            logits = MODEL.predict(x, verbose=0)
        elif hasattr(MODEL, "__call__"):
            try:
                logits = MODEL(x, training=False).numpy()
            except Exception:
                logits = MODEL(x).numpy()
        else:
            raise ValueError("Unknown model type for prediction.")

    return postprocess_logits(logits)
