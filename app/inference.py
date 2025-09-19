# app/inference.py
import os
import sys
import types
import numpy as np
import logging

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ใช้ TensorFlow backend กับ Keras 3 เสมอ
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

MODEL_PATH = os.getenv("MODEL_PATH", "models/EfficientNetV2B0_Original_best.keras")
MODEL_KIND = None  # "torch" or "tf"
MODEL = None
CLASS_NAMES = os.getenv("CLASS_NAMES", "")
CLASS_NAMES = [c.strip() for c in CLASS_NAMES.split(",")] if CLASS_NAMES else None


# ---------- Lazy imports ----------
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


# ---------- Monkeypatch / Compatibility ----------
def _monkeypatch_lambda(tf, ks=None):
    """
    แพตช์ชั้น Lambda ให้ยอมรับคีย์พิเศษจากโมเดลที่ serialize มาจากเวอร์ชันอื่น
    เช่น function_type, module, output_shape_type, output_shape_module
    ทำทั้งใน tf.keras และ (ถ้ามี) standalone keras
    """
    # สร้างคลาสที่ tolerant ต่อ kwargs พิเศษ
    class PatchedLambda(tf.keras.layers.Lambda):
        def __init__(self, function=None, **kwargs):
            # คีย์ที่เจอใน error:
            # Unrecognized keyword arguments passed to Lambda:
            # {'function_type': 'lambda', 'module': '__main__',
            #  'output_shape_type': 'raw', 'output_shape_module': None}
            kwargs.pop("function_type", None)
            kwargs.pop("module", None)
            kwargs.pop("output_shape_type", None)
            kwargs.pop("output_shape_module", None)
            # บางโมเดลอาจมีคีย์อื่นที่ไม่จำเป็น
            for k in list(kwargs.keys()):
                if k.startswith("_unused") or k.endswith("_module_path"):
                    kwargs.pop(k, None)
            super().__init__(function=function, **kwargs)

    # monkeypatch ที่ตำแหน่งหลักที่ deserializer จะอ้างถึง
    tf.keras.layers.Lambda = PatchedLambda  # type: ignore

    # ถ้ามี standalone keras ให้แพตช์ด้วย
    if ks is not None:
        try:
            ks.layers.Lambda = PatchedLambda  # type: ignore
        except Exception:
            pass

    return PatchedLambda


def _patch_tf_keras_imports():
    """
    โมเดลบางตัวอ้างถึง internal path แบบ tf_keras.src.engine.functional.Functional
    เราสร้าง mock module เพื่อไม่ให้ deserializer ล้มกลางทาง
    """
    try:
        import tensorflow as tf  # ใช้ใน mock class mapping
    except Exception:
        return

    mock_modules = [
        "tf_keras.src.engine.functional",
        "tf_keras.src.engine.base_layer",
        "tf_keras.src.engine.input_layer",
        "tf_keras.src.engine.training",
        "tf_keras.src.engine.network",
        "tf_keras.src.engine.sequential",
    ]

    for module_name in mock_modules:
        if module_name not in sys.modules:
            try:
                m = types.ModuleType(module_name)
                # เติม symbol หลักๆ ให้พอใช้งาน
                if "functional" in module_name:
                    m.Functional = tf.keras.Model
                if "base_layer" in module_name:
                    m.Layer = tf.keras.layers.Layer
                if "input_layer" in module_name:
                    m.InputLayer = tf.keras.layers.InputLayer
                if "training" in module_name:
                    m.Model = tf.keras.Model
                if "network" in module_name:
                    m.Network = tf.keras.Model
                if "sequential" in module_name:
                    m.Sequential = tf.keras.Sequential
                sys.modules[module_name] = m
                logger.info(f"Created mock module: {module_name}")
            except Exception as e:
                logger.warning(f"Could not create mock module {module_name}: {e}")


def _create_comprehensive_custom_objects(tf, PatchedLambda):
    """
    รวม custom_objects ให้ครบ: ชั้นพื้นฐาน, EfficientNet, และ mapping พิเศษ
    """
    custom_objects = {
        # aliases
        "tf_keras": tf.keras,
        "keras": tf.keras,

        # core classes
        "Functional": tf.keras.Model,
        "Model": tf.keras.Model,
        "Sequential": tf.keras.Sequential,
        "Input": tf.keras.Input,

        # layers
        "Dense": tf.keras.layers.Dense,
        "Conv2D": tf.keras.layers.Conv2D,
        "DepthwiseConv2D": getattr(tf.keras.layers, "DepthwiseConv2D", None),
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
        "ZeroPadding2D": tf.keras.layers.ZeroPadding2D,
        "Add": tf.keras.layers.Add,
        "Concatenate": tf.keras.layers.Concatenate,
        "Multiply": tf.keras.layers.Multiply,
        "Lambda": PatchedLambda,  # สำคัญสุด

        # บาง deserializer อาจดูชื่อเต็มของโมดูล
        "tf_keras.src.engine.functional": sys.modules.get("tf_keras.src.engine.functional"),
        "tf_keras.src.engine.functional.Functional": tf.keras.Model,
    }

    # เพิ่ม EfficientNet และโมเดลยอดนิยม (มีบางตัวอาจหายไปในบาง build)
    for name in [
        "EfficientNetV2B0", "EfficientNetV2B1", "EfficientNetV2B2",
        "EfficientNetV2B3", "EfficientNetV2S", "EfficientNetV2M", "EfficientNetV2L",
        "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3",
        "EfficientNetB4", "EfficientNetB5", "EfficientNetB6", "EfficientNetB7",
        "ResNet50", "ResNet101", "ResNet152",
        "ResNet50V2", "ResNet101V2", "ResNet152V2",
        "VGG16", "VGG19",
        "MobileNet", "MobileNetV2", "MobileNetV3Small", "MobileNetV3Large",
        "DenseNet121", "DenseNet169", "DenseNet201",
        "InceptionV3", "InceptionResNetV2",
        "Xception", "NASNetMobile", "NASNetLarge",
    ]:
        try:
            custom_objects[name] = getattr(tf.keras.applications, name)
        except Exception:
            pass

    # preprocessing (ถ้ามี)
    try:
        custom_objects["preprocess_input"] = tf.keras.applications.efficientnet_v2.preprocess_input
    except Exception:
        pass
    try:
        custom_objects["decode_predictions"] = tf.keras.applications.efficientnet_v2.decode_predictions
    except Exception:
        pass

    return custom_objects


def detect_model_kind(path: str):
    p = path.lower()
    if p.endswith((".pt", ".pth")):
        return "torch"
    if p.endswith((".h5", ".keras")) or "savedmodel" in p:
        return "tf"
    # เดา: ถ้ามี torch ก็ถือว่า torch, ไม่งั้น tf
    return "torch" if _try_import_torch() is not None else "tf"


# ---------- Load / Predict ----------
def load_model():
    """
    ลำดับความพยายาม:
    1) standalone keras (Keras 3) + monkeypatch Lambda + safe_mode=False + custom_objects
    2) tf.keras.load_model(...) + monkeypatch + custom_objects + compile=False + safe_mode=False
    3) tf.saved_model.load(...) (กรณีไม่ใช่ .h5/.keras)
    4) tf.compat.v1 keras loader (ฉุกเฉิน)
    """
    global MODEL_KIND, MODEL

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    MODEL_KIND = detect_model_kind(MODEL_PATH)

    if MODEL_KIND == "torch":
        torch = _try_import_torch()
        assert torch is not None, "PyTorch not installed. Add to requirements.txt"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        MODEL = (
            torch.jit.load(MODEL_PATH, map_location=device)
            if MODEL_PATH.endswith(".pt")
            else torch.load(MODEL_PATH, map_location=device)
        )
        MODEL.eval()
        return MODEL_KIND, MODEL

    # ---- TensorFlow / Keras path ----
    tf = _try_import_tf()
    assert tf is not None, "TensorFlow not installed. Add to requirements.txt"

    # แพตช์ internal modules ก่อน
    _patch_tf_keras_imports()

    # พยายามใช้ standalone keras ก่อน (ถ้ามี)
    ks = None
    try:
        import keras as _ks  # Keras 3 (standalone)
        ks = _ks
    except Exception:
        ks = None

    # แพตช์ Lambda ทั้ง tf.keras และ (ถ้ามี) keras
    PatchedLambda = _monkeypatch_lambda(tf, ks)

    # 1) Standalone Keras
    if ks is not None:
        try:
            MODEL = ks.models.load_model(
                MODEL_PATH,
                compile=False,
                safe_mode=False,
                custom_objects={"Lambda": PatchedLambda},
            )
            logger.info("Model loaded via standalone Keras (safe_mode=False, compile=False, PatchedLambda)")
            return MODEL_KIND, MODEL
        except Exception as e:
            logger.warning(f"Standalone Keras load failed: {e}")

    # 2) tf.keras loader (พร้อม custom_objects ครบ)
    try:
        custom_objects = _create_comprehensive_custom_objects(tf, PatchedLambda)
        MODEL = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False,
        )
        logger.info("Model loaded via tf.keras with PatchedLambda and custom_objects (safe_mode=False, compile=False)")
        return MODEL_KIND, MODEL
    except Exception as e2:
        logger.warning(f"tf.keras load_model failed: {e2}")

    # 3) SavedModel loader (ถ้าไม่ใช่ไฟล์ .h5/.keras)
    try:
        if not MODEL_PATH.endswith((".h5", ".keras")):
            MODEL = tf.saved_model.load(MODEL_PATH)
            logger.info("Model loaded as SavedModel via tf.saved_model.load")
            return MODEL_KIND, MODEL
    except Exception as e3:
        logger.warning(f"tf.saved_model.load failed: {e3}")

    # 4) TF v1 compat (ทางเลือกสุดท้าย)
    try:
        import tensorflow.compat.v1 as tf_v1
        tf_v1.disable_eager_execution()
        MODEL = tf_v1.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded with TensorFlow v1 compatibility loader")
        return MODEL_KIND, MODEL
    except Exception as e4:
        logger.error(f"All loading methods failed. Last error: {e4}")
        raise RuntimeError(f"All loading methods failed. Last error: {e4}")


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


def _softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def postprocess_logits(logits: np.ndarray):
    """
    รองรับทั้ง binary/multi-class
    return: dict(probabilities=[...], top_idx=int, top_label=str, top_conf=float)
    """
    logits = logits[0] if logits.ndim > 1 else logits
    probs = _softmax(logits)
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
            x_t = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()  # N,H,W,3 -> N,3,H,W
            logits = MODEL(x_t).cpu().numpy()
    else:
        # Keras model สากล
        if hasattr(MODEL, "predict"):
            logits = MODEL.predict(x, verbose=0)
        elif hasattr(MODEL, "__call__"):
            # SavedModel หรือ tf.function
            try:
                y = MODEL(x, training=False)
            except TypeError:
                y = MODEL(x)
            logits = y.numpy() if hasattr(y, "numpy") else np.array(y)
        else:
            raise ValueError("Unknown model type")

    return postprocess_logits(logits)
