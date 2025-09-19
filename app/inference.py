import os
import sys
import types
import numpy as np
import logging

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure standalone Keras uses TF backend if present
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

MODEL_PATH = os.getenv("MODEL_PATH", "models/EfficientNetV2B0_Original_best.keras")
MODEL_KIND = None    # "torch" or "tf"
MODEL = None
CLASS_NAMES = os.getenv("CLASS_NAMES", "")  # optional: "cat,dog,car"
CLASS_NAMES = [c.strip() for c in CLASS_NAMES.split(",")] if CLASS_NAMES else None

# -----------------------
# Lazy imports
# -----------------------
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

# -----------------------
# Monkey patches
# -----------------------
_LEGACY_LAMBDA_KEYS = {
    # legacy / internal keys that appear in serialized Lambda in different TF/Keras versions
    "function_type", "module",
    "output_shape_type", "output_shape_module",
    # some serializers also push these inside kwargs
    "output_shape", "mask", "arguments"
}

def _patch_lambda_init():
    """
    Monkey-patch keras.layers.Lambda.__init__ (ทั้งใน tf.keras และ keras)
    ให้กรอง kwargs แปลกๆ ที่มากับโมเดลเวอร์ชันเก่า/ต่างเวอร์ชันออกก่อน
    """
    try:
        import tensorflow as tf
        Lambda_tf = tf.keras.layers.Lambda
    except Exception:
        Lambda_tf = None

    try:
        import keras as ks
        Lambda_ks = ks.layers.Lambda
    except Exception:
        Lambda_ks = None

    def _wrap_lambda_init(LambdaClass):
        if LambdaClass is None:
            return
        if getattr(LambdaClass, "_patched_for_legacy_kwargs", False):
            return  # already patched

        original_init = LambdaClass.__init__

        def new_init(self, *args, **kwargs):
            # กรอง legacy kwargs ออกให้หมดก่อนเข้า __init__ จริง
            # หมายเหตุ: อย่าโยน 'function' ให้ super().__init__ ของ Base Layer เด็ดขาด
            # ปล่อยให้ Lambda ของ Keras จัดการเอง
            clean_kwargs = dict(kwargs)
            for k in list(clean_kwargs.keys()):
                if k in _LEGACY_LAMBDA_KEYS:
                    # เก็บค่าไว้ในตัวแปร local เผื่อ Lambda ต้องใช้ต่อไป
                    # แต่ไม่ส่งต่อเป็น kwargs ไปยัง Base Layer
                    pass
            # เอา key แปลกๆ ออก
            for k in _LEGACY_LAMBDA_KEYS:
                clean_kwargs.pop(k, None)
            # เรียกของเดิมด้วย kwargs ที่สะอาด
            return original_init(self, *args, **clean_kwargs)

        LambdaClass.__init__ = new_init  # type: ignore
        LambdaClass._patched_for_legacy_kwargs = True  # type: ignore
        logger.info(f"Patched {LambdaClass.__module__}.{LambdaClass.__name__}.__init__ to drop legacy kwargs")

    _wrap_lambda_init(Lambda_tf)
    _wrap_lambda_init(Lambda_ks)

def _patch_tf_keras_imports():
    """
    สร้าง mock modules สำหรับเส้นทาง import ภายใน tf_keras.* ที่บางครั้งหายไป
    เพื่อให้ deserializer หาเจอ (เช่น tf_keras.src.engine.functional)
    """
    try:
        import tensorflow as tf
    except Exception:
        return

    mock_modules = [
        'tf_keras.src.engine.functional',
        'tf_keras.src.engine.base_layer',
        'tf_keras.src.engine.input_layer',
        'tf_keras.src.engine.training',
        'tf_keras.src.engine.network',
        'tf_keras.src.engine.sequential',
    ]
    for module_name in mock_modules:
        if module_name not in sys.modules:
            try:
                m = types.ModuleType(module_name)
                if 'functional' in module_name:
                    m.Functional = tf.keras.Model
                if 'base_layer' in module_name:
                    m.Layer = tf.keras.layers.Layer
                if 'input_layer' in module_name:
                    m.InputLayer = tf.keras.layers.InputLayer
                if 'training' in module_name:
                    m.Model = tf.keras.Model
                if 'network' in module_name:
                    m.Network = tf.keras.Model
                if 'sequential' in module_name:
                    m.Sequential = tf.keras.Sequential
                sys.modules[module_name] = m
                logger.info(f"Created mock module: {module_name}")
            except Exception as e:
                logger.warning(f"Could not create mock module {module_name}: {e}")

def _create_comprehensive_custom_objects(tf):
    """
    custom_objects ครอบจักรวาล + ใส่ CompatibleLambda ไว้เป็น fallback อีกชั้น
    (ถึงแม้เราจะ monkey-patch __init__ ไปแล้ว)
    """
    class CompatibleLambda(tf.keras.layers.Lambda):
        def __init__(self, function=None, output_shape=None, mask=None, arguments=None, **kwargs):
            # ตัด legacy kwargs ออกให้หมดก่อน
            kwargs.pop('function_type', None)
            kwargs.pop('module', None)
            kwargs.pop('output_shape_type', None)
            kwargs.pop('output_shape_module', None)
            # เรียก __init__ ของ Lambda จริง โดยส่งเฉพาะพารามิเตอร์ที่มันรู้จัก
            super().__init__(function=function, output_shape=output_shape, mask=mask, arguments=arguments, **kwargs)

    custom_objects = {
        'Lambda': CompatibleLambda,
        'tf_keras': tf.keras,
        'keras': tf.keras,
        'Functional': tf.keras.Model,
        'Model': tf.keras.Model,
        'Sequential': tf.keras.Sequential,
        'Input': tf.keras.Input,
        # common layers
        'Dense': tf.keras.layers.Dense,
        'Conv2D': tf.keras.layers.Conv2D,
        'MaxPooling2D': tf.keras.layers.MaxPooling2D,
        'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
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
        'AveragePooling2D': tf.keras.layers.AveragePooling2D,
        'GlobalMaxPooling2D': tf.keras.layers.GlobalMaxPooling2D,
        'ZeroPadding2D': tf.keras.layers.ZeroPadding2D,
        'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D,
        'Normalization': tf.keras.layers.Normalization,
        'Rescaling': tf.keras.layers.Rescaling,
    }

    # EfficientNet family + preprocess
    try:
        custom_objects.update({
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
            'preprocess_input': tf.keras.applications.efficientnet_v2.preprocess_input,
            'decode_predictions': tf.keras.applications.efficientnet_v2.decode_predictions,
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
    # เดา: ทดลองโหลดตามลำดับ
    if _try_import_torch() is not None:
        return "torch"
    return "tf"

# -----------------------
# Loader
# -----------------------
def load_model():
    """
    กลยุทธ์โหลดโมเดล:
    1) monkey-patch Lambda.__init__ (กันทุกทาง)
    2) (ถ้ามี) ลอง standalone Keras (safe_mode=False, compile=False)
    3) tf.keras.load_model ด้วย/ไม่ด้วย custom_objects + safe_mode=False, compile=False
    4) สำหรับ SavedModel: tf.saved_model.load
    5) v1 compat (กรณีพิเศษ)
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

    # TF / Keras
    tf = _try_import_tf()
    assert tf is not None, "TensorFlow not installed. Add to requirements.txt"

    # 1) แพตช์ Lambda + mock tf_keras.*
    _patch_lambda_init()
    _patch_tf_keras_imports()

    # 2) ลอง standalone Keras (Keras 3)
    try:
        import keras as ks  # standalone Keras 3
        _patch_lambda_init()  # ย้ำอีกครั้งใน namespace ของ keras
        try:
            MODEL = ks.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
            logger.info("Model loaded via standalone Keras (safe_mode=False, compile=False)")
            return MODEL_KIND, MODEL
        except Exception as e_keras:
            logger.warning(f"Standalone Keras load failed: {e_keras}")
    except Exception:
        pass

    # 3) ลอง tf.keras หลายแบบ
    custom_objects = _create_comprehensive_custom_objects(tf)

    # 3.1 มาตรฐาน (อาศัย monkey patch)
    try:
        MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        logger.info("Model loaded via tf.keras (safe_mode=False, compile=False)")
        return MODEL_KIND, MODEL
    except Exception as e1:
        logger.warning(f"Standard loading failed: {e1}")

    # 3.2 ใส่ custom_objects
    try:
        MODEL = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False
        )
        logger.info("Model loaded with custom_objects (Lambda tolerant)")
        return MODEL_KIND, MODEL
    except Exception as e2:
        logger.warning(f"Loading with custom objects failed: {e2}")

    # 4) SavedModel (กรณีไม่ใช่ .h5/.keras)
    try:
        if not MODEL_PATH.endswith((".h5", ".keras")):
            MODEL = tf.saved_model.load(MODEL_PATH)
            logger.info("Model loaded successfully as SavedModel")
            return MODEL_KIND, MODEL
    except Exception as e3:
        logger.error(f"SavedModel loading failed: {e3}")

    # 5) v1 compat
    try:
        import tensorflow.compat.v1 as tf_v1
        tf_v1.disable_eager_execution()
        _patch_lambda_init()  # ย้ำ
        MODEL = tf_v1.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("Model loaded with TF v1 compatibility")
        return MODEL_KIND, MODEL
    except Exception as e4:
        logger.error(f"TensorFlow v1 compatibility loading failed: {e4}")
        raise Exception(f"All loading methods failed. Last error: {e4}")

def _infer_target_size_from_model(default=(224, 224)):
    """
    พยายามดึง input size จากโมเดล ถ้าไม่ได้ใช้ค่า default
    """
    try:
        if MODEL is None:
            return default
        # Keras Model
        if hasattr(MODEL, "inputs") and MODEL.inputs:
            ish = getattr(MODEL.inputs[0].shape, "as_list", lambda: list(MODEL.inputs[0].shape))()
            # ish: [None, H, W, C] หรือ [None, C, H, W]
            if len(ish) == 4:
                # assume channels_last
                H, W = ish[1], ish[2]
                if isinstance(H, int) and isinstance(W, int) and H > 0 and W > 0:
                    return (H, W)
        # SavedModel (callable)
        if hasattr(MODEL, "signatures") and "serving_default" in MODEL.signatures:
            sig = MODEL.signatures["serving_default"]
            for t in sig.structured_input_signature[1].values():
                # คาดว่าเป็น [None, H, W, C]
                shap = t.shape
                if len(shap) == 4 and shap[1] and shap[2]:
                    return (int(shap[1]), int(shap[2]))
    except Exception:
        pass
    return default

# -----------------------
# Pre/Post process & Predict
# -----------------------
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
    global MODEL, MODEL_KIND
    if MODEL is None:
        load_model()

    target_size = _infer_target_size_from_model(default=(224, 224))
    x = preprocess_image(img_array, target_size=target_size)

    if MODEL_KIND == "torch":
        import torch
        with torch.no_grad():
            x_t = torch.from_numpy(x).permute(0,3,1,2).contiguous()  # N,H,W,3 -> N,3,H,W
            logits = MODEL(x_t).cpu().numpy()
    else:
        # Handle different TensorFlow model types
        if hasattr(MODEL, 'predict'):
            logits = MODEL.predict(x, verbose=0)
        elif hasattr(MODEL, '__call__'):
            try:
                logits = MODEL(x, training=False).numpy()
            except Exception:
                logits = MODEL(x).numpy()
        else:
            raise ValueError("Unknown model type")

    return postprocess_logits(logits)
