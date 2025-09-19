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

def _create_comprehensive_custom_objects(tf):
    """Create comprehensive custom objects for model loading compatibility"""
    custom_objects = {
        'tf_keras': tf.keras,
        'keras': tf.keras,
        'Functional': tf.keras.Model,
        'Model': tf.keras.Model,
        'Sequential': tf.keras.Sequential,
        'Input': tf.keras.Input,
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
        'Lambda': tf.keras.layers.Lambda,
        'Add': tf.keras.layers.Add,
        'Concatenate': tf.keras.layers.Concatenate,
        'Multiply': tf.keras.layers.Multiply,
        'AveragePooling2D': tf.keras.layers.AveragePooling2D,
        'GlobalMaxPooling2D': tf.keras.layers.GlobalMaxPooling2D,
        'ZeroPadding2D': tf.keras.layers.ZeroPadding2D,
    }
    
    # Add specific handlers for tf_keras.src.engine.functional error
    try:
        # Try to import and add the missing module
        import sys
        if 'tf_keras.src.engine.functional' not in sys.modules:
            # Create a mock module for tf_keras.src.engine.functional
            import types
            functional_module = types.ModuleType('tf_keras.src.engine.functional')
            functional_module.Functional = tf.keras.Model
            sys.modules['tf_keras.src.engine.functional'] = functional_module
        
        custom_objects['tf_keras.src.engine.functional'] = sys.modules['tf_keras.src.engine.functional']
        custom_objects['tf_keras.src.engine.functional.Functional'] = tf.keras.Model
    except Exception as e:
        logger.warning(f"Could not create tf_keras.src.engine.functional mock: {e}")
        pass
    
    # Add EfficientNet models
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
        })
    except:
        pass
    
    # Add other common models
    try:
        custom_objects.update({
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
    except:
        pass
    
    # Add preprocessing functions
    try:
        custom_objects.update({
            'preprocess_input': tf.keras.applications.efficientnet_v2.preprocess_input,
            'decode_predictions': tf.keras.applications.efficientnet_v2.decode_predictions,
        })
    except:
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

def _patch_tf_keras_imports():
    """Patch missing tf_keras imports that cause deserialization errors"""
    import sys
    import types
    
    # Create mock modules for missing tf_keras imports
    mock_modules = [
        'tf_keras.src.engine.functional',
        'tf_keras.src.engine.base_layer',
        'tf_keras.src.engine.input_layer',
        'tf_keras.src.engine.training',
        'tf_keras.src.engine.network',
        'tf_keras.src.engine.sequential',
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
                import tensorflow as tf
                mock_module = types.ModuleType(module_name)
                
                # Add common classes that might be referenced
                if 'functional' in module_name:
                    mock_module.Functional = tf.keras.Model
                if 'base_layer' in module_name:
                    mock_module.Layer = tf.keras.layers.Layer
                if 'input_layer' in module_name:
                    mock_module.InputLayer = tf.keras.layers.InputLayer
                if 'training' in module_name:
                    mock_module.Model = tf.keras.Model
                if 'network' in module_name:
                    mock_module.Network = tf.keras.Model
                if 'sequential' in module_name:
                    mock_module.Sequential = tf.keras.Sequential
                
                sys.modules[module_name] = mock_module
                logger.info(f"Created mock module: {module_name}")
            except Exception as e:
                logger.warning(f"Could not create mock module {module_name}: {e}")

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
        
        # Patch missing imports before loading
        _patch_tf_keras_imports()
        
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
                custom_objects = _create_comprehensive_custom_objects(tf)
                
                if MODEL_PATH.endswith((".h5", ".keras")):
                    MODEL = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
                else:
                    MODEL = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
                logger.info("Model loaded successfully with custom objects and compile=False")
            except Exception as e2:
                logger.warning(f"Loading with custom objects failed: {e2}")
                try:
                    # Third try: Use tf.saved_model.load for SavedModel format
                    if not MODEL_PATH.endswith((".h5", ".keras")):
                        try:
                            MODEL = tf.saved_model.load(MODEL_PATH)
                            logger.info("Model loaded successfully as SavedModel")
                        except Exception as e3:
                            logger.error(f"SavedModel loading failed: {e3}")
                            # Fourth try: Try loading with different TensorFlow versions
                            try:
                                # Try with older TensorFlow compatibility
                                import tensorflow.compat.v1 as tf_v1
                                tf_v1.disable_eager_execution()
                                MODEL = tf_v1.keras.models.load_model(MODEL_PATH)
                                logger.info("Model loaded successfully with TensorFlow v1 compatibility")
                            except Exception as e4:
                                logger.error(f"TensorFlow v1 compatibility loading failed: {e4}")
                                raise Exception(f"All loading methods failed. Last error: {e4}")
                    else:
                        # For .keras/.h5 files, try a different approach
                        try:
                            # Try loading with safe_mode=False
                            MODEL = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
                            logger.info("Model loaded successfully with safe_mode=False")
                        except Exception as e5:
                            logger.error(f"Loading with safe_mode=False failed: {e5}")
                            raise Exception(f"All loading methods failed. Last error: {e5}")
                except Exception as e3:
                    logger.error(f"Third loading attempt failed: {e3}")
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
