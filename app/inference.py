# app/inference.py
import os
import sys
import types
import logging
import base64
from functools import lru_cache
from typing import Dict, Any, Optional

# Ensure standalone Keras (if used) selects TensorFlow backend
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf
import cv2
from typing import List, Tuple, Dict

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
    "/app/models/EfficientNetB0_Augmented_best.keras",
    "models/EfficientNetB0_Augmented_best.keras",
    "EfficientNetB0_Augmented_best.keras",
]

_MODEL_PRIORITY = [
    "EfficientNetB0_Augmented",
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

# ---------- 4) การจัดการโมเดลหลายตัว ----------
# ตัวแปรเก็บโมเดลปัจจุบัน
_current_model = None
_current_model_path = "models/MobileNetV2_Augmented_best.keras"  # Default to MobileNetV2

def get_available_models() -> List[str]:
    """คืนรายการโมเดลที่มีอยู่ในโฟลเดอร์ models"""
    models = []
    models_dir = "models"
    if os.path.isdir(models_dir):
        for fname in os.listdir(models_dir):
            if fname.lower().endswith(".keras"):
                models.append(fname)
    return sorted(models)

def set_model_path(model_name: str) -> str:
    """ตั้งค่า path ของโมเดลที่ต้องการใช้"""
    global _current_model_path
    
    # ตรวจสอบว่าโมเดลมีอยู่จริงหรือไม่
    models_dir = "models"
    model_path = os.path.join(models_dir, model_name)
    
    if not os.path.exists(model_path):
        # ลองหาใน path อื่น ๆ
        for base_path in ["", "/app/models", "models"]:
            test_path = os.path.join(base_path, model_name)
            if os.path.exists(test_path):
                model_path = test_path
                break
        else:
            raise FileNotFoundError(f"Model not found: {model_name}")
    
    _current_model_path = model_path
    logger.info(f"Model path set to: {_current_model_path}")
    return _current_model_path

def get_current_model_path() -> str:
    """คืน path ของโมเดลปัจจุบัน"""
    global _current_model_path
    if _current_model_path is None:
        _current_model_path = _resolve_model_path()
    return _current_model_path

# ---------- 5) โหลดโมเดลด้วย fallback หลายแบบ ----------
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

def load_model() -> Any:
    global _current_model, _current_model_path
    
    # ใช้โมเดลปัจจุบันถ้ามีอยู่แล้ว
    if _current_model is not None:
        logger.info("Using cached model")
        return _current_model
    
    # ใช้ path ปัจจุบันหรือหาใหม่
    if _current_model_path is None:
        _current_model_path = _resolve_model_path()
    
    path = _current_model_path
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
            _current_model = model  # เก็บโมเดลไว้ใน cache
            return model
        except Exception as e:
            logger.warning("%s load failed: %s", name, e)
            last_err = e

    raise RuntimeError(f"All loading methods failed. Last error: {last_err}")

def switch_model(model_name: str) -> Any:
    """เปลี่ยนโมเดลเป็นโมเดลที่ระบุ"""
    global _current_model, _current_model_path
    
    try:
        # ตั้งค่า path ใหม่
        new_path = set_model_path(model_name)
        
        # ล้างโมเดลเก่า
        _current_model = None
        
        # โหลดโมเดลใหม่
        model = load_model()
        
        logger.info(f"Successfully switched to model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to switch model to {model_name}: {e}")
        raise e

def clear_model_cache():
    """ล้าง cache ของโมเดลเพื่อให้โหลดโมเดลใหม่"""
    global _current_model
    try:
        # ล้างโมเดลปัจจุบัน
        _current_model = None
        logger.info("Model cache cleared successfully")
        return True
    except Exception as e:
        logger.warning(f"Failed to clear model cache: {e}")
        return False

def reload_model() -> Any:
    """รีโหลดโมเดลใหม่"""
    try:
        clear_model_cache()
        model = load_model()
        logger.info("Model reloaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise e

def get_current_model_info() -> Dict[str, Any]:
    """คืนข้อมูลโมเดลที่ใช้อยู่ปัจจุบัน"""
    try:
        path = get_current_model_path()
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
def detect_objects_with_bounding_boxes(img: np.ndarray, model) -> List[Dict[str, Any]]:
    """
    ตรวจจับวัตถุในภาพและคืนค่า bounding boxes
    ลำดับความสำคัญ:
      1) Vertical projection จากแถบบรรทัดบน (12–62% ของความสูงภาพ)
      2) หากไม่พบกล่องจาก projection ให้ fallback เป็น sliding window เดิม
    """
    try:
        proj_dets, process_steps = detect_by_vertical_projection(img, model, return_steps=True)
        if proj_dets:
            logger.info("Projection-based detection found %d boxes", len(proj_dets))
            return proj_dets, process_steps
        else:
            logger.info("Projection-based detection found 0 boxes, fallback to sliding window")
            process_steps = {}  # ไม่มี process steps สำหรับ sliding window
    except Exception as e:
        logger.warning("Projection-based detection failed: %s; fallback to sliding window", e)
        process_steps = {}  # ไม่มี process steps สำหรับ sliding window
    # resize ภาพถ้าใหญ่เกินไปเพื่อป้องกันการค้าง
    original_h, original_w = img.shape[:2]
    h, w = original_h, original_w
    
    # ปรับขนาดภาพให้เหมาะสมกับขนาดภาพ
    if original_h * original_w > 500000:  # ภาพใหญ่กว่า 500K pixels
        max_size = 800  # ลดขนาดลง
        logger.info(f"Large image detected: {original_h}x{original_w} ({original_h*original_w:,} pixels), resizing to {max_size}x{max_size}")
    elif original_h * original_w > 200000:  # ภาพขนาดกลาง
        max_size = 1024
        logger.info(f"Medium image detected: {original_h}x{original_w} ({original_h*original_w:,} pixels), resizing to {max_size}x{max_size}")
    else:
        max_size = 1200  # ภาพเล็กสามารถใช้ขนาดใหญ่ได้
    
    if h > max_size or w > max_size:
        # คำนวณสัดส่วนการ resize
        scale = min(max_size / h, max_size / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # resize ภาพ
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        logger.info(f"Resized image from {original_h}x{original_w} to {new_h}x{new_w} for sliding window processing")
    
    window_size = 192
    stride = 96  # 50% overlap
    
    detections = []
    confidence_threshold = 0.3  # threshold สำหรับการตรวจจับ
    
    # คำนวณจำนวน windows ที่จะประมวลผล
    total_windows = ((h - window_size) // stride + 1) * ((w - window_size) // stride + 1)
    logger.info(f"Initial calculation: {total_windows} windows for image {h}x{w} with stride {stride}")
    
    # ปรับ stride ตามขนาดภาพเพื่อลดการประมวลผล
    if total_windows > 1000:  # ภาพใหญ่มาก
        stride = max(256, stride * 4)
        total_windows = ((h - window_size) // stride + 1) * ((w - window_size) // stride + 1)
        logger.info(f"Very large image, increased stride to {stride}, now processing {total_windows} windows")
    elif total_windows > 500:  # ภาพใหญ่
        stride = max(192, stride * 3)
        total_windows = ((h - window_size) // stride + 1) * ((w - window_size) // stride + 1)
        logger.info(f"Large image, increased stride to {stride}, now processing {total_windows} windows")
    elif total_windows > 200:  # ภาพขนาดกลาง
        stride = max(128, stride * 2)
        total_windows = ((h - window_size) // stride + 1) * ((w - window_size) // stride + 1)
        logger.info(f"Medium image, increased stride to {stride}, now processing {total_windows} windows")
    
    # จำกัดจำนวน windows สูงสุดเพื่อป้องกันการแล็ค
    max_windows = 300  # ลดจาก 500 เป็น 300
    if total_windows > max_windows:
        stride = max(256, int((h * w) ** 0.5 / (max_windows ** 0.5)))
        total_windows = ((h - window_size) // stride + 1) * ((w - window_size) // stride + 1)
        logger.info(f"Limited to max {max_windows} windows, adjusted stride to {stride}, now processing {total_windows} windows")
    
    # สร้าง sliding windows และ batch processing
    windows = []
    window_positions = []
    
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            # ตัดภาพเป็น window
            window = img[y:y+window_size, x:x+window_size]
            
            # preprocess window
            x_processed = _resize_and_scale(window, size=(192, 192), model=model)
            windows.append(x_processed)
            window_positions.append((x, y))
    
    logger.info(f"Created {len(windows)} windows for processing")
    
    # กำหนด max_process_windows สำหรับ fallback
    max_process_windows = min(50, len(windows))  # ประมวลผลไม่เกิน 50 windows
    
    # ทำนายแบบ batch เพื่อเพิ่มประสิทธิภาพ
    if windows:
        try:
            # ปรับ batch size ตามขนาดภาพ
            if len(windows) > 100:
                batch_size = min(4, len(windows))  # batch เล็กสำหรับภาพใหญ่
            elif len(windows) > 50:
                batch_size = min(6, len(windows))  # batch กลางสำหรับภาพขนาดกลาง
            else:
                batch_size = min(8, len(windows))  # batch ปกติสำหรับภาพเล็ก
            
            # ใช้ max_process_windows ที่กำหนดไว้แล้ว
            windows_to_process = windows[:max_process_windows]
            positions_to_process = window_positions[:max_process_windows]
            
            logger.info(f"Processing {len(windows_to_process)} windows in batches of {batch_size}")
            
            # ประมวลผลแบบ batch
            for batch_start in range(0, len(windows_to_process), batch_size):
                batch_end = min(batch_start + batch_size, len(windows_to_process))
                batch_windows = windows_to_process[batch_start:batch_end]
                batch_positions = positions_to_process[batch_start:batch_end]
                
                if not batch_windows:
                    break
                    
                windows_array = np.array(batch_windows)
                logger.info(f"Processing batch {batch_start//batch_size + 1}: windows {batch_start+1}-{batch_end}")
                
                # ทำนาย
                preds = model(windows_array, training=False) if callable(model) else model.predict(windows_array)
                preds = np.array(preds)
                
                # ประมวลผลผลลัพธ์
                for i, (x, y) in enumerate(batch_positions):
                    if preds.ndim == 2:
                        probs = preds[i]
                    elif preds.ndim == 1:
                        probs = preds
                    else:
                        probs = preds[i].reshape(-1)
                    
                    # หาคลาสทั้งหมดที่มีความน่าจะเป็นสูงกว่า threshold
                    for class_id, prob in enumerate(probs):
                        if prob > confidence_threshold:
                            detection = {
                                'bbox': [x, y, x + window_size, y + window_size],
                                'class_id': class_id,
                                'confidence': float(prob),
                                'class_name': str(class_id)  # แสดงเป็นตัวเลข 0-9
                            }
                            detections.append(detection)
                            logger.info(f"Found detection: class={class_id}, conf={prob:.3f}, bbox=({x},{y},{x+window_size},{y+window_size})")
            
            logger.info(f"Batch processing completed, found {len(detections)} detections")
                        
        except Exception as e:
            logger.warning(f"Batch prediction error: {e}")
            # fallback ไปใช้วิธีเดิม
            logger.info("Falling back to individual window processing")
            for i, (x, y) in enumerate(window_positions[:batch_size]):
                try:
                    window = img[y:y+window_size, x:x+window_size]
                    x_processed = _resize_and_scale(window, size=(192, 192), model=model)
                    x_processed = np.expand_dims(x_processed, axis=0)
                    
                    preds = model(x_processed, training=False) if callable(model) else model.predict(x_processed)
                    preds = np.array(preds)
                    if preds.ndim == 2:
                        probs = preds[0]
                    elif preds.ndim == 1:
                        probs = preds
                    else:
                        probs = preds.reshape(-1)
                    
                    for class_id, prob in enumerate(probs):
                        if prob > confidence_threshold:
                            detection = {
                                'bbox': [x, y, x + window_size, y + window_size],
                                'class_id': class_id,
                                'confidence': float(prob),
                                'class_name': str(class_id)  # แสดงเป็นตัวเลข 0-9
                            }
                            detections.append(detection)
                            logger.info(f"Found detection (fallback): class={class_id}, conf={prob:.3f}, bbox=({x},{y},{x+window_size},{y+window_size})")
                except Exception as e2:
                    logger.warning(f"Individual prediction error for window at ({x}, {y}): {e2}")
                    continue
    
    # ถ้าไม่มี detections และยังมี windows เหลือ ให้ประมวลผลเพิ่มเติม (จำกัดการทำงาน)
    if len(detections) == 0 and len(windows) > max_process_windows and len(windows) <= 100:
        logger.info("No detections found in first batch, processing additional windows...")
        additional_windows = windows[max_process_windows:min(max_process_windows + 20, len(windows))]  # เพิ่มอีก 20 windows
        additional_positions = window_positions[max_process_windows:min(max_process_windows + 20, len(window_positions))]
        
        # ประมวลผลทีละ window เพื่อความเสถียร
        for i, (x, y) in enumerate(additional_positions):
            try:
                window = img[y:y+window_size, x:x+window_size]
                x_processed = _resize_and_scale(window, size=(192, 192), model=model)
                x_processed = np.expand_dims(x_processed, axis=0)
                
                preds = model(x_processed, training=False) if callable(model) else model.predict(x_processed)
                preds = np.array(preds)
                if preds.ndim == 2:
                    probs = preds[0]
                elif preds.ndim == 1:
                    probs = preds
                else:
                    probs = preds.reshape(-1)
                
                for class_id, prob in enumerate(probs):
                    if prob > confidence_threshold:
                        detection = {
                            'bbox': [x, y, x + window_size, y + window_size],
                            'class_id': class_id,
                            'confidence': float(prob),
                            'class_name': str(class_id)  # แสดงเป็นตัวเลข 0-9
                        }
                        detections.append(detection)
                        logger.info(f"Found detection (additional): class={class_id}, conf={prob:.3f}, bbox=({x},{y},{x+window_size},{y+window_size})")
            except Exception as e2:
                logger.warning(f"Individual prediction error for window at ({x}, {y}): {e2}")
                continue
        
        logger.info(f"Additional processing completed, total detections: {len(detections)}")
    elif len(detections) == 0 and len(windows) > 100:
        logger.info("No detections found and too many windows, stopping to prevent lag")
    
    # ถ้ายังไม่มี detections เลย ให้ลองประมวลผลแบบง่าย ๆ (จำกัดการทำงาน)
    if len(detections) == 0 and total_windows <= 50:  # จำกัดการทำงานเฉพาะภาพเล็ก
        logger.info("No detections found, trying simplified approach...")
        # ลองประมวลผลแค่ส่วนกลางของภาพ
        center_y = h // 2
        center_x = w // 2
        half_window = window_size // 2
        
        y_start = max(0, center_y - half_window)
        y_end = min(h, center_y + half_window)
        x_start = max(0, center_x - half_window)
        x_end = min(w, center_x + half_window)
        
        try:
            center_window = img[y_start:y_end, x_start:x_end]
            if center_window.size > 0:
                x_processed = _resize_and_scale(center_window, size=(192, 192), model=model)
                x_processed = np.expand_dims(x_processed, axis=0)
                
                preds = model(x_processed, training=False) if callable(model) else model.predict(x_processed)
                preds = np.array(preds)
                if preds.ndim == 2:
                    probs = preds[0]
                elif preds.ndim == 1:
                    probs = preds
                else:
                    probs = preds.reshape(-1)
                
                for class_id, prob in enumerate(probs):
                    if prob > confidence_threshold:
                        detection = {
                            'bbox': [x_start, y_start, x_end, y_end],
                            'class_id': class_id,
                            'confidence': float(prob),
                            'class_name': str(class_id)  # แสดงเป็นตัวเลข 0-9
                        }
                        detections.append(detection)
                        logger.info(f"Found detection (center): class={class_id}, conf={prob:.3f}, bbox=({x_start},{y_start},{x_end},{y_end})")
                logger.info(f"Center window processing found {len(detections)} detections")
        except Exception as e:
            logger.warning(f"Center window processing failed: {e}")
    elif len(detections) == 0:
        logger.info("No detections found and too many windows, stopping processing to prevent infinite loop")
    
    # ใช้ Non-Maximum Suppression เพื่อลด duplicate detections
    logger.info(f"Before NMS: {len(detections)} detections")
    if detections:
        logger.info(f"Sample detection: {detections[0]}")
    final_detections = apply_nms(detections, iou_threshold=0.3)
    logger.info(f"After NMS: {len(final_detections)} detections")
    if final_detections:
        logger.info(f"Sample final detection: {final_detections[0]}")

    # แม็ปพิกัดกรอบกลับไปยังสเกลของภาพต้นฉบับ หากมีการย่อภาพตอนตรวจจับ
    if (original_w != w) or (original_h != h):
        scale_x = w / float(original_w)
        scale_y = h / float(original_h)
        inv_scale_x = 1.0 / scale_x if scale_x != 0 else 1.0
        inv_scale_y = 1.0 / scale_y if scale_y != 0 else 1.0
        logger.info(f"Scaling coordinates: original={original_w}x{original_h}, processed={w}x{h}, scale={scale_x:.3f}x{scale_y:.3f}")
        mapped = []
        for i, det in enumerate(final_detections):
            x1, y1, x2, y2 = det['bbox']
            orig_bbox = [x1, y1, x2, y2]
            x1 = int(round(x1 * inv_scale_x))
            y1 = int(round(y1 * inv_scale_y))
            x2 = int(round(x2 * inv_scale_x))
            y2 = int(round(y2 * inv_scale_y))
            # จำกัดพิกัดไม่ให้ออกนอกภาพต้นฉบับ
            x1 = max(0, min(x1, original_w - 1))
            y1 = max(0, min(y1, original_h - 1))
            x2 = max(0, min(x2, original_w - 1))
            y2 = max(0, min(y2, original_h - 1))
            new_det = dict(det)
            new_det['bbox'] = [x1, y1, x2, y2]
            mapped.append(new_det)
            logger.info(f"Detection {i+1}: {orig_bbox} -> {[x1, y1, x2, y2]}")
        logger.info("Mapped detections back to original image scale: %dx%d", original_w, original_h)
        return mapped

    return final_detections, process_steps

def apply_nms(detections: List[Dict[str, Any]], iou_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Non-Maximum Suppression เพื่อลด duplicate detections
    แยกทำ NMS ตามคลาสเพื่อให้สามารถตรวจจับหลายคลาสได้
    """
    if not detections:
        return []
    
    # แยก detections ตามคลาส
    class_detections = {}
    for det in detections:
        class_id = det['class_id']
        if class_id not in class_detections:
            class_detections[class_id] = []
        class_detections[class_id].append(det)
    
    # ทำ NMS แยกตามคลาส
    final_detections = []
    for class_id, class_dets in class_detections.items():
        # เรียงตาม confidence จากมากไปน้อย
        class_dets = sorted(class_dets, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while class_dets:
            # เอาตัวที่มี confidence สูงสุด
            current = class_dets.pop(0)
            keep.append(current)
            
            # ลบ detections ที่ overlap มากเกินไป
            class_dets = [det for det in class_dets 
                         if calculate_iou(current['bbox'], det['bbox']) < iou_threshold]
        
        final_detections.extend(keep)
    
    # เรียงผลลัพธ์ตาม confidence จากมากไปน้อย
    return sorted(final_detections, key=lambda x: x['confidence'], reverse=True)

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    คำนวณ Intersection over Union (IoU) ระหว่าง 2 bounding boxes
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # คำนวณ intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # คำนวณ union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def draw_bounding_boxes(img: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    วาด bounding boxes บนภาพ
    แสดงสีที่แตกต่างกันสำหรับแต่ละคลาส
    """
    logger.info(f"Drawing {len(detections)} bounding boxes on image {img.shape}")
    img_with_boxes = img.copy()
    # แปลงเป็น BGR เพื่อให้การวาดด้วย OpenCV แสดงสีถูกต้อง
    is_rgb = True
    try:
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
        is_rgb = False
    except Exception:
        pass
    
    # สีสำหรับแต่ละ class (เพิ่มสีมากขึ้น)
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (0, 128, 255),    # Light Blue
        (255, 0, 128),    # Pink
        (128, 255, 0),    # Lime
        (255, 128, 128),  # Light Red
        (128, 255, 128),  # Light Green
        (128, 128, 255),  # Light Blue
        (255, 255, 128),  # Light Yellow
        (255, 128, 255),  # Light Magenta
    ]
    
    # สร้าง dictionary สำหรับเก็บสีของแต่ละคลาส
    class_colors = {}
    
    for i, detection in enumerate(detections):
        x_min, y_min, x_max, y_max = detection['bbox']
        class_id = detection['class_id']
        confidence = detection['confidence']
        
        logger.info(f"Drawing box {i+1}: class={class_id}, conf={confidence:.3f}, bbox=({x_min},{y_min},{x_max},{y_max})")
        
        # เลือกสีสำหรับคลาสนี้ (ใช้สีเดิมถ้ามีแล้ว)
        if class_id not in class_colors:
            class_colors[class_id] = colors[class_id % len(colors)]
        color = class_colors[class_id]
        
        # วาด bounding box (ขนาดกรอบตามคลาส)
        box_thickness = 6
        cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), color, box_thickness)
        
        # วาด label
        label = f"{detection['class_name']}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # วาด background สำหรับ label (กันกรณีเลยขอบบน)
        bg_top = max(0, y_min - label_size[1] - 10)
        bg_bottom = max(0, y_min)
        cv2.rectangle(img_with_boxes, 
                     (x_min, bg_top),
                     (x_min + label_size[0], bg_bottom),
                     color, -1)
        
        # วาด text
        text_y = max(0, y_min - 5)
        cv2.putText(img_with_boxes, label,
                   (x_min, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # แปลงกลับเป็น RGB เพื่อให้แสดงผลถูกต้องบนเว็บ
    if not is_rgb:
        try:
            img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
    
    logger.info(f"Successfully drew {len(detections)} bounding boxes")
    return img_with_boxes

# -------------------- Projection-based pipeline --------------------
def _rectify_and_enhance_image_removed(img: np.ndarray) -> np.ndarray:
    """
    แก้ไขภาพเอียงและเพิ่มความชัด
    1) หาเส้นขอบของป้ายด้วย Canny edge detection
    2) หา contour ที่ใหญ่ที่สุด (น่าจะเป็นป้าย)
    3) หา perspective transform เพื่อแก้เอียง
    4) เพิ่มความชัดด้วย unsharp mask และ contrast enhancement
    """
    try:
        h, w = img.shape[:2]
        if h < 50 or w < 50:
            logger.warning("Image too small for rectification, skipping")
            return img
        
        # 1) Edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Gaussian blur เพื่อลด noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # 2) หา contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("No contours found, skipping rectification")
            return enhance_image(img)
        
        # หา contour ที่ใหญ่ที่สุด
        largest_contour = max(contours, key=cv2.contourArea)
        
        # ตรวจสอบว่า contour ใหญ่พอหรือไม่ (อย่างน้อย 10% ของภาพ)
        contour_area = cv2.contourArea(largest_contour)
        image_area = h * w
        if contour_area < image_area * 0.1:
            logger.warning("Largest contour too small, skipping rectification")
            return enhance_image(img)
        
        # 3) หา perspective transform
        # หา convex hull ของ contour
        hull = cv2.convexHull(largest_contour)
        
        # หา approximate polygon
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) < 4:
            logger.warning("Not enough corners found, skipping rectification")
            return enhance_image(img)
        
        # เรียงลำดับจุดตามทิศทาง (top-left, top-right, bottom-right, bottom-left)
        rect_points = order_points(approx.reshape(-1, 2))
        
        # คำนวณขนาดของสี่เหลี่ยมที่ต้องการ
        (tl, tr, br, bl) = rect_points
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # กำหนดจุดปลายทาง
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        # คำนวณ perspective transform matrix
        M = cv2.getPerspectiveTransform(rect_points.astype("float32"), dst)
        
        # ทำ perspective transform
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        
        logger.info(f"Image rectified: {warped.shape}")
        
        # 4) เพิ่มความชัด
        enhanced = enhance_image(warped)
        
        return enhanced
        
    except Exception as e:
        logger.warning(f"Rectification failed: {e}, using original image")
        return enhance_image(img)

def _order_points_removed(pts):
    """เรียงลำดับจุดตามทิศทาง (top-left, top-right, bottom-right, bottom-left)"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # top-left จะมีผลรวม x+y น้อยที่สุด
    # bottom-right จะมีผลรวม x+y มากที่สุด
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    # top-right จะมีผลต่าง x-y น้อยที่สุด
    # bottom-left จะมีผลต่าง x-y มากที่สุด
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect

def _enhance_image_removed(img: np.ndarray) -> np.ndarray:
    """
    เพิ่มความชัดและ contrast ของภาพ
    1) Unsharp mask เพื่อเพิ่มความชัด
    2) CLAHE (Contrast Limited Adaptive Histogram Equalization)
    3) Gamma correction
    """
    try:
        # แปลงเป็น BGR สำหรับ OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 1) Unsharp mask
        gaussian = cv2.GaussianBlur(img_bgr, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(img_bgr, 1.5, gaussian, -0.5, 0)
        
        # 2) CLAHE สำหรับแต่ละ channel
        lab = cv2.cvtColor(unsharp_mask, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 3) Gamma correction
        gamma = 1.2  # ปรับได้ตามต้องการ
        lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
        gamma_corrected = cv2.LUT(enhanced_bgr, lookup_table)
        
        # แปลงกลับเป็น RGB
        enhanced_rgb = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB)
        
        logger.info("Image enhanced successfully")
        return enhanced_rgb
        
    except Exception as e:
        logger.warning(f"Enhancement failed: {e}, using original image")
        return img

def _resize_keep_aspect_pad(rgb_img: np.ndarray, target_size: int = 244, pad_value: int = 0) -> np.ndarray:
    """Resize โดยรักษาสัดส่วน แล้ว pad ให้ได้ภาพสี่เหลี่ยมจัตุรัส target_size x target_size (RGB uint8)."""
    h, w = rgb_img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    scale = min(target_size / float(h), target_size / float(w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas

def get_model_input_size(model) -> int:
    """ตรวจสอบขนาด input ที่โมเดลต้องการ"""
    try:
        if hasattr(model, 'input_shape') and model.input_shape:
            if isinstance(model.input_shape, list):
                # สำหรับ multi-input models
                input_shape = model.input_shape[0]
            else:
                input_shape = model.input_shape
            
            if len(input_shape) >= 3:
                # ใช้ขนาดแรกที่พบ (height หรือ width)
                return int(input_shape[1])  # height
        return 224  # default สำหรับ MobileNetV2
    except Exception:
        return 224  # fallback for MobileNetV2

def detect_by_vertical_projection(img: np.ndarray, model, return_steps: bool = False) -> List[Dict[str, Any]]:
    """
    ขั้นตอนตามที่ร้องขอ:
      1) รับภาพป้ายเต็ม (img เป็น RGB)
      2) ครอป "แถบบรรทัดบน" ~ 12–62% ของความสูงภาพ
      3) เทา → Otsu threshold → closing
      4) Vertical projection หา segment คอลัมน์ที่มีหมึก
      5) ขยายกล่อง (padding) → resize แบบรักษาสัดส่วน + pad เป็น 244x244
      6) ส่งเข้าโมเดล classifier และสร้าง detections
    """
    original_h, original_w = img.shape[:2]
    if original_h == 0 or original_w == 0:
        return []

    # เก็บภาพขั้นตอนต่างๆ
    process_steps = {}

    # 2) Crop top band 12% - 62% (ใช้ภาพต้นฉบับโดยตรง)
    y1_band = max(0, int(round(original_h * 0.12)))
    y2_band = min(original_h, int(round(original_h * 0.62)))
    if y2_band <= y1_band:
        return []
    band_rgb = img[y1_band:y2_band, :]
    logger.info(f"Original image: {original_w}x{original_h}, Band crop: {band_rgb.shape[1]}x{band_rgb.shape[0]} (y1={y1_band}, y2={y2_band})")
    
    if return_steps:
        # แปลงเป็น base64 สำหรับแสดงผล
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(band_rgb, cv2.COLOR_RGB2BGR))
        process_steps['band_crop'] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    # 3) Gray -> Otsu -> Closing
    band_bgr = cv2.cvtColor(band_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2GRAY)
    # หมึกเป็นสีเข้ม ใช้ THRESH_BINARY_INV เพื่อให้หมึกเป็นสีขาว
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    if return_steps:
        # แปลง threshold เป็น RGB สำหรับแสดงผล
        th_rgb = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(th_rgb, cv2.COLOR_RGB2BGR))
        process_steps['threshold'] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    # 4) Vertical projection
    # ผลรวมของพิกเซลขาว (หมึก) ต่อคอลัมน์
    col_sum = np.sum(closed > 0, axis=0)  # shape (W,)
    # กำหนด threshold เล็กน้อยเพื่อกัน noise (ลดจาก 5% เป็น 2%)
    min_col_height = max(1, int(0.02 * closed.shape[0]))
    mask_cols = col_sum >= min_col_height
    
    if return_steps:
        # สร้างภาพ vertical projection
        projection_img = np.zeros((closed.shape[0], closed.shape[1], 3), dtype=np.uint8)
        for x in range(len(col_sum)):
            if mask_cols[x]:
                projection_img[:, x] = [0, 255, 0]  # สีเขียวสำหรับคอลัมน์ที่มีหมึก
        _, buffer = cv2.imencode('.jpg', projection_img)
        process_steps['projection'] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    # หา run-length ของ True เพื่อได้ช่วงคอลัมน์ต่อเนื่อง
    detections: List[Dict[str, Any]] = []
    runs: List[Tuple[int, int]] = []
    in_run = False
    run_start = 0
    
    # กรองเส้นขอบซ้ายและขวา (5% แรกและ 5% สุดท้าย)
    edge_margin = max(1, int(0.05 * mask_cols.shape[0]))
    
    for x in range(mask_cols.shape[0]):
        # ข้ามเส้นขอบซ้ายและขวา
        if x < edge_margin or x >= mask_cols.shape[0] - edge_margin:
            if in_run:
                in_run = False
                runs.append((run_start, x - 1))
            continue
            
        if mask_cols[x] and not in_run:
            in_run = True
            run_start = x
        elif not mask_cols[x] and in_run:
            in_run = False
            runs.append((run_start, x - 1))
    if in_run:
        runs.append((run_start, mask_cols.shape[0] - 1))
    
    logger.info(f"Initial runs found: {len(runs)} (excluding edge margins)")

    # กรองกล่องที่แคบเกินไป (เพิ่มความกว้างขั้นต่ำและความสูงขั้นต่ำ)
    min_box_width = max(8, int(0.01 * original_w))  # เพิ่มขั้นต่ำเป็น 8 และ 1% ของความกว้าง
    min_box_height = int(0.1 * band_rgb.shape[0])  # อย่างน้อย 10% ของความสูงแถบ
    
    filtered_runs = []
    for (x1, x2) in runs:
        box_width = x2 - x1 + 1
        # ตรวจสอบความกว้างและความสูงของกล่อง
        if box_width >= min_box_width:
            # ตรวจสอบว่ากล่องนี้มีเนื้อหาจริงหรือไม่ (ไม่ใช่เส้นขอบ)
            box_region = closed[:, x1:x2+1]
            content_height = np.sum(np.any(box_region > 0, axis=1))
            if content_height >= min_box_height:
                filtered_runs.append((x1, x2))
    
    logger.info(f"Filtered runs: {len(runs)} -> {len(filtered_runs)} (min_width: {min_box_width}, min_height: {min_box_height})")

    if not filtered_runs:
        return []

    # แบ่งสีหมึกเป็น 2 ชุดตามระยะห่าง
    selected_runs = []
    total_runs = len(filtered_runs)
    
    if total_runs < 1:
        logger.info(f"No ink colors found")
    else:
        # คำนวณระยะห่างระหว่างสีหมึก
        gaps = []
        for i in range(len(filtered_runs) - 1):
            gap = filtered_runs[i+1][0] - filtered_runs[i][1] - 1  # ระยะห่างระหว่างสีหมึก
            gaps.append(gap)
        
        logger.info(f"Gaps between ink colors: {gaps}")
        
        if len(gaps) > 0:
            # หาจุดแบ่งที่ระยะห่างมากที่สุด
            max_gap_idx = np.argmax(gaps)
            max_gap = gaps[max_gap_idx]
            avg_gap = np.mean(gaps)
            
            logger.info(f"Max gap: {max_gap} at position {max_gap_idx}, Average gap: {avg_gap:.1f}")
            
            # ถ้าระยะห่างมากที่สุดมากกว่าค่าเฉลี่ย 1.2 เท่า ให้แบ่งเป็น 2 ชุด
            if max_gap > avg_gap * 1.2:
                # ชุดแรก: สีหมึก 0 ถึง max_gap_idx
                first_group = filtered_runs[:max_gap_idx + 1]
                # ชุดที่สอง: สีหมึก max_gap_idx + 1 ถึงสุดท้าย
                second_group = filtered_runs[max_gap_idx + 1:]
                
                logger.info(f"Split into 2 groups: First group {len(first_group)} colors, Second group {len(second_group)} colors")
                
                # ชุดแรก: ถ้ามี 3 ตัว ให้ใช้แค่ตัวแรก, ถ้ามี 1-2 ตัว ไม่ต้องใช้
                if len(first_group) >= 3:
                    selected_runs.extend(first_group[:1])  # ใช้แค่ตัวแรก
                    logger.info(f"First group: Using first 1 out of {len(first_group)} colors (3+ colors rule)")
                else:
                    # ถ้ามี 1-2 ตัว ไม่ใช้เลย
                    logger.info(f"First group: Not using any of {len(first_group)} colors (1-2 colors rule)")
                
                # ชุดที่สอง: ใช้ทั้งหมด
                if len(second_group) > 0:
                    selected_runs.extend(second_group)
                    logger.info(f"Second group: Using all {len(second_group)} colors")
                else:
                    logger.info(f"Second group: No colors to use")
            else:
                # ไม่พบการแบ่งที่ชัดเจน ให้ใช้ทั้งหมด
                selected_runs = filtered_runs
                logger.info(f"No clear split found, using all {total_runs} colors")
        else:
            # มีแค่ 1 สีหมึก
            selected_runs = filtered_runs
            logger.info(f"Only 1 ink color, using it")
    
    logger.info(f"Selected runs: {len(selected_runs)} out of {total_runs} total runs")

    if return_steps:
        # สร้างภาพกล่องตัวอักษร (แสดงการแบ่งชุด)
        boxes_img = band_rgb.copy()
        
        # วาดกล่องที่กรองแล้ว (สีน้ำเงิน)
        for i, (x1, x2) in enumerate(filtered_runs):
            cv2.rectangle(boxes_img, (x1, 0), (x2, band_rgb.shape[0]-1), (255, 0, 0), 1)
            cv2.putText(boxes_img, str(i+1), (x1+2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # วาดกล่องที่เลือกแล้ว (สีเขียว) และแสดงชุด
        for i, (x1, x2) in enumerate(selected_runs):
            cv2.rectangle(boxes_img, (x1, 0), (x2, band_rgb.shape[0]-1), (0, 255, 0), 2)
            # แสดงหมายเลขชุด (G1 = ชุดแรก, G2 = ชุดที่สอง)
            if i == 0 and len(selected_runs) > 1:
                # ตัวแรกเป็นชุดแรก (ถ้ามี)
                cv2.putText(boxes_img, "G1-1", (x1+2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                # ตัวที่เหลือเป็นชุดที่สอง
                cv2.putText(boxes_img, f"G2-{i}", (x1+2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # วาดเส้นแบ่งชุด (ถ้ามี)
        if len(filtered_runs) > 1:
            gaps = []
            for i in range(len(filtered_runs) - 1):
                gap = filtered_runs[i+1][0] - filtered_runs[i][1] - 1
                gaps.append(gap)
            
            if len(gaps) > 0:
                max_gap_idx = np.argmax(gaps)
                max_gap = gaps[max_gap_idx]
                avg_gap = np.mean(gaps)
                
                if max_gap > avg_gap * 1.2:
                    # วาดเส้นแบ่งชุด
                    split_x = filtered_runs[max_gap_idx][1] + max_gap // 2
                    cv2.line(boxes_img, (split_x, 0), (split_x, band_rgb.shape[0]-1), (0, 0, 255), 2)
                    cv2.putText(boxes_img, "SPLIT", (split_x+5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(boxes_img, cv2.COLOR_RGB2BGR))
        process_steps['boxes'] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    # 5) Padding และเตรียมอินพุต classifier
    pad_ratio = 0.05  # 5% แต่ไม่น้อยกว่า 2 พิกเซล
    crops_rgb: List[np.ndarray] = []
    boxes_abs: List[Tuple[int, int, int, int]] = []
    for (x1, x2) in selected_runs:
        # กล่องใน band
        bx1 = x1
        bx2 = x2
        by1 = 0
        by2 = band_rgb.shape[0] - 1
        # padding ในระบบพิกัดของภาพต้นฉบับ
        bw = bx2 - bx1 + 1
        pad = max(2, int(round(bw * pad_ratio)))
        abs_x1 = max(0, bx1 - pad)
        abs_x2 = min(original_w - 1, bx2 + pad)
        abs_y1 = max(0, y1_band - pad)
        abs_y2 = min(original_h - 1, y2_band + pad)
        # crop จากภาพต้นฉบับ RGB
        crop = img[abs_y1:abs_y2 + 1, abs_x1:abs_x2 + 1]
        if crop.size == 0:
            continue
        # ตรวจสอบขนาดกล่องก่อน resize
        crop_h, crop_w = crop.shape[:2]
        if crop_h < 20 or crop_w < 20:
            logger.warning(f"Crop {len(crops_rgb)+1} too small: {crop_w}x{crop_h}, skipping")
            continue
            
        # ใช้เทคนิค square pad + resize เพื่อรักษาสัดส่วน
        # 1. หาขนาดสี่เหลี่ยมจัตุรัสที่เหมาะสม
        max_dimension = max(crop_h, crop_w)
        if max_dimension < 30:
            square_size = 64  # ขั้นต่ำ 64x64
        elif max_dimension < 50:
            square_size = 128  # ขั้นต่ำ 128x128
        else:
            square_size = min(244, max(128, max_dimension * 2))  # 2 เท่าของขนาดเดิม แต่ไม่เกิน 244
        
        # 2. Square pad (ใส่แถบขาวให้เป็นสี่เหลี่ยมจัตุรัส)
        crop_squared = _resize_keep_aspect_pad(crop, target_size=square_size, pad_value=255)  # ใช้สีขาวแทนสีดำ
        
        # 3. Resize เป็น 244x244
        crop_resized = cv2.resize(crop_squared, (244, 244), interpolation=cv2.INTER_AREA)
        
        logger.info(f"Crop {len(crops_rgb)+1}: Original {crop_w}x{crop_h} -> Square pad {square_size}x{square_size} -> Resize 244x244")
        crops_rgb.append(crop_resized)
        boxes_abs.append((abs_x1, abs_y1, abs_x2, abs_y2))
        
        if return_steps and len(crops_rgb) == 1:  # แสดงเฉพาะกล่องแรก
            # ภาพ crop + padding
            crop_with_padding = img[abs_y1:abs_y2 + 1, abs_x1:abs_x2 + 1]
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(crop_with_padding, cv2.COLOR_RGB2BGR))
            process_steps['crop'] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
            
            # ภาพ square pad
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(crop_squared, cv2.COLOR_RGB2BGR))
            process_steps['square_pad'] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
            
            # ภาพ resize 244x244
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))
            process_steps['resize244'] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    if not crops_rgb:
        return []

    # 6) ส่งเข้าโมเดล classifier (batch)
    # ตรวจสอบขนาด input ที่โมเดลต้องการ
    input_size = get_model_input_size(model)
    logger.info(f"Model input size detected: {input_size}x{input_size}")
    
    # เตรียมเป็น float32 [0,1] และ grayscale->rgb เพื่อสอดคล้องกับ preprocess เดิม
    batch = []
    for crop in crops_rgb:
        x = crop.astype(np.float32)
        # ใช้ภาพขาวดำโดยตรง (ไม่ต้องแปลงกลับเป็น RGB)
        t = tf.convert_to_tensor(x, dtype=tf.float32)
        # ไม่แปลงเป็น grayscale เพราะโมเดลต้องการ 3 channels
        # t = tf.image.rgb_to_grayscale(t)  # แปลงเป็นเทา
        
        # ใช้ขนาดที่โมเดลต้องการ (224x224 สำหรับ MobileNetV2)
        t = tf.image.resize(t, (input_size, input_size), method="bilinear")
        x_resized = (t.numpy() / 255.0)
        
        # Apply preprocessing based on model type
        model_name = get_current_model_path().split('/')[-1].replace('.keras', '')
        if 'EfficientNet' in model_name:
            x_resized = tf.keras.applications.efficientnet.preprocess_input(x_resized * 255.0)
        elif 'MobileNet' in model_name:
            x_resized = tf.keras.applications.mobilenet_v2.preprocess_input(x_resized * 255.0)
        elif 'NASNet' in model_name:
            x_resized = tf.keras.applications.nasnet.preprocess_input(x_resized * 255.0)
        else:
            x_resized = x_resized  # No preprocessing
        logger.info(f"Final input for model: {x_resized.shape} (resized to {input_size}x{input_size})")
        batch.append(x_resized)
        
        if return_steps and len(batch) == 1:  # แสดงเฉพาะภาพแรก
            # ใช้ภาพจากขั้นตอนที่ 7 (224x224) เป็นฐานสำหรับแสดงผล
            base_img = crop_resized.copy()
            
            # ภาพ grayscale (แปลงภาพ 224x224 เป็นขาวดำ) - แสดงเป็นขาวดำจริงๆ
            gray_base = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)
            # ไม่ต้องแปลงกลับเป็น RGB เพื่อให้เห็นเป็นขาวดำจริงๆ
            _, buffer = cv2.imencode('.jpg', gray_base)
            process_steps['grayscale_rgb'] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    batch_array = np.stack(batch, axis=0)

    preds = model(batch_array, training=False) if callable(model) else model.predict(batch_array)
    preds = np.array(preds)
    if preds.ndim == 1:
        preds = preds.reshape(-1, preds.shape[0])

    detections = []
    for i, (x1, y1, x2, y2) in enumerate(boxes_abs):
        if i >= preds.shape[0]:
            break
        probs = preds[i].reshape(-1)
        class_id = int(np.argmax(probs))
        conf = float(probs[class_id])
        
        # แปลง class_id เป็นตัวเลข 0-9 (ตรงกับ class_id)
        # เพราะโมเดลเทรนด้วย class 0-9 และเราต้องการแสดง 0-9
        display_number = class_id
        
        det = {
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "class_id": class_id,
            "confidence": conf,
            "class_name": str(display_number),  # แสดงเป็นตัวเลข 0-9
            "position": i + 1,  # ตำแหน่งที่ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        }
        detections.append(det)

    # ไม่ต้องแปลงพิกัดเพราะใช้ภาพต้นฉบับโดยตรงแล้ว
    logger.info("Projection detections using original image coordinates: %dx%d", original_w, original_h)
    
    # ไม่ต้อง NMS เพราะกล่องเป็นคอลัมน์ไม่ overlap กันมากนัก แต่จัดเรียงตาม x1 เพื่ออ่านง่าย
    detections = sorted(detections, key=lambda d: d["bbox"][0])
    logger.info("Projection detections: %d", len(detections))
    
    if return_steps:
        logger.info(f"Returning {len(process_steps)} process steps: {list(process_steps.keys())}")
        return detections, process_steps
    return detections

def _resize_and_scale(img: np.ndarray, size=(192, 192), model=None) -> np.ndarray:
    """รับ ndarray (H,W,3) RGB/BGR/uint8 ก็ได้ -> float32 [0,1] size x size"""
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
    
    # ใช้ขนาดที่โมเดลต้องการถ้ามี
    if model is not None:
        try:
            input_size = get_model_input_size(model)
            size = (input_size, input_size)
        except Exception:
            pass  # ใช้ size เดิม
    
    x = tf.image.resize(x, size, method="bilinear").numpy()
    x = x / 255.0
    return x

def test_draw_bounding_boxes():
    """Test function to verify bounding box drawing works"""
    import numpy as np
    # Create a test image
    test_img = np.ones((400, 400, 3), dtype=np.uint8) * 255  # White image
    
    # Create test detections
    test_detections = [
        {
            'bbox': [50, 50, 150, 150],
            'class_id': 0,
            'confidence': 0.85,
            'class_name': 'Test_Class_0'
        },
        {
            'bbox': [200, 200, 300, 300],
            'class_id': 1,
            'confidence': 0.92,
            'class_name': 'Test_Class_1'
        }
    ]
    
    # Draw bounding boxes
    result_img = draw_bounding_boxes(test_img, test_detections)
    
    # Save test image
    import cv2
    cv2.imwrite('test_bounding_boxes.jpg', result_img)
    logger.info("Test bounding box image saved as 'test_bounding_boxes.jpg'")
    
    return result_img

def predict_from_ndarray(img: np.ndarray, enable_detection: bool = True) -> Dict[str, Any]:
    """
    รับภาพเป็น ndarray แล้วคืนค่า:
    {
        "ok": True,
        "probabilities": [..],
        "top1": {"index": int, "prob": float},
        "detections": [{"bbox": [x1,y1,x2,y2], "class_id": int, "confidence": float, "class_name": str}],
        "model_info": {"name": str, "path": str, "size_mb": float}
    }
    """
    try:
        model = load_model()
    except Exception as e:
        logger.error("Model load error: %s", e)
        return {"ok": False, "error": str(e)}
    
    # ตรวจสอบขนาดภาพก่อนประมวลผล
    h, w = img.shape[:2]
    original_size = h * w
    
    # ปรับขนาดภาพตามขนาดเพื่อประสิทธิภาพที่ดีขึ้น
    if original_size > 5000000:  # ภาพใหญ่มาก (มากกว่า 5MP)
        logger.warning(f"Very large image detected: {h}x{w} ({original_size:,} pixels), resizing for performance")
        scale = (2000000 / original_size) ** 0.5
        new_h = int(h * scale)
        new_w = int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        logger.info(f"Resized image to {h}x{w} ({h*w:,} pixels, scale: {scale:.2f})")
    elif original_size > 2000000:  # ภาพใหญ่ (มากกว่า 2MP)
        logger.warning(f"Large image detected: {h}x{w} ({original_size:,} pixels), resizing for performance")
        scale = (1500000 / original_size) ** 0.5
        new_h = int(h * scale)
        new_w = int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        logger.info(f"Resized image to {h}x{w} ({h*w:,} pixels, scale: {scale:.2f})")
    else:
        logger.info(f"Processing image: {h}x{w} ({original_size:,} pixels)")

    try:
        # เก็บข้อมูลโมเดลที่ใช้
        model_info = get_current_model_info()
        
        # ทำนายแบบเดิม (classification)
        x = _resize_and_scale(img, size=(192, 192), model=model)
        x = np.expand_dims(x, axis=0)  # (1,size,size,3)
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
        
        result = {
            "ok": True,
            "probabilities": [float(p) for p in probs.tolist()],
            "top1": {"index": top_idx, "prob": top_prob},
            "model_info": {
                "name": model_info.get("model_name", "Unknown"),
                "path": model_info.get("model_path", "Unknown"),
                "size_mb": model_info.get("file_size_mb", 0)
            }
        }
        
        # เพิ่ม object detection ถ้าเปิดใช้งาน
        if enable_detection:
            try:
                detection_result = detect_objects_with_bounding_boxes(img, model)
                if isinstance(detection_result, tuple) and len(detection_result) == 2:
                    detections, process_steps = detection_result
                else:
                    detections = detection_result
                    process_steps = {}
                
                result["detections"] = detections
                result["process_steps"] = process_steps
                
                # สร้างสถิติการตรวจจับ
                class_counts = {}
                for det in detections:
                    class_id = det['class_id']
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                
                result["detection_stats"] = {
                    "total_objects": len(detections),
                    "unique_classes": len(class_counts),
                    "class_counts": class_counts
                }
                
                logger.info(f"Found {len(detections)} objects across {len(class_counts)} classes")
            except Exception as e:
                logger.warning(f"Object detection failed: {e}")
                result["detections"] = []
                result["process_steps"] = {}
                result["detection_stats"] = {"total_objects": 0, "unique_classes": 0, "class_counts": {}}
        
        return result
    except Exception as e:
        logger.exception("Predict error")
        return {"ok": False, "error": f"Predict failed: {e}"}
    finally:
        # ทำความสะอาด memory
        try:
            import gc
            gc.collect()
        except Exception:
            pass
