# Troubleshooting Guide

## Overview

คู่มือนี้จะช่วยแก้ไขปัญหาที่พบบ่อยในการใช้งานระบบ **CNN Model Deployment**

## Table of Contents

- [Common Issues](#common-issues)
- [Installation Problems](#installation-problems)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Model Issues](#model-issues)
- [UI/UX Problems](#uiux-problems)
- [Deployment Issues](#deployment-issues)
- [Debugging Tools](#debugging-tools)
- [Getting Help](#getting-help)

## Common Issues

### **1. ภาพไม่ขึ้นในเว็บ**

#### **อาการ:**
- อัปโหลดภาพแล้วไม่เห็นภาพ
- แสดง error message
- ระบบค้าง

#### **สาเหตุ:**
- ไฟล์ไม่รองรับ
- ขนาดไฟล์ใหญ่เกินไป
- ไฟล์เสียหาย

#### **วิธีแก้:**
```bash
# ตรวจสอบรูปแบบไฟล์
file image.jpg
# ควรแสดง: image.jpg: JPEG image data

# ตรวจสอบขนาดไฟล์
ls -lh image.jpg
# ควรแสดง: -rw-r--r-- 1 user user 2.5M Jan 1 12:00 image.jpg

# ตรวจสอบไฟล์ที่รองรับ
# รองรับ: JPG, JPEG, PNG
# ไม่รองรับ: GIF, BMP, TIFF, WEBP
```

#### **Prevention:**
- ใช้ไฟล์ JPG, PNG, JPEG เท่านั้น
- ขนาดไฟล์ไม่เกิน 10MB
- ตรวจสอบไฟล์ก่อนอัปโหลด

### **2. โมเดลไม่โหลด**

#### **อาการ:**
- Error: "Model not found"
- ระบบค้างที่การโหลดโมเดล
- API ไม่ตอบสนอง

#### **สาเหตุ:**
- ไฟล์โมเดลไม่มี
- ไฟล์โมเดลเสียหาย
- Memory ไม่เพียงพอ

#### **วิธีแก้:**
```bash
# ตรวจสอบไฟล์โมเดล
ls -la models/
# ควรเห็น: MobileNetV2_Augmented_best.keras

# ตรวจสอบขนาดไฟล์
du -h models/*.keras
# ควรเห็น: 15M models/MobileNetV2_Augmented_best.keras

# ตรวจสอบ permissions
ls -la models/MobileNetV2_Augmented_best.keras
# ควรเห็น: -rw-r--r-- 1 user user 15M Jan 1 12:00 models/MobileNetV2_Augmented_best.keras
```

#### **Prevention:**
- ตรวจสอบไฟล์โมเดลก่อนรัน
- ใช้ไฟล์โมเดลที่ถูกต้อง
- มี memory เพียงพอ

### **3. ผลลัพธ์ไม่ถูกต้อง**

#### **อาการ:**
- ตรวจจับตัวเลขผิด
- ไม่พบตัวเลขที่ควรมี
- ความมั่นใจต่ำ

#### **สาเหตุ:**
- ภาพไม่ชัดเจน
- โมเดลไม่เหมาะกับภาพ
- ภาพไม่เหมาะสม

#### **วิธีแก้:**
```python
# ตรวจสอบภาพ
import cv2
import numpy as np

# โหลดภาพ
img = cv2.imread('image.jpg')

# ตรวจสอบขนาด
print(f"Image shape: {img.shape}")

# ตรวจสอบความชัด
laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
print(f"Image sharpness: {laplacian_var}")

# ตรวจสอบความสว่าง
mean_brightness = np.mean(img)
print(f"Mean brightness: {mean_brightness}")
```

#### **Prevention:**
- ใช้ภาพที่ชัดเจน
- แสงสว่างเพียงพอ
- ตัวเลขเด่นชัด

### **4. ระบบค้าง**

#### **อาการ:**
- ระบบไม่ตอบสนอง
- Progress indicator ค้าง
- Browser ไม่ตอบสนอง

#### **สาเหตุ:**
- ภาพใหญ่เกินไป
- Memory ไม่เพียงพอ
- CPU ใช้ 100%

#### **วิธีแก้:**
```bash
# ตรวจสอบ memory
free -h
# ควรมี free memory อย่างน้อย 2GB

# ตรวจสอบ CPU
top
# ควรไม่ใช้ CPU 100% ตลอดเวลา

# ตรวจสอบ disk space
df -h
# ควรมี free space อย่างน้อย 1GB
```

#### **Prevention:**
- ใช้ภาพขนาดเล็ก
- มี memory เพียงพอ
- ปิดโปรแกรมอื่นที่ไม่จำเป็น

## Installation Problems

### **1. Python Version Error**

#### **Error:**
```
Python 3.7.0 is not supported. Please use Python 3.8+
```

#### **วิธีแก้:**
```bash
# ตรวจสอบ Python version
python --version
# ควรแสดง: Python 3.8.0 หรือใหม่กว่า

# ติดตั้ง Python 3.9
# Ubuntu/Debian
sudo apt update
sudo apt install python3.9 python3.9-pip

# macOS
brew install python@3.9

# Windows
# ดาวน์โหลดจาก https://python.org
```

### **2. Dependencies Installation Error**

#### **Error:**
```
ERROR: Could not find a version that satisfies the requirement tensorflow
```

#### **วิธีแก้:**
```bash
# อัปเดต pip
pip install --upgrade pip

# ติดตั้ง dependencies ทีละตัว
pip install tensorflow
pip install fastapi
pip install opencv-python
pip install numpy
pip install pillow

# หรือใช้ requirements.txt
pip install -r requirements.txt
```

### **3. CUDA Error**

#### **Error:**
```
Could not load dynamic library 'cudart64_110.dll'
```

#### **วิธีแก้:**
```bash
# ติดตั้ง CPU version
pip uninstall tensorflow
pip install tensorflow-cpu

# หรือติดตั้ง CUDA version
pip install tensorflow[and-cuda]
```

### **4. OpenCV Error**

#### **Error:**
```
ImportError: No module named 'cv2'
```

#### **วิธีแก้:**
```bash
# ติดตั้ง OpenCV
pip install opencv-python

# หรือติดตั้ง opencv-python-headless
pip install opencv-python-headless
```

## Runtime Errors

### **1. Model Loading Error**

#### **Error:**
```
ValueError: Unknown layer: CustomLayer
```

#### **วิธีแก้:**
```python
# ตรวจสอบโมเดล
import tensorflow as tf

try:
    model = tf.keras.models.load_model('models/model.keras')
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading error: {e}")
```

### **2. Image Processing Error**

#### **Error:**
```
cv2.error: OpenCV(4.8.0) /tmp/opencv/modules/imgproc/src/resize.cpp:4051: error: (-215:Assertion failed) !ssize.empty() in function 'resize'
```

#### **วิธีแก้:**
```python
# ตรวจสอบภาพ
import cv2

def validate_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Invalid image file")
        return False
    
    if img.shape[0] == 0 or img.shape[1] == 0:
        print("Empty image")
        return False
    
    print(f"Image shape: {img.shape}")
    return True
```

### **3. Memory Error**

#### **Error:**
```
RuntimeError: CUDA out of memory
```

#### **วิธีแก้:**
```python
# จำกัด memory usage
import tensorflow as tf

# จำกัด GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ใช้ CPU แทน
tf.config.set_visible_devices([], 'GPU')
```

### **4. API Error**

#### **Error:**
```
HTTPException: 422 Unprocessable Entity
```

#### **วิธีแก้:**
```python
# ตรวจสอบ request
from fastapi import HTTPException

def validate_file(file):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=422, detail="File must be an image")
    
    if file.size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=422, detail="File too large")
    
    return True
```

## Performance Issues

### **1. Slow Prediction**

#### **อาการ:**
- การทำนายใช้เวลานาน
- ระบบค้างนาน

#### **วิธีแก้:**
```python
# ใช้ batch processing
def predict_batch(images):
    batch = np.array(images)
    predictions = model.predict(batch)
    return predictions

# จำกัดขนาดภาพ
def resize_image(img, max_size=800):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
    return img
```

### **2. High Memory Usage**

#### **อาการ:**
- Memory ใช้มาก
- ระบบช้า

#### **วิธีแก้:**
```python
# ใช้ garbage collection
import gc

def process_image(img):
    result = model.predict(img)
    del img  # ลบตัวแปรที่ไม่ใช้
    gc.collect()  # เรียก garbage collection
    return result

# จำกัด batch size
BATCH_SIZE = 4
for i in range(0, len(images), BATCH_SIZE):
    batch = images[i:i+BATCH_SIZE]
    predictions = model.predict(batch)
```

### **3. Slow Image Loading**

#### **อาการ:**
- ภาพโหลดช้า
- ระบบค้าง

#### **วิธีแก้:**
```python
# ใช้ lazy loading
def load_image_lazy(img_path):
    def _load():
        return cv2.imread(img_path)
    return _load

# ใช้ image compression
def compress_image(img, quality=85):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode('.jpg', img, encode_param)
    return encoded_img
```

## Model Issues

### **1. Model Not Found**

#### **Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/model.keras'
```

#### **วิธีแก้:**
```bash
# ตรวจสอบไฟล์โมเดล
ls -la models/
# ควรเห็นไฟล์ .keras

# ดาวน์โหลดโมเดลใหม่
# (หากไม่มี)
```

### **2. Model Loading Error**

#### **Error:**
```
ValueError: Unknown layer: CustomLayer
```

#### **วิธีแก้:**
```python
# ใช้ custom_objects
custom_objects = {
    'CustomLayer': CustomLayer
}
model = tf.keras.models.load_model('model.keras', custom_objects=custom_objects)
```

### **3. Model Prediction Error**

#### **Error:**
```
ValueError: Input 0 of layer "conv2d" is incompatible with the layer: expected axis -1 of input shape to have value 3, but received input with shape (None, 224, 224, 1)
```

#### **วิธีแก้:**
```python
# ตรวจสอบ input shape
print(f"Model input shape: {model.input_shape}")
print(f"Image shape: {img.shape}")

# แปลง grayscale เป็น RGB
if len(img.shape) == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
```

### **4. Model Performance Issues**

#### **อาการ:**
- ความแม่นยำต่ำ
- ผลลัพธ์ไม่ถูกต้อง

#### **วิธีแก้:**
```python
# ใช้ preprocessing ที่ถูกต้อง
def preprocess_image(img, model_name):
    if 'EfficientNet' in model_name:
        img = tf.keras.applications.efficientnet.preprocess_input(img)
    elif 'MobileNet' in model_name:
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img
```

## UI/UX Problems

### **1. Layout Issues**

#### **อาการ:**
- Layout เสีย
- Elements ซ้อนกัน
- ไม่ responsive

#### **วิธีแก้:**
```css
/* ตรวจสอบ CSS */
.container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

@media (max-width: 768px) {
    .container {
        flex-direction: column;
    }
}
```

### **2. JavaScript Errors**

#### **Error:**
```
Uncaught TypeError: Cannot read property 'value' of null
```

#### **วิธีแก้:**
```javascript
// ตรวจสอบ element ก่อนใช้
const element = document.getElementById('element-id');
if (element) {
    element.value = 'new value';
} else {
    console.error('Element not found');
}
```

### **3. Image Display Issues**

#### **อาการ:**
- ภาพไม่แสดง
- ภาพบิดเบี้ยว
- ภาพใหญ่เกินไป

#### **วิธีแก้:**
```css
/* จำกัดขนาดภาพ */
img {
    max-width: 100%;
    height: auto;
    object-fit: contain;
}

/* ใช้ aspect ratio */
.image-container {
    aspect-ratio: 16/9;
    overflow: hidden;
}
```

## Deployment Issues

### **1. Port Already in Use**

#### **Error:**
```
OSError: [Errno 98] Address already in use
```

#### **วิธีแก้:**
```bash
# ตรวจสอบ port ที่ใช้
lsof -i :8000

# ฆ่า process ที่ใช้ port
kill -9 <PID>

# หรือใช้ port อื่น
uvicorn app.main:app --port 8001
```

### **2. Docker Build Error**

#### **Error:**
```
ERROR: failed to solve: failed to resolve source
```

#### **วิธีแก้:**
```dockerfile
# ใช้ base image ที่ถูกต้อง
FROM python:3.9-slim

# ติดตั้ง dependencies ก่อน
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1
```

### **3. Environment Variables**

#### **Error:**
```
KeyError: 'MODEL_PATH'
```

#### **วิธีแก้:**
```python
# ใช้ default values
import os

MODEL_PATH = os.getenv('MODEL_PATH', 'models/')
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', '10485760'))
```

## Debugging Tools

### **1. Logging**

#### **Python Logging:**
```python
import logging

# ตั้งค่า logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# ใช้ logging
logger.info("Processing image...")
logger.error(f"Error: {e}")
```

#### **JavaScript Logging:**
```javascript
// ใช้ console.log
console.log('Processing file...');
console.error('Error:', error);

// ใช้ performance API
console.time('image-processing');
// ... processing code ...
console.timeEnd('image-processing');
```

### **2. Debug Mode**

#### **FastAPI Debug:**
```python
# เปิด debug mode
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
```

#### **Frontend Debug:**
```javascript
// เปิด Dev Mode
const devMode = true;

if (devMode) {
    console.log('Dev Mode enabled');
    // แสดง debug information
}
```

### **3. Performance Monitoring**

#### **Python Profiling:**
```python
import cProfile
import pstats

# Profile function
def profile_function():
    cProfile.run('your_function()', 'profile_output.prof')
    
    # Analyze results
    stats = pstats.Stats('profile_output.prof')
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

#### **Browser DevTools:**
```javascript
// ใช้ Performance API
const start = performance.now();
// ... code ...
const end = performance.now();
console.log(`Execution time: ${end - start} milliseconds`);
```

## Getting Help

### **1. Self-Help**

#### **Check Logs:**
```bash
# ดู application logs
tail -f logs/app.log

# ดู system logs
journalctl -u your-service

# ดู browser console
# เปิด Developer Tools (F12)
```

#### **Check Documentation:**
- [README.md](README.md)
- [API Documentation](docs/API_DOCUMENTATION.md)
- [Installation Guide](docs/INSTALLATION_GUIDE.md)
- [User Guide](docs/USER_GUIDE.md)
- [Developer Guide](docs/DEVELOPER_GUIDE.md)

### **2. Community Help**

#### **GitHub Issues:**
- [Create Issue](https://github.com/yourusername/cnn-model-deployment/issues/new)
- [Search Issues](https://github.com/yourusername/cnn-model-deployment/issues)
- [Discussions](https://github.com/yourusername/cnn-model-deployment/discussions)

#### **Stack Overflow:**
- Tag: `fastapi`, `tensorflow`, `opencv`
- Include: error message, code, environment

### **3. Professional Support**

#### **Contact Information:**
- **Email:** your-email@example.com
- **Discord:** [Discord Server](https://discord.gg/your-server)
- **GitHub:** [@yourusername](https://github.com/yourusername)

#### **Response Times:**
- **Critical Issues:** 24 hours
- **Regular Issues:** 3-5 days
- **Feature Requests:** 1-2 weeks

### **4. Issue Reporting**

#### **Bug Report Template:**
```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## 📸 Screenshots
(If applicable)

## Environment
- OS: [e.g. Windows 10, macOS 12, Ubuntu 20.04]
- Python: [e.g. 3.9.7]
- Browser: [e.g. Chrome 91, Firefox 89]
- Version: [e.g. 1.0.0]

## Additional Context
Any other context about the problem
```

## Additional Resources

### **Documentation**
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Python Documentation](https://docs.python.org/)

### **Tutorials**
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [TensorFlow Tutorial](https://www.tensorflow.org/tutorials)
- [OpenCV Tutorial](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

### **Community**
- [FastAPI Discord](https://discord.gg/9Z9CgHQ)
- [TensorFlow Forum](https://discuss.tensorflow.org/)
- [Python Reddit](https://www.reddit.com/r/Python/)
- [Stack Overflow](https://stackoverflow.com/)

---

## **สรุป**

คู่มือนี้ครอบคลุมปัญหาที่พบบ่อยและวิธีแก้ไข หากยังมีปัญหาที่ไม่พบในคู่มือนี้ สามารถติดต่อขอความช่วยเหลือได้

**สู้ๆ ครับ!**
