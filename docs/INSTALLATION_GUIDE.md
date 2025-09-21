# Installation Guide

## Prerequisites

### **System Requirements**
- **Python:** 3.8+ (แนะนำ 3.9+)
- **RAM:** 4GB+ (แนะนำ 8GB+)
- **Storage:** 2GB+ free space
- **OS:** Windows 10+, macOS 10.14+, Ubuntu 18.04+

### **Software Requirements**
- **Python Package Manager:** pip
- **Git:** สำหรับ clone repository
- **Web Browser:** Chrome, Firefox, Safari, Edge

## การติดตั้ง

### **1. Clone Repository**
```bash
git clone https://github.com/yourusername/cnn-model-deployment.git
cd cnn-model-deployment
```

### **2. สร้าง Virtual Environment**
```bash
# สร้าง virtual environment
python -m venv venv

# เปิดใช้งาน (Windows)
venv\Scripts\activate

# เปิดใช้งาน (macOS/Linux)
source venv/bin/activate
```

### **3. ติดตั้ง Dependencies**
```bash
pip install -r requirements.txt
```

### **4. ตรวจสอบการติดตั้ง**
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import fastapi; print('FastAPI version:', fastapi.__version__)"
```

## โครงสร้างโปรเจค

```
cnn-model-deployment/
├── app/
│   ├── main.py              # FastAPI application
│   ├── inference.py         # AI model logic
│   └── templates/
│       └── index.html       # Frontend UI
├── models/                  # โมเดล AI (ต้องมีไฟล์ .keras)
├── docs/                    # เอกสาร
├── requirements.txt         # Dependencies
├── render.yaml             # Deployment config
└── README.md               # เอกสารหลัก
```

## การรันแอปพลิเคชัน

### **Development Mode**
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### **Production Mode**
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### **เปิดเว็บเบราว์เซอร์**
```
http://localhost:8000
```

## การตรวจสอบ

### **1. ตรวจสอบ API**
```bash
curl http://localhost:8000/available-models
```

**Expected Response:**
```json
{
  "models": [
    "MobileNetV2_Augmented_best.keras",
    "EfficientNetB0_Augmented_best.keras"
  ]
}
```

### **2. ตรวจสอบ Model Info**
```bash
curl http://localhost:8000/model-info
```

**Expected Response:**
```json
{
  "model_name": "MobileNetV2_Augmented",
  "model_path": "models/MobileNetV2_Augmented_best.keras",
  "file_size_mb": 15.2,
  "input_size": 224,
  "num_classes": 10
}
```

### **3. ตรวจสอบ Web Interface**
- เปิด `http://localhost:8000`
- ควรเห็นหน้าเว็บที่สวยงาม
- ลองอัปโหลดภาพทดสอบ

## Troubleshooting

### **ปัญหาที่พบบ่อย**

#### **1. ModuleNotFoundError**
```bash
# Error: ModuleNotFoundError: No module named 'tensorflow'
pip install tensorflow
```

#### **2. CUDA Error (GPU)**
```bash
# Error: Could not load dynamic library 'cudart64_110.dll'
# วิธีแก้: ติดตั้ง CPU version
pip uninstall tensorflow
pip install tensorflow-cpu
```

#### **3. Port Already in Use**
```bash
# Error: Address already in use
# วิธีแก้: เปลี่ยน port
python -m uvicorn app.main:app --port 8001
```

#### **4. Model File Not Found**
```bash
# Error: Model file not found
# วิธีแก้: ตรวจสอบไฟล์โมเดล
ls models/
# ควรมีไฟล์ .keras
```

#### **5. Memory Error**
```bash
# Error: Out of memory
# วิธีแก้: ลดขนาดภาพหรือใช้โมเดลเล็กกว่า
```

### **Debug Mode**

#### **เปิด Debug Logging**
```python
# ใน app/main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### **ตรวจสอบ Logs**
```bash
# ดู logs แบบ real-time
tail -f logs/app.log
```

## Configuration

### **Environment Variables**
```bash
# สร้างไฟล์ .env
echo "MODEL_PATH=models/" > .env
echo "MAX_FILE_SIZE=10485760" >> .env
echo "DEBUG=True" >> .env
```

### **Model Configuration**
```python
# ใน app/inference.py
MODEL_CONFIG = {
    "default_model": "MobileNetV2_Augmented_best.keras",
    "input_size": 224,
    "confidence_threshold": 0.3
}
```

## Deployment

### **1. Local Production**
```bash
# ใช้ Gunicorn
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### **2. Docker**
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build และรัน
docker build -t cnn-model-deployment .
docker run -p 8000:8000 cnn-model-deployment
```

### **3. Cloud Deployment**

#### **Render.com**
```yaml
# render.yaml
services:
  - type: web
    name: cnn-model-deployment
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

#### **Heroku**
```bash
# สร้างไฟล์ Procfile
echo "web: uvicorn app.main:app --host 0.0.0.0 --port $PORT" > Procfile

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

## Performance Tuning

### **1. Memory Optimization**
```python
# ใน app/inference.py
import gc

# หลังประมวลผล
gc.collect()
```

### **2. Model Caching**
```python
# Cache โมเดลใน memory
_model_cache = {}

def load_model():
    if 'model' not in _model_cache:
        _model_cache['model'] = keras.models.load_model(model_path)
    return _model_cache['model']
```

### **3. Batch Processing**
```python
# ประมวลผลหลายภาพพร้อมกัน
batch_size = 8
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    results = model.predict(batch)
```

## Security

### **1. File Upload Security**
```python
# ตรวจสอบไฟล์
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_file(file):
    if file.size > MAX_FILE_SIZE:
        raise ValueError("File too large")
    if not file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
        raise ValueError("Invalid file type")
```

### **2. CORS Configuration**
```python
# ใน app/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # เปลี่ยนเป็น domain จริงใน production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Monitoring

### **1. Health Check**
```python
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}
```

### **2. Metrics**
```python
# วัดประสิทธิภาพ
import time

start_time = time.time()
# ... processing ...
end_time = time.time()
logger.info(f"Processing took {end_time - start_time:.2f} seconds")
```

## Testing

### **1. Unit Tests**
```python
# test_app.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", files={"file": ("test.jpg", open("test.jpg", "rb"))})
    assert response.status_code == 200
```

### **2. Run Tests**
```bash
pip install pytest
pytest test_app.py -v
```

## Additional Resources

### **Documentation**
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Python Documentation](https://docs.python.org/)

### **Tutorials**
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [TensorFlow Tutorial](https://www.tensorflow.org/tutorials)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)

### **Community**
- [FastAPI Discord](https://discord.gg/9Z9CgHQ)
- [TensorFlow Forum](https://discuss.tensorflow.org/)
- [Python Reddit](https://www.reddit.com/r/Python/)

---

## **พร้อมใช้งานแล้ว!**

หากติดตั้งสำเร็จ คุณจะเห็นหน้าเว็บที่สวยงามพร้อมใช้งาน AI Model สำหรับตรวจจับตัวเลข 0-9 ในภาพ

**สู้ๆ ครับ!**
