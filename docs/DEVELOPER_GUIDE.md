# Developer Guide

## Overview

คู่มือนี้สำหรับนักพัฒนาที่ต้องการ **แก้ไข, ขยาย, หรือปรับแต่ง** ระบบ CNN Model Deployment

## Architecture Overview

### **System Architecture**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   AI Models     │
│   (HTML/CSS/JS) │◄──►│   (FastAPI)     │◄──►│   (TensorFlow)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **File Structure**
```
app/
├── main.py              # FastAPI application
├── inference.py         # AI model logic
└── templates/
    └── index.html       # Frontend UI

models/                  # AI models (.keras files)
docs/                    # Documentation
requirements.txt         # Dependencies
```

## Backend Development

### **FastAPI Application Structure**

#### **main.py**
```python
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="CNN Model Deployment")

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("templates/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Form(None)):
    # AI processing logic
    pass
```

#### **inference.py**
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

# Model management
_current_model = None
_current_model_path = "models/MobileNetV2_Augmented_best.keras"

def load_model():
    """Load AI model from file"""
    global _current_model
    if _current_model is None:
        _current_model = keras.models.load_model(_current_model_path)
    return _current_model

def predict_from_ndarray(img: np.ndarray, enable_detection: bool = True):
    """Main prediction function"""
    # Image preprocessing
    # Model inference
    # Post-processing
    # Return results
    pass
```

### **API Endpoints**

#### **1. POST /predict**
```python
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form(None)
):
    """
    Upload image and get prediction
    
    Args:
        file: Image file (JPG, PNG, JPEG)
        model_name: Model to use (optional)
    
    Returns:
        JSON with prediction results
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Switch model if specified
        if model_name:
            switch_model(model_name)
        
        # Get prediction
        result = predict_from_ndarray(img, enable_detection=True)
        
        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}
```

#### **2. GET /available-models**
```python
@app.get("/available-models")
async def get_available_models():
    """Get list of available models"""
    models = []
    for file in os.listdir("models/"):
        if file.endswith(".keras"):
            models.append(file)
    return {"models": models}
```

#### **3. POST /switch-model**
```python
@app.post("/switch-model")
async def switch_model_endpoint(request: ModelSwitchRequest):
    """Switch current model"""
    try:
        switch_model(request.model_name)
        return {
            "success": True,
            "message": "Model switched successfully",
            "current_model": request.model_name
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### **Model Management**

#### **Model Loading**
```python
def load_model():
    """Load model with caching"""
    global _current_model, _current_model_path
    
    if _current_model is None or _current_model_path != get_current_model_path():
        _current_model = keras.models.load_model(get_current_model_path())
        _current_model_path = get_current_model_path()
        logger.info(f"Model loaded: {_current_model_path}")
    
    return _current_model
```

#### **Model Switching**
```python
def switch_model(model_name: str):
    """Switch to different model"""
    global _current_model, _current_model_path
    
    model_path = f"models/{model_name}"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    _current_model = None  # Clear cache
    _current_model_path = model_path
    logger.info(f"Switched to model: {model_name}")
```

### **Image Processing Pipeline**

#### **1. Preprocessing**
```python
def _resize_and_scale(img: np.ndarray, size: tuple, model) -> np.ndarray:
    """Resize and normalize image for model input"""
    # Resize image
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Apply model-specific preprocessing
    model_name = get_current_model_path().split('/')[-1].replace('.keras', '')
    if 'EfficientNet' in model_name:
        img_normalized = tf.keras.applications.efficientnet.preprocess_input(img_normalized * 255.0)
    elif 'MobileNet' in model_name:
        img_normalized = tf.keras.applications.mobilenet_v2.preprocess_input(img_normalized * 255.0)
    elif 'NASNet' in model_name:
        img_normalized = tf.keras.applications.nasnet.preprocess_input(img_normalized * 255.0)
    
    return img_normalized
```

#### **2. Object Detection**
```python
def detect_objects_with_bounding_boxes(img: np.ndarray, model) -> List[Dict[str, Any]]:
    """Detect objects using vertical projection + sliding window"""
    try:
        # Try projection-based detection first
        proj_dets, process_steps = detect_by_vertical_projection(img, model, return_steps=True)
        if proj_dets:
            return proj_dets, process_steps
        else:
            # Fallback to sliding window
            return sliding_window_detection(img, model)
    except Exception as e:
        logger.warning(f"Detection failed: {e}")
        return [], {}
```

#### **3. Post-processing**
```python
def apply_nms(detections: List[Dict[str, Any]], iou_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """Apply Non-Maximum Suppression to remove duplicate detections"""
    if not detections:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # Apply NMS
    keep = []
    while detections:
        current = detections.pop(0)
        keep.append(current)
        
        # Remove overlapping detections
        detections = [det for det in detections 
                     if calculate_iou(current['bbox'], det['bbox']) < iou_threshold]
    
    return keep
```

## Frontend Development

### **HTML Structure**

#### **index.html**
```html
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CNN Model Deployment</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <!-- Header -->
    <header class="header">
        <h1>CNN Model Deployment</h1>
        <button id="dev-mode-toggle" class="dev-mode-toggle">Dev Mode</button>
    </header>

    <!-- Main Content -->
    <main class="main">
        <!-- Upload Section -->
        <section class="upload-section">
            <div class="upload-area" id="upload-area">
                <div class="upload-content">
                    <svg class="upload-icon">...</svg>
                    <h3>ลากไฟล์มาวางที่นี่</h3>
                    <p>หรือคลิกเพื่อเลือกไฟล์</p>
                </div>
                <input type="file" id="file-input" accept="image/*" hidden>
            </div>
        </section>

        <!-- Dev Mode Controls -->
        <section class="dev-controls" id="dev-controls" style="display: none;">
            <div class="model-selection">
                <label for="model-select">เลือกโมเดล:</label>
                <select id="model-select" class="model-select">
                    <option value="MobileNetV2_Augmented_best.keras">MobileNetV2</option>
                    <!-- ... other options ... -->
                </select>
                <button id="reload-model" class="reload-btn">รีโหลดโมเดล</button>
            </div>
        </section>

        <!-- Results Section -->
        <section class="results-section" id="results-section" style="display: none;">
            <!-- Image Display -->
            <div class="image-display">
                <div class="image-container">
                    <img id="original-image" alt="Original Image">
                    <img id="result-image" alt="Result Image">
                </div>
            </div>

            <!-- Detection Details -->
            <div class="detection-details">
                <h3>รายละเอียดการตรวจจับ</h3>
                <div id="detection-list"></div>
            </div>

            <!-- Dev Mode: Process Steps -->
            <div class="process-steps" id="process-steps" style="display: none;">
                <h3>ขั้นตอนการประมวลผล</h3>
                <div class="process-cards" id="process-cards"></div>
            </div>

            <!-- Dev Mode: JSON Output -->
            <div class="json-output" id="json-output" style="display: none;">
                <h3>JSON Output</h3>
                <pre id="json-display"></pre>
            </div>
        </section>
    </main>

    <script src="/static/script.js"></script>
</body>
</html>
```

### **CSS Styling**

#### **Modern Design System**
```css
:root {
    /* Colors */
    --primary-blue: #2563eb;
    --secondary-blue: #3b82f6;
    --light-blue: #dbeafe;
    --white: #ffffff;
    --gray-100: #f3f4f6;
    --gray-500: #6b7280;
    --gray-900: #111827;
    
    /* Typography */
    --font-family: 'Inter', system-ui, sans-serif;
    --font-size-sm: 0.875rem;
    --font-size-base: 1rem;
    --font-size-lg: 1.125rem;
    --font-size-xl: 1.25rem;
    --font-size-2xl: 1.5rem;
    
    /* Spacing */
    --spacing-1: 0.25rem;
    --spacing-2: 0.5rem;
    --spacing-4: 1rem;
    --spacing-6: 1.5rem;
    --spacing-8: 2rem;
    
    /* Border Radius */
    --radius-sm: 0.375rem;
    --radius-md: 0.5rem;
    --radius-lg: 0.75rem;
    --radius-xl: 1rem;
    
    /* Shadows */
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

/* Base Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: var(--font-family);
    line-height: 1.6;
    color: var(--gray-900);
    background: linear-gradient(135deg, var(--light-blue) 0%, var(--white) 100%);
    min-height: 100vh;
}

/* Header */
.header {
    background: var(--white);
    box-shadow: var(--shadow-sm);
    padding: var(--spacing-4) var(--spacing-6);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.header h1 {
    font-size: var(--font-size-2xl);
    font-weight: 700;
    color: var(--primary-blue);
}

/* Upload Area */
.upload-area {
    border: 2px dashed var(--primary-blue);
    border-radius: var(--radius-lg);
    padding: var(--spacing-8);
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    background: var(--white);
}

.upload-area:hover {
    border-color: var(--secondary-blue);
    background: var(--light-blue);
}

.upload-area.dragover {
    border-color: var(--secondary-blue);
    background: var(--light-blue);
    transform: scale(1.02);
}

/* Cards */
.card {
    background: var(--white);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    padding: var(--spacing-6);
    margin-bottom: var(--spacing-4);
}

/* Buttons */
.btn {
    background: var(--primary-blue);
    color: var(--white);
    border: none;
    border-radius: var(--radius-md);
    padding: var(--spacing-2) var(--spacing-4);
    font-size: var(--font-size-sm);
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
}

.btn:hover {
    background: var(--secondary-blue);
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}

/* Responsive Design */
@media (max-width: 768px) {
    .header {
        padding: var(--spacing-4);
    }
    
    .header h1 {
        font-size: var(--font-size-xl);
    }
    
    .upload-area {
        padding: var(--spacing-6);
    }
}
```

### **JavaScript Functionality**

#### **Main Application Logic**
```javascript
class CNNModelApp {
    constructor() {
        this.devMode = false;
        this.currentModel = 'MobileNetV2_Augmented_best.keras';
        this.performanceMonitor = new PerformanceMonitor();
        
        this.initializeEventListeners();
        this.loadAvailableModels();
    }

    initializeEventListeners() {
        // File upload
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Dev mode toggle
        document.getElementById('dev-mode-toggle').addEventListener('click', this.toggleDevMode.bind(this));
        
        // Model selection
        document.getElementById('model-select').addEventListener('change', this.handleModelChange.bind(this));
        document.getElementById('reload-model').addEventListener('click', this.reloadModel.bind(this));
    }

    async handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            await this.processFile(file);
        }
    }

    async processFile(file) {
        try {
            this.showProgress();
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('model_name', this.currentModel);
            
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.ok) {
                this.displayResults(result);
            } else {
                this.showError(result.error);
            }
        } catch (error) {
            this.showError('เกิดข้อผิดพลาดในการประมวลผล');
        } finally {
            this.hideProgress();
        }
    }

    displayResults(result) {
        // Display images
        document.getElementById('original-image').src = result.preview_base64;
        document.getElementById('result-image').src = result.preview_with_boxes_base64;
        
        // Display detections
        this.displayDetections(result.detections);
        
        // Display process steps (Dev Mode)
        if (this.devMode && result.process_steps) {
            this.displayProcessSteps(result.process_steps);
        }
        
        // Display JSON (Dev Mode)
        if (this.devMode) {
            this.displayJSON(result);
        }
        
        // Show results section
        document.getElementById('results-section').style.display = 'block';
    }

    displayDetections(detections) {
        const detectionList = document.getElementById('detection-list');
        detectionList.innerHTML = '';
        
        detections.forEach((detection, index) => {
            const position = detection.position || (index + 1);
            const detectionHTML = `
                <div class="detection-item">
                    <div>
                        <span class="detection-class">เลข ${detection.class_name}</span>
                        <div style="font-size: 0.8em; color: #666;">
                            ตำแหน่ง: ${position}
                        </div>
                        <div style="font-size: 0.7em; color: #999;">
                            พิกัด: (${detection.bbox[0]}, ${detection.bbox[1]}) ถึง (${detection.bbox[2]}, ${detection.bbox[3]})
                        </div>
                    </div>
                    <div class="detection-confidence">${(detection.confidence * 100).toFixed(1)}%</div>
                </div>
            `;
            detectionList.innerHTML += detectionHTML;
        });
    }

    toggleDevMode() {
        this.devMode = !this.devMode;
        const devControls = document.getElementById('dev-controls');
        const processSteps = document.getElementById('process-steps');
        const jsonOutput = document.getElementById('json-output');
        
        if (this.devMode) {
            devControls.style.display = 'block';
            processSteps.style.display = 'block';
            jsonOutput.style.display = 'block';
        } else {
            devControls.style.display = 'none';
            processSteps.style.display = 'none';
            jsonOutput.style.display = 'none';
        }
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    new CNNModelApp();
});
```

#### **Performance Monitoring**
```javascript
class PerformanceMonitor {
    constructor() {
        this.startTime = performance.now();
        this.metrics = {};
    }

    logPageLoad() {
        const loadTime = performance.now() - this.startTime;
        console.log(`🚀 Page loaded in ${loadTime.toFixed(2)}ms`);
        
        // Core Web Vitals
        this.logCoreWebVitals();
    }

    logAPICall(url, duration) {
        console.log(`📡 API ${url} completed in ${duration.toFixed(2)}ms`);
        
        if (duration > 10000) {
            console.warn('🐌 Very slow API performance - consider optimization');
        }
    }

    logImageLoad(duration) {
        console.log(`🖼️ Image loaded in ${duration.toFixed(2)}ms`);
        
        if (duration > 5000) {
            console.warn('🚨 Very slow image load - needs optimization!');
        }
    }

    logCoreWebVitals() {
        const navigation = performance.getEntriesByType('navigation')[0];
        const dns = navigation.domainLookupEnd - navigation.domainLookupStart;
        const tcp = navigation.connectEnd - navigation.connectStart;
        const domContentLoaded = navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart;
        const totalLoad = navigation.loadEventEnd - navigation.loadEventStart;
        
        console.log('📊 Performance Metrics:');
        console.log(`   - DNS Lookup: ${dns.toFixed(2)}ms`);
        console.log(`   - TCP Connect: ${tcp.toFixed(2)}ms`);
        console.log(`   - DOM Content Loaded: ${domContentLoaded.toFixed(2)}ms`);
        console.log(`   - Total Load Time: ${totalLoad.toFixed(2)}ms`);
        
        // Performance Score
        let score = 'EXCELLENT';
        if (totalLoad > 3000) score = 'GOOD';
        if (totalLoad > 5000) score = 'FAIR';
        if (totalLoad > 10000) score = 'NEEDS IMPROVEMENT';
        
        console.log(`🏆 Performance Score: ${score} (${totalLoad.toFixed(2)}ms)`);
    }
}
```

## Testing

### **Unit Tests**
```python
# test_app.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict():
    """Test prediction endpoint"""
    with open("test_image.jpg", "rb") as f:
        response = client.post("/predict", files={"file": f})
    
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] == True
    assert "detections" in data

def test_available_models():
    """Test available models endpoint"""
    response = client.get("/available-models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) > 0

def test_switch_model():
    """Test model switching"""
    response = client.post("/switch-model", json={"model_name": "EfficientNetB0_Augmented_best.keras"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
```

### **Integration Tests**
```python
# test_integration.py
def test_full_pipeline():
    """Test complete prediction pipeline"""
    # Load test image
    img = cv2.imread("test_image.jpg")
    
    # Test prediction
    result = predict_from_ndarray(img, enable_detection=True)
    
    assert result["ok"] == True
    assert "detections" in result
    assert "process_steps" in result
    assert len(result["detections"]) > 0
```

### **Performance Tests**
```python
# test_performance.py
import time

def test_prediction_speed():
    """Test prediction speed"""
    img = cv2.imread("test_image.jpg")
    
    start_time = time.time()
    result = predict_from_ndarray(img, enable_detection=True)
    end_time = time.time()
    
    duration = end_time - start_time
    assert duration < 10.0  # Should complete within 10 seconds
    print(f"Prediction took {duration:.2f} seconds")
```

## Deployment

### **Docker Deployment**
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **Docker Compose**
```yaml
# docker-compose.yml
version: '3.8'

services:
  cnn-app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped
```

### **Production Deployment**
```bash
# Build and run
docker build -t cnn-model-deployment .
docker run -d -p 8000:8000 --name cnn-app cnn-model-deployment

# Or with docker-compose
docker-compose up -d
```

## Configuration

### **Environment Variables**
```bash
# .env
MODEL_PATH=models/
MAX_FILE_SIZE=10485760
ALLOWED_EXTENSIONS=jpg,jpeg,png
DEBUG=False
LOG_LEVEL=INFO
```

### **Model Configuration**
```python
# config.py
MODEL_CONFIG = {
    "MobileNetV2_Augmented_best.keras": {
        "input_size": 224,
        "preprocessing": "mobilenet_v2",
        "description": "Fast and efficient model"
    },
    "EfficientNetB0_Augmented_best.keras": {
        "input_size": 224,
        "preprocessing": "efficientnet",
        "description": "Most accurate model"
    }
}
```

## Monitoring & Logging

### **Logging Configuration**
```python
# logging_config.py
import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
```

### **Health Check**
```python
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check model loading
        model = load_model()
        
        # Check model info
        model_info = get_current_model_info()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model": model_info["model_name"],
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
```

## Best Practices

### **Code Organization**
- **Separation of Concerns** - แยก logic แต่ละส่วน
- **Error Handling** - จัดการ error อย่างเหมาะสม
- **Logging** - ใช้ logging แทน print
- **Type Hints** - ใช้ type hints สำหรับ Python

### **Performance Optimization**
- **Model Caching** - Cache โมเดลใน memory
- **Batch Processing** - ประมวลผลหลายภาพพร้อมกัน
- **Image Optimization** - ปรับขนาดภาพให้เหมาะสม
- **Memory Management** - ใช้ gc.collect() หลังประมวลผล

### **Security**
- **File Validation** - ตรวจสอบไฟล์ที่อัปโหลด
- **Input Sanitization** - ทำความสะอาด input
- **Rate Limiting** - จำกัดการเรียก API
- **CORS Configuration** - ตั้งค่า CORS ให้เหมาะสม

---

## **พร้อมพัฒนาแล้ว!**

ตอนนี้คุณสามารถแก้ไขและขยายระบบได้แล้ว!

**หากมีคำถามหรือต้องการความช่วยเหลือ สามารถติดต่อได้ที่ [GitHub Issues](https://github.com/yourusername/cnn-model-deployment/issues)**

**Happy Coding!**
