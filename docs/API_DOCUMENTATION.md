# API Documentation

## Overview

API นี้ใช้สำหรับ **CNN Model Deployment** ที่สามารถตรวจจับและจำแนกตัวเลข 0-9 ในภาพได้

**Base URL:** `http://localhost:8000`  
**Framework:** FastAPI  
**Version:** 1.0.0

## Endpoints

### 1. **POST /predict**
อัปโหลดภาพและทำนายผล

#### **Request**
```http
POST /predict
Content-Type: multipart/form-data
```

**Parameters:**
- `file` (file, required) - ไฟล์ภาพ (JPG, PNG, JPEG)
- `model_name` (string, optional) - ชื่อโมเดลที่ต้องการใช้

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg" \
  -F "model_name=MobileNetV2_Augmented_best.keras"
```

#### **Response**
```json
{
  "ok": true,
  "filename": "image.jpg",
  "preview_base64": "data:image/jpeg;base64,/9j/4AAQ...",
  "preview_with_boxes_base64": "data:image/jpeg;base64,/9j/4AAQ...",
  "result": {
    "probabilities": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "top1": {
      "index": 9,
      "prob": 0.95
    }
  },
  "detections": [
    {
      "bbox": [32, 33, 118, 197],
      "class_id": 7,
      "confidence": 0.95,
      "class_name": "7",
      "position": 1
    }
  ],
  "process_steps": {
    "band_crop": "data:image/jpeg;base64,/9j/4AAQ...",
    "threshold": "data:image/jpeg;base64,/9j/4AAQ...",
    "projection": "data:image/jpeg;base64,/9j/4AAQ...",
    "boxes": "data:image/jpeg;base64,/9j/4AAQ...",
    "crop": "data:image/jpeg;base64,/9j/4AAQ...",
    "square_pad": "data:image/jpeg;base64,/9j/4AAQ...",
    "resize244": "data:image/jpeg;base64,/9j/4AAQ...",
    "grayscale_rgb": "data:image/jpeg;base64,/9j/4AAQ..."
  },
  "detection_stats": {
    "total_objects": 1,
    "unique_classes": 1,
    "class_counts": {
      "7": 1
    }
  },
  "model_info": {
    "name": "MobileNetV2_Augmented",
    "path": "models/MobileNetV2_Augmented_best.keras",
    "size_mb": 15.2
  }
}
```

### 2. **GET /available-models**
ดูรายการโมเดลที่มี

#### **Request**
```http
GET /available-models
```

#### **Response**
```json
{
  "models": [
    "MobileNetV2_Augmented_best.keras",
    "EfficientNetB0_Augmented_best.keras",
    "EfficientNetV2B0_Original_best.keras",
    "MobileNetV3Large_Augmented_best.keras",
    "NASNetMobile_Augmented_best.keras"
  ]
}
```

### 3. **POST /switch-model**
เปลี่ยนโมเดลที่ใช้งาน

#### **Request**
```http
POST /switch-model
Content-Type: application/json
```

**Body:**
```json
{
  "model_name": "EfficientNetB0_Augmented_best.keras"
}
```

#### **Response**
```json
{
  "success": true,
  "message": "Model switched successfully",
  "current_model": "EfficientNetB0_Augmented_best.keras"
}
```

### 4. **GET /model-info**
ดูข้อมูลโมเดลปัจจุบัน

#### **Request**
```http
GET /model-info
```

#### **Response**
```json
{
  "model_name": "MobileNetV2_Augmented",
  "model_path": "models/MobileNetV2_Augmented_best.keras",
  "file_size_mb": 15.2,
  "input_size": 224,
  "num_classes": 10
}
```

### 5. **GET /**
หน้าเว็บหลัก

#### **Request**
```http
GET /
```

#### **Response**
```html
<!DOCTYPE html>
<html>
<head>
    <title>CNN Model Deployment</title>
    <!-- ... -->
</head>
<body>
    <!-- ... -->
</body>
</html>
```

## Response Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `400` | Bad Request |
| `422` | Validation Error |
| `500` | Internal Server Error |

## Error Responses

### **400 Bad Request**
```json
{
  "ok": false,
  "error": "Invalid file format. Please upload JPG, PNG, or JPEG files only."
}
```

### **422 Validation Error**
```json
{
  "detail": [
    {
      "loc": ["body", "file"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### **500 Internal Server Error**
```json
{
  "ok": false,
  "error": "Model loading failed: File not found"
}
```

## Data Models

### **Detection Object**
```typescript
interface Detection {
  bbox: [number, number, number, number];  // [x1, y1, x2, y2]
  class_id: number;                        // 0-9
  confidence: number;                      // 0.0-1.0
  class_name: string;                      // "0"-"9"
  position: number;                        // 1-10
}
```

### **Model Info**
```typescript
interface ModelInfo {
  model_name: string;
  model_path: string;
  file_size_mb: number;
  input_size: number;
  num_classes: number;
}
```

### **Process Steps**
```typescript
interface ProcessSteps {
  band_crop: string;        // Base64 image
  threshold: string;        // Base64 image
  projection: string;       // Base64 image
  boxes: string;           // Base64 image
  crop: string;            // Base64 image
  square_pad: string;      // Base64 image
  resize244: string;       // Base64 image
  grayscale_rgb: string;   // Base64 image
}
```

## Usage Examples

### **Python Example**
```python
import requests

# อัปโหลดภาพ
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    data = {'model_name': 'MobileNetV2_Augmented_best.keras'}
    response = requests.post('http://localhost:8000/predict', files=files, data=data)
    
result = response.json()
print(f"Found {len(result['detections'])} objects")
```

### **JavaScript Example**
```javascript
// อัปโหลดภาพ
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('model_name', 'MobileNetV2_Augmented_best.keras');

fetch('/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Detections:', data.detections);
});
```

### **cURL Example**
```bash
# อัปโหลดภาพ
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg" \
  -F "model_name=MobileNetV2_Augmented_best.keras"

# ดูรายการโมเดล
curl -X GET "http://localhost:8000/available-models"

# เปลี่ยนโมเดล
curl -X POST "http://localhost:8000/switch-model" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "EfficientNetB0_Augmented_best.keras"}'
```

## Configuration

### **Environment Variables**
```bash
# Optional
MODEL_PATH=models/
MAX_FILE_SIZE=10485760  # 10MB
ALLOWED_EXTENSIONS=jpg,jpeg,png
```

### **Model Configuration**
```python
# app/inference.py
MODEL_CONFIG = {
    "MobileNetV2_Augmented_best.keras": {
        "input_size": 224,
        "preprocessing": "mobilenet_v2"
    },
    "EfficientNetB0_Augmented_best.keras": {
        "input_size": 224,
        "preprocessing": "efficientnet"
    }
}
```

## Performance

### **Response Times**
- **Small images (< 1MB):** ~2-5 seconds
- **Medium images (1-5MB):** ~5-10 seconds
- **Large images (> 5MB):** ~10-20 seconds

### **Memory Usage**
- **Base memory:** ~200MB
- **Per model:** ~50-100MB
- **Per image:** ~10-50MB

## Troubleshooting

### **Common Issues**

1. **Model not found**
   - ตรวจสอบไฟล์โมเดลในโฟลเดอร์ `models/`
   - ตรวจสอบชื่อไฟล์ให้ถูกต้อง

2. **Out of memory**
   - ลดขนาดภาพก่อนอัปโหลด
   - ใช้โมเดลขนาดเล็กกว่า

3. **Slow performance**
   - ใช้ GPU acceleration
   - ปรับ batch size

### **Debug Mode**
```python
# เปิด debug mode
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Changelog

### **v1.0.0** (2024-01-01)
- Initial release
- 5 model support
- Dev Mode
- Performance monitoring
- Modern UI

---

**Support:** หากมีปัญหาหรือคำถาม สามารถติดต่อได้ที่ [GitHub Issues](https://github.com/yourusername/cnn-model-deployment/issues)
