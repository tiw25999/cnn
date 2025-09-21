# CNN Model Deployment - Web Application

## เกี่ยวกับโปรเจค

โปรเจคนี้เป็น **Web Application สำหรับ Deploy CNN Model** ที่สามารถตรวจจับและจำแนกตัวเลข 0-9 ในภาพได้ โดยใช้เทคโนโลยี **FastAPI** และ **TensorFlow** พร้อมด้วย UI ที่สวยงามและใช้งานง่าย

## Features หลัก

### **AI Detection**
- **ตรวจจับตัวเลข 0-9** ในภาพอัตโนมัติ
- **แสดงตำแหน่ง 1-10** ของตัวเลขที่พบ
- **ความแม่นยำสูง** ด้วยโมเดลที่เทรนมาแล้ว
- **ประมวลผลเร็ว** < 10 วินาทีต่อภาพ

### **User Interface**
- **Modern Design** - ดีไซน์สวยงาม ทันสมัย
- **Blue Theme** - ธีมสีน้ำเงิน-ขาว ดูเป็นทางการ
- **Responsive** - รองรับทุกขนาดหน้าจอ
- **Drag & Drop** - ลากไฟล์มาวางได้เลย

### **Developer Features**
- **Dev Mode** - โหมดสำหรับนักพัฒนา
- **Model Switching** - เปลี่ยนโมเดลได้ 5 แบบ
- **Process Steps** - แสดงขั้นตอนการประมวลผล
- **Performance Monitoring** - วัดประสิทธิภาพแบบ real-time

## Architecture

### **Backend (FastAPI)**
```python
# app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI(title="CNN Model Deployment")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # AI Model Processing
    return {"result": "prediction"}
```

### **Frontend (HTML/CSS/JavaScript)**
- **Modern UI** - Minimalist + Creative Design
- **Real-time Processing** - แสดงผลทันที
- **Performance Monitoring** - วัดความเร็ว
- **Dev Mode Toggle** - สลับโหมดได้

### **AI Model Integration**
```python
# app/inference.py
import tensorflow as tf
from tensorflow import keras

def load_model():
    model = keras.models.load_model("models/MobileNetV2_Augmented_best.keras")
    return model
```

## การติดตั้งและใช้งาน

### **1. ติดตั้ง Dependencies**
```bash
pip install -r requirements.txt
```

### **2. รันแอปพลิเคชัน**
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### **3. เปิดเว็บเบราว์เซอร์**
```
http://localhost:8000
```

## โครงสร้างโปรเจค

```
.
├── app/
│   ├── main.py              # FastAPI application
│   ├── inference.py         # AI model logic
│   └── templates/
│       └── index.html       # Frontend UI
├── models/
│   ├── MobileNetV2_Augmented_best.keras (Default)
│   ├── EfficientNetB0_Augmented_best.keras
│   ├── EfficientNetV2B0_Original_best.keras
│   ├── MobileNetV3Large_Augmented_best.keras
│   └── NASNetMobile_Augmented_best.keras
├── requirements.txt         # Dependencies
├── render.yaml             # Deployment config
└── README.md               # This file
```

## วิธีการใช้งาน

### **สำหรับผู้ใช้ทั่วไป:**
1. **อัปโหลดภาพ** - ลากไฟล์มาวางหรือคลิกเลือกไฟล์
2. **รอผลลัพธ์** - ระบบจะประมวลผลอัตโนมัติ
3. **ดูผลลัพธ์** - แสดงตัวเลขที่พบและตำแหน่ง

### **สำหรับนักพัฒนา:**
1. **เปิด Dev Mode** - คลิกปุ่ม "Dev Mode" มุมขวาบน
2. **เลือกโมเดล** - เปลี่ยนโมเดลได้ 5 แบบ
3. **ดู Process Steps** - ดูขั้นตอนการประมวลผล
4. **ดู JSON Output** - ดูข้อมูลดิบแบบ JSON

## Models ที่รองรับ

| Model | Type | Input Size | Performance |
|-------|------|------------|-------------|
| **MobileNetV2** (Default) | Augmented | 224×224 | Fast |
| **EfficientNetB0** | Augmented | 224×224 | Accurate |
| **EfficientNetV2B0** | Original | 224×224 | Latest |
| **MobileNetV3Large** | Augmented | 224×224 | Fastest |
| **NASNetMobile** | Augmented | 224×224 | Smart |

## Performance Results

```
Page loaded in 19.10ms
Performance Score: EXCELLENT
API /predict completed in 6.9s
Image loaded in 12-26ms
```

## Deployment Options

### **1. Render (แนะนำ)**
```yaml
# render.yaml
services:
  - type: web
    name: cnn-model-deployment
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### **2. Other Options:**
- **Heroku** - Easy deployment
- **AWS EC2** - Scalable cloud
- **Google Cloud Run** - Serverless
- **Docker** - Containerized deployment

## Technical Details

### **Web Framework: FastAPI**
- **Performance สูง** - เร็วที่สุดใน Python frameworks
- **Type Hints** - รองรับ TypeScript-style type checking
- **Auto Documentation** - สร้าง API docs อัตโนมัติ
- **Modern Python** - ใช้ Python 3.6+ features

### **AI Framework: TensorFlow**
- **Deep Learning** - CNN models สำหรับ image classification
- **Model Loading** - โหลดโมเดล .keras files
- **Preprocessing** - ปรับขนาดและ normalize ภาพ
- **Batch Processing** - ประมวลผลหลายภาพพร้อมกัน

### **Frontend Technologies**
- **HTML5** - Semantic markup
- **CSS3** - Modern styling with animations
- **JavaScript ES6+** - Modern JavaScript features
- **Responsive Design** - Mobile-first approach

## Design System

### **Color Palette**
- **Primary Blue:** `#2563eb` (Blue-600)
- **Secondary Blue:** `#3b82f6` (Blue-500)
- **Light Blue:** `#dbeafe` (Blue-100)
- **White:** `#ffffff`
- **Gray:** `#6b7280` (Gray-500)

### **Typography**
- **Font Family:** Inter, system-ui, sans-serif
- **Headings:** 700 weight
- **Body:** 400 weight
- **Code:** 500 weight

### **Components**
- **Cards** - Rounded corners, subtle shadows
- **Buttons** - Hover effects, smooth transitions
- **Forms** - Clean inputs, focus states
- **Icons** - Professional, consistent style

## Troubleshooting

### **ปัญหาที่พบบ่อย:**

1. **ภาพไม่ขึ้น**
   - ตรวจสอบขนาดไฟล์ (แนะนำ < 10MB)
   - ตรวจสอบรูปแบบไฟล์ (รองรับ JPG, PNG, JPEG)

2. **โมเดลไม่โหลด**
   - ตรวจสอบไฟล์โมเดลในโฟลเดอร์ `models/`
   - ตรวจสอบ dependencies ใน `requirements.txt`

3. **Performance ช้า**
   - ลดขนาดภาพก่อนอัปโหลด
   - ใช้โมเดล MobileNetV2 สำหรับความเร็ว

## API Documentation

### **Endpoints:**

#### `POST /predict`
อัปโหลดภาพและทำนายผล
```json
{
  "file": "image_file",
  "model_name": "MobileNetV2_Augmented_best.keras"
}
```

#### `GET /available-models`
ดูรายการโมเดลที่มี
```json
{
  "models": [
    "MobileNetV2_Augmented_best.keras",
    "EfficientNetB0_Augmented_best.keras"
  ]
}
```

#### `POST /switch-model`
เปลี่ยนโมเดล
```json
{
  "model_name": "EfficientNetB0_Augmented_best.keras"
}
```

## Contributing

1. Fork โปรเจค
2. สร้าง feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit การเปลี่ยนแปลง (`git commit -m 'Add some AmazingFeature'`)
4. Push ไปยัง branch (`git push origin feature/AmazingFeature`)
5. เปิด Pull Request

## License

โปรเจคนี้ใช้ MIT License - ดูรายละเอียดใน [LICENSE](LICENSE) file

## Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## Acknowledgments

- **FastAPI** - สำหรับ web framework ที่ยอดเยี่ยม
- **TensorFlow** - สำหรับ AI/ML capabilities
- **Inter Font** - สำหรับ typography ที่สวยงาม
- **Heroicons** - สำหรับ icons ที่เป็นทางการ

---

## **Workshop 9: Deploy Model to Web App**

โปรเจคนี้เป็นส่วนหนึ่งของ **Workshop 9** ที่สอนการ Deploy AI Model ไปยัง Web Application

### **สิ่งที่ได้เรียนรู้:**
- **Web Framework** - FastAPI, HTML, CSS, JavaScript
- **AI Integration** - TensorFlow, Model Loading, Preprocessing
- **UI/UX Design** - Modern, Responsive, User-friendly
- **Performance** - Optimization, Monitoring, Caching
- **Deployment** - Cloud platforms, Production ready

### **เหมาะสำหรับ:**
- **นักศึกษา** - เรียนรู้การ Deploy AI Model
- **นักพัฒนา** - ใช้เป็น template สำหรับโปรเจค
- **บริษัท** - ใช้ในงานจริง

**สู้ๆ ครับ!**
