# Dockerfile
FROM python:3.11-slim

# libgl สำหรับ OpenCV; clean ให้บาง
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ติดตั้ง dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ด
COPY app ./app
COPY models ./models

# เปิดพอร์ต
EXPOSE 10000

# Render จะตั้ง PORT เป็น env เสมอ ใช้ $PORT ถ้าไม่มีให้ fallback 10000
ENV PORT=10000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
