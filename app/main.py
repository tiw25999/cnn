# app/main.py
import base64
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from .utils_image import bytes_to_rgb_ndarray
from .inference import predict_from_ndarray, load_model, get_current_model_info, draw_bounding_boxes

logger = logging.getLogger(__name__)

app = FastAPI(title="Image Upload & Predict")

# สร้าง thread pool สำหรับการประมวลผลแบบ async
executor = ThreadPoolExecutor(max_workers=2)

# เปิด CORS ถ้าจะยิงจากเว็บอื่น/React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # แก้ตามโดเมนจริงเพื่อความปลอดภัย
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="app/templates")

@app.on_event("startup")
def _load_model_on_startup():
    try:
        load_model()
        print("Model loaded OK.")
    except Exception as e:
        print("Model load error:", e)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/model-info")
def model_info():
    """ดูข้อมูลโมเดลที่ใช้อยู่ปัจจุบัน"""
    return JSONResponse(get_current_model_info())

@app.get("/test-bounding-boxes")
def test_bounding_boxes():
    """Test endpoint to verify bounding box drawing works"""
    try:
        from .inference import test_draw_bounding_boxes
        test_img = test_draw_bounding_boxes()
        
        # Convert to base64
        import cv2
        _, buffer = cv2.imencode('.jpg', test_img)
        b64 = base64.b64encode(buffer).decode("utf-8")
        
        return JSONResponse({
            "ok": True,
            "message": "Test bounding box image created",
            "test_image_base64": f"data:image/jpeg;base64,{b64}"
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

@app.post("/test-rectify-enhance")
async def test_rectify_enhance(file: UploadFile = File(...)):
    """Test endpoint to verify image rectification and enhancement works"""
    try:
        raw = await file.read()
        img = bytes_to_rgb_ndarray(raw)
        
        from .inference import rectify_and_enhance_image
        enhanced_img = rectify_and_enhance_image(img)
        
        # Convert to base64
        import cv2
        _, buffer = cv2.imencode('.jpg', enhanced_img)
        b64 = base64.b64encode(buffer).decode("utf-8")
        
        return JSONResponse({
            "ok": True,
            "message": "Image rectified and enhanced successfully",
            "original_size": f"{img.shape[1]}x{img.shape[0]}",
            "enhanced_size": f"{enhanced_img.shape[1]}x{enhanced_img.shape[0]}",
            "enhanced_image_base64": f"data:image/jpeg;base64,{b64}"
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

def process_image_sync(raw: bytes, filename: str):
    """ฟังก์ชันสำหรับประมวลผลภาพแบบ sync ใน thread pool"""
    try:
        img = bytes_to_rgb_ndarray(raw)
        result = predict_from_ndarray(img, enable_detection=True)
        
        # ส่งตัวอย่างภาพกลับ (base64) เผื่ออยากแสดง preview ที่ client
        b64 = base64.b64encode(raw).decode("utf-8")
        
        # สร้างภาพที่มี bounding boxes
        img_with_boxes = None
        logger.info(f"Result keys: {result.keys()}")
        logger.info(f"Result ok: {result.get('ok')}")
        logger.info(f"Has detections: {'detections' in result}")
        if "detections" in result:
            logger.info(f"Detections count: {len(result['detections'])}")
            if result["detections"]:
                logger.info(f"Sample detection: {result['detections'][0]}")
        
        if result.get("ok") and "detections" in result and len(result["detections"]) > 0:
            try:
                logger.info(f"Drawing {len(result['detections'])} bounding boxes")
                img_with_boxes = draw_bounding_boxes(img, result["detections"])
                # แปลงเป็น base64
                import cv2
                _, buffer = cv2.imencode('.jpg', img_with_boxes)
                b64_with_boxes = base64.b64encode(buffer).decode("utf-8")
                logger.info("Successfully created image with bounding boxes")
            except Exception as e:
                logger.warning(f"Failed to draw bounding boxes: {e}")
                b64_with_boxes = b64
        else:
            logger.info("No detections found, using original image")
            b64_with_boxes = b64
        
        # รวมข้อมูลโมเดลเข้าไปใน response
        response_data = {
            "ok": True,
            "filename": filename,
            "preview_base64": f"data:image/jpeg;base64,{b64}",
            "preview_with_boxes_base64": f"data:image/jpeg;base64,{b64_with_boxes}",
            "result": result
        }
        
        # ถ้า result มี model_info ให้เพิ่มเข้าไปใน response หลักด้วย
        if result.get("ok") and "model_info" in result:
            response_data["model_info"] = result["model_info"]
            
        return response_data
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        logger.info(f"Processing image: {file.filename}, size: {len(raw)} bytes")
        
        # ตรวจสอบขนาดไฟล์
        if len(raw) > 10 * 1024 * 1024:  # มากกว่า 10MB
            return JSONResponse({
                "ok": False, 
                "error": "ไฟล์ใหญ่เกินไป กรุณาใช้ไฟล์ขนาดไม่เกิน 10MB"
            }, status_code=400)
        
        # ประมวลผลใน thread pool เพื่อไม่ให้ UI ค้าง
        loop = asyncio.get_event_loop()
        
        # กำหนด timeout ตามขนาดไฟล์
        timeout_seconds = 30  # default timeout
        if len(raw) > 5 * 1024 * 1024:  # มากกว่า 5MB
            timeout_seconds = 60
        elif len(raw) > 2 * 1024 * 1024:  # มากกว่า 2MB
            timeout_seconds = 45
        
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(executor, process_image_sync, raw, file.filename),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.warning(f"Processing timeout after {timeout_seconds} seconds for file: {file.filename}")
            return JSONResponse({
                "ok": False, 
                "error": f"การประมวลผลใช้เวลานานเกินไป (เกิน {timeout_seconds} วินาที) กรุณาลองใช้ภาพขนาดเล็กกว่า"
            }, status_code=408)
        
        if result.get("ok"):
            return JSONResponse(result)
        else:
            return JSONResponse(result, status_code=400)
            
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
