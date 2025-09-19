# app/main.py
import base64
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from .utils_image import bytes_to_rgb_ndarray
from .inference import predict_from_ndarray, load_model

app = FastAPI(title="Image Upload & Predict")

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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        img = bytes_to_rgb_ndarray(raw)
        result = predict_from_ndarray(img)

        # ส่งตัวอย่างภาพกลับ (base64) เผื่ออยากแสดง preview ที่ client
        b64 = base64.b64encode(raw).decode("utf-8")
        return JSONResponse({
            "ok": True,
            "filename": file.filename,
            "preview_base64": f"data:image/jpeg;base64,{b64}",
            "result": result
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
