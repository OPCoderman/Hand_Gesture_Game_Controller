import os
import time
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import xgboost as xgb
from .model_utils import predict_hand_sign
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
# Metrics
# Metrics with correct labels you use in your middleware
REQUEST_COUNT = Counter('hand_sign_requests_total', 'Total classification requests', ['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Histogram('hand_sign_request_latency_seconds', 'Request latency', ['endpoint'])
PREDICTION_COUNT = Counter('hand_sign_prediction_total', 'Count of predictions per class', ['class'])




app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
booster = xgb.Booster()
booster.load_model("src/model.model")

# Input model
class LandmarkInput(BaseModel):
    landmarks: List[List[float]]

@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/predict")
def predict(data: LandmarkInput):
    landmarks = data.landmarks

    # Manual validation
    if len(landmarks) != 21:
        raise HTTPException(status_code=400, detail="Expected exactly 21 landmarks.")
    if not all(len(lm) == 2 for lm in landmarks):
        raise HTTPException(status_code=400, detail="Each landmark must have exactly 2 values (x, y).")


    hand_sign = predict_hand_sign(landmarks)
    PREDICTION_COUNT.labels(**{'class': hand_sign}).inc()
    return {"hand_sign": hand_sign}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    endpoint = request.url.path
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(process_time)
    REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, http_status=str(response.status_code)).inc()

    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
