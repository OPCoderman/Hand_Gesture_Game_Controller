from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import xgboost as xgb
from .model_utils import predict_hand_sign
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Metrics
REQUEST_COUNT = Counter("app_request_count", "Total number of requests", ["method", "endpoint", "http_status"])
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Request latency in seconds", ["endpoint"])

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

@app.post("/predict")
def predict(data: LandmarkInput):
    landmarks = data.landmarks

    # Manual validation
    if len(landmarks) != 21:
        raise HTTPException(status_code=400, detail="Expected exactly 21 landmarks.")
    if not all(len(lm) == 2 for lm in landmarks):
        raise HTTPException(status_code=400, detail="Each landmark must have exactly 2 values (x, y).")

    hand_sign = predict_hand_sign(landmarks)
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
    REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, http_status=response.status_code).inc()

    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

app.mount("/", StaticFiles(directory="static", html=True), name="static")