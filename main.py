#main.py
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from preprocessing import preprocess_image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mnist_cnn.keras"

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"model file not found: {MODEL_PATH}")

    app.state.model = tf.keras.models.load_model(MODEL_PATH)

    expected_shape = tuple(app.state.model.input_shape[1:])
    if expected_shape != (28,28,1):
        raise RuntimeError(f"expected model input (28,28,1) got {expected_shape}")

    yield


app = FastAPI(
    title="MNIST digit recognition API",
    description="predict a single handwritten digit from 0 to 9",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET","POST"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        raise HTTPException(status_code=415, detail="upload a JPEG, PNG or WEBP image")

    #limit upload filw to 8MB
    image_bytes = await file.read(8*1024*1024+1)

    if len(image_bytes) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be smaller than 8mb")

    try:
        model_input = preprocess_image(image_bytes)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    probabilities = app.state.model.predict(model_input,verbose=0)
    r_probabilities=probabilities[0]
    predicted_digit = int(np.argmax(r_probabilities))
    confidence = float(r_probabilities[predicted_digit])

    return {
        "digit": predicted_digit,
        "confidence": confidence,
        "probabilities": [float(value) for value in r_probabilities],
        "all predictions": probabilities.tolist(),
    }