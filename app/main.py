import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routes import predict, retrain, metrics
from app.database import init_db

MODEL_PATH = os.getenv("MODEL_PATH", "models/coffeeguard_model.h5")
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    import tensorflow as tf
    print(f"Loading model from {MODEL_PATH}")
    app.state.model = tf.keras.models.load_model(MODEL_PATH)
    app.state.start_time = START_TIME
    app.state.model_path = MODEL_PATH
    print("Model loaded successfully")

    init_db()

    yield
    print("Shutting down")


app = FastAPI(
    title="CoffeeGuard API",
    description="Arabica Coffee Leaf Disease Classification API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(retrain.router)
app.include_router(metrics.router)

app.mount("/static", StaticFiles(directory="app/frontend"), name="static")


@app.get("/")
async def root():
    return FileResponse("app/frontend/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}
