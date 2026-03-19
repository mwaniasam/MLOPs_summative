import time
from fastapi import APIRouter, Request

router = APIRouter()

prediction_log = []


@router.get("/metrics")
async def get_metrics(request: Request):
    """
    Return model uptime, performance metrics, and prediction statistics.
    """
    uptime_seconds = time.time() - request.app.state.start_time
    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    seconds = int(uptime_seconds % 60)

    return {
        "model": "CoffeeGuard — Arabica Coffee Leaf Disease Classifier",
        "version": "1.0.0",
        "uptime": f"{hours}h {minutes}m {seconds}s",
        "uptime_seconds": round(uptime_seconds, 2),
        "status": "online",
        "model_path": request.app.state.model_path,
        "performance": {
            "overall_accuracy": "98.80%",
            "macro_f1_score": "98.36%",
            "weighted_f1_score": "98.78%",
            "macro_precision": "98.50%",
            "macro_recall": "98.31%",
            "best_val_accuracy": "99.33%"
        },
        "classes": [
            "Cerscospora",
            "Healthy",
            "Leaf rust",
            "Miner",
            "Phoma"
        ],
        "dataset": {
            "name": "Arabica Coffee Leaf Disease Dataset",
            "total_images": 58549,
            "train_images": 46836,
            "val_images": 11713,
            "classes": 5
        }
    }
