import os
import shutil
import threading
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List

router = APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

retraining_status = {
    "is_training": False,
    "last_status": "idle",
    "last_accuracy": None,
    "last_trained_at": None,
    "images_uploaded": 0
}


@router.post("/upload")
async def upload_images(
    label: str,
    files: List[UploadFile] = File(...)
):
    """
    Upload multiple images for a given disease class label.
    Images are saved to disk for retraining.
    """
    if not label:
        raise HTTPException(status_code=400, detail="Label is required")

    label_dir = os.path.join(UPLOAD_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    saved = []
    for file in files:
        if not file.content_type.startswith("image/"):
            continue
        dest = os.path.join(label_dir, file.filename)
        with open(dest, "wb") as f:
            content = await file.read()
            f.write(content)
        saved.append(file.filename)

    retraining_status["images_uploaded"] += len(saved)

    return {
        "message": f"{len(saved)} images uploaded for class '{label}'",
        "label": label,
        "saved_files": saved,
        "total_uploaded": retraining_status["images_uploaded"]
    }


def run_retraining(model_path, upload_dir):
    """
    Run retraining in a background thread so the API stays responsive.
    """
    retraining_status["is_training"] = True
    retraining_status["last_status"] = "training"

    try:
        from src.model import retrain
        history = retrain(upload_dir, model_path=model_path, epochs=5)

        val_acc = history.history.get("val_accuracy", [None])[-1]
        retraining_status["last_accuracy"] = round(float(val_acc) * 100, 2) if val_acc else None
        retraining_status["last_status"] = "completed"

        import datetime
        retraining_status["last_trained_at"] = datetime.datetime.utcnow().isoformat()

    except Exception as e:
        retraining_status["last_status"] = f"failed: {str(e)}"

    finally:
        retraining_status["is_training"] = False


@router.post("/retrain")
async def trigger_retrain(request: Request):
    """
    Trigger model retraining on the uploaded images.
    Retraining runs in background — check /retrain/status for progress.
    """
    if retraining_status["is_training"]:
        raise HTTPException(status_code=409, detail="Retraining is already in progress")

    class_dirs = [
        d for d in os.listdir(UPLOAD_DIR)
        if os.path.isdir(os.path.join(UPLOAD_DIR, d))
    ]

    if not class_dirs:
        raise HTTPException(
            status_code=400,
            detail="No uploaded images found. Upload images first using /upload"
        )

    model_path = request.app.state.model_path

    thread = threading.Thread(
        target=run_retraining,
        args=(model_path, UPLOAD_DIR),
        daemon=True
    )
    thread.start()

    return {
        "message": "Retraining started in background",
        "classes_found": class_dirs,
        "check_status": "/retrain/status"
    }


@router.get("/retrain/status")
async def retrain_status():
    """
    Check the current status of retraining.
    """
    return retraining_status
