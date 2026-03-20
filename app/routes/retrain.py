import os
import threading
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import List

from app.database import (
    log_upload,
    get_pending_uploads,
    log_retrain_start,
    log_retrain_complete,
    log_retrain_failed,
    get_retrain_history
)

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
    Images are saved to disk and recorded in the database.
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
        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)

        log_upload(
            filename=file.filename,
            label=label,
            file_path=dest
        )
        saved.append(file.filename)

    retraining_status["images_uploaded"] += len(saved)

    return {
        "message": f"{len(saved)} images uploaded for class '{label}' and saved to database",
        "label": label,
        "saved_files": saved,
        "total_uploaded": retraining_status["images_uploaded"]
    }


@router.get("/uploads")
async def list_uploads():
    """
    Return all uploaded images recorded in the database.
    """
    return {"uploads": get_pending_uploads()}


def run_retraining(model_path, upload_dir, retrain_id):
    """
    Run retraining in a background thread.
    Logs progress and results to the database.
    """
    retraining_status["is_training"] = True
    retraining_status["last_status"] = "training"

    try:
        from src.model import retrain
        history = retrain(upload_dir, model_path=model_path, epochs=5)

        val_acc = history.history.get("accuracy", [None])[-1]
        accuracy = round(float(val_acc) * 100, 2) if val_acc else None
        images_used = sum(
            len(os.listdir(os.path.join(upload_dir, d)))
            for d in os.listdir(upload_dir)
            if os.path.isdir(os.path.join(upload_dir, d))
        )

        log_retrain_complete(
            retrain_id=retrain_id,
            accuracy=accuracy,
            images_used=images_used,
            epochs=5
        )

        retraining_status["last_accuracy"] = accuracy
        retraining_status["last_status"] = "completed"

        import datetime
        retraining_status["last_trained_at"] = datetime.datetime.utcnow().isoformat()

    except Exception as e:
        error = str(e)
        log_retrain_failed(retrain_id, error)
        retraining_status["last_status"] = f"failed: {error}"

    finally:
        retraining_status["is_training"] = False


@router.post("/retrain")
async def trigger_retrain(request: Request):
    """
    Trigger model retraining on the uploaded images.
    Logs the retraining event to the database.
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
    retrain_id = log_retrain_start()

    thread = threading.Thread(
        target=run_retraining,
        args=(model_path, UPLOAD_DIR, retrain_id),
        daemon=True
    )
    thread.start()

    return {
        "message": "Retraining started in background",
        "retrain_id": retrain_id,
        "classes_found": class_dirs,
        "check_status": "/retrain/status"
    }


@router.get("/retrain/status")
async def retrain_status():
    """
    Check the current status of retraining.
    """
    return retraining_status


@router.get("/retrain/history")
async def retrain_history():
    """
    Return the last 10 retraining events from the database.
    """
    return {"history": get_retrain_history()}
