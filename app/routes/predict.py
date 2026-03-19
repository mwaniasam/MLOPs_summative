import time
from fastapi import APIRouter, UploadFile, File, HTTPException, Request

router = APIRouter()


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Accept a single coffee leaf image and return the predicted disease class.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        start = time.time()
        image_bytes = await file.read()
        model = request.app.state.model

        from src.prediction import predict_from_bytes
        result = predict_from_bytes(image_bytes, model)

        result["latency_ms"] = round((time.time() - start) * 1000, 2)
        result["filename"] = file.filename
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
