import tensorflow as tf
import numpy as np
from src.preprocessing import preprocess_uploaded_image


CLASSES = ["Cerscospora", "Healthy", "Leaf rust", "Miner", "Phoma"]

DISEASE_INFO = {
    "Cerscospora": {
        "description": "Cercospora Leaf Spot — causes circular brown spots with yellow halo on leaves.",
        "severity": "Moderate",
        "action": "Apply copper-based fungicide. Remove and destroy affected leaves. Improve air circulation."
    },
    "Healthy": {
        "description": "No disease detected. The leaf appears healthy.",
        "severity": "None",
        "action": "Continue regular farm maintenance and monitoring."
    },
    "Leaf rust": {
        "description": "Coffee Leaf Rust — the most damaging coffee disease. Causes orange-brown powdery spots.",
        "severity": "High",
        "action": "Apply systemic fungicide immediately. Isolate affected plants. Monitor neighboring plants closely."
    },
    "Miner": {
        "description": "Leaf Miner — insect damage causing pale tunnels and trails through the leaf tissue.",
        "severity": "Moderate",
        "action": "Apply appropriate insecticide. Remove heavily damaged leaves. Introduce natural predators where possible."
    },
    "Phoma": {
        "description": "Phoma Twig Blight — causes dark lesions on stems and leaves, leads to dieback.",
        "severity": "Moderate",
        "action": "Prune affected branches. Apply fungicide to cut surfaces. Avoid overhead irrigation."
    }
}


def predict_from_bytes(image_bytes, model):
    """
    Run inference on raw image bytes uploaded via the API.

    Args:
        image_bytes: raw bytes of the uploaded image
        model: loaded Keras model

    Returns:
        dict with predicted class, confidence, all probabilities, and disease info
    """
    image = preprocess_uploaded_image(image_bytes)
    predictions = model.predict(image, verbose=0)
    predicted_idx = int(tf.argmax(predictions[0]).numpy())
    confidence = float(predictions[0][predicted_idx]) * 100

    predicted_class = CLASSES[predicted_idx]

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "all_probabilities": {
            CLASSES[i]: round(float(predictions[0][i]) * 100, 2)
            for i in range(len(CLASSES))
        },
        "disease_info": DISEASE_INFO[predicted_class]
    }


def predict_from_path(image_path, model):
    """
    Run inference on an image file path.
    Used for testing and batch prediction.

    Args:
        image_path: absolute or relative path to image file
        model: loaded Keras model

    Returns:
        dict with predicted class, confidence, all probabilities, and disease info
    """
    image = tf.io.read_file(image_path)
    image_bytes = tf.io.read_file(image_path)
    raw = open(image_path, "rb").read()
    return predict_from_bytes(raw, model)


def predict_batch(image_paths, model):
    """
    Run inference on a list of image paths and return results for each.

    Args:
        image_paths: list of image file paths
        model: loaded Keras model

    Returns:
        list of prediction dicts
    """
    results = []
    for path in image_paths:
        try:
            result = predict_from_path(path, model)
            result["image_path"] = path
            results.append(result)
        except Exception as e:
            results.append({
                "image_path": path,
                "error": str(e)
            })
    return results
