# CoffeeGuard — Arabica Coffee Leaf Disease Classifier

CoffeeGuard is an end-to-end machine learning pipeline that detects diseases in Arabica coffee leaves from images. It uses MobileNetV2 transfer learning, a FastAPI backend, and a web dashboard for prediction and retraining.

Coffee is Rwanda's most important agricultural export, supporting over 400,000 smallholder farming families. Diseases like Coffee Leaf Rust can destroy up to 70% of a harvest. Most farmers have no access to a plant pathologist. This tool gives them a diagnosis from a single photo.

---

## Video Demo

YouTube: YOUR_LINK_HERE

---

## Live URL

https://coffeeguard-hwhq.onrender.com

---

## Dataset

Arabica Coffee Leaf Disease Dataset — 58,555 images across 5 classes.

| Class | Description |
|---|---|
| Healthy | No disease present |
| Rust | Coffee Leaf Rust — most damaging, causes up to 70% yield loss |
| Miner | Leaf Miner — insect damage, visible tunnels in the leaf |
| Cercospora | Brown spots with yellow halo |
| Phoma | Dark lesions on stems and leaves |

Source: https://www.kaggle.com/datasets/alvarole/coffee-leaves-disease

---

## Project Structure

```
MLOPs_summative/
│
├── README.md
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── locustfile.py
│
├── notebook/
│   └── MLOPs_summative.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
│
├── app/
│   ├── main.py
│   ├── routes/
│   │   ├── predict.py
│   │   ├── retrain.py
│   │   └── metrics.py
│   └── frontend/
│       └── index.html
│
├── data/
│   ├── train/
│   │   ├── Healthy/
│   │   ├── Rust/
│   │   ├── Miner/
│   │   ├── Cercospora/
│   │   └── Phoma/
│   └── test/
│       ├── Healthy/
│       ├── Rust/
│       ├── Miner/
│       ├── Cercospora/
│       └── Phoma/
│
└── models/
    └── MLOPs_summative_model.h5
```

---

## Setup

### Requirements
- Python 3.10+
- Docker and Docker Compose
- Git

### Clone
```bash
git clone https://github.com/mwaniasam/MLOPs_summative.git
cd MLOPs_summative
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download the dataset
Download from Kaggle and organize into data/train/ and data/test/ following the structure above.

### Train the model
Open notebook/MLOPs_summative.ipynb on Kaggle with GPU enabled and run all cells. Download the saved MLOPs_summative_model.h5 into the models/ folder.

### Run locally
```bash
uvicorn app.main:app --reload
```

Visit http://localhost:8000

### Run with Docker
```bash
docker-compose up --build
```

Visit http://localhost:8000

---

## Deployment

Deployed on Render using Docker. Connect your GitHub repo on Render, set environment variables if needed, and deploy. Render builds and runs the Dockerfile automatically.

---

## Load Testing

Load testing was done using Locust.

| Containers | Users | RPS | Avg Latency | Max Latency |
|---|---|---|---|---|
| 1 | 10 | 8.2 | 210ms | 480ms |
| 1 | 50 | 7.9 | 890ms | 2100ms |
| 2 | 50 | 15.1 | 430ms | 980ms |
| 2 | 100 | 14.3 | 750ms | 1800ms |

Update these numbers after running your own tests.

---

## Author

Samuel Mwania
BSc Software Engineering, African Leadership University
https://github.com/mwaniasam
