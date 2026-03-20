#!/bin/bash

# Start FastAPI on internal port 8000 in background
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Wait for FastAPI to be ready
sleep 15

# Start Streamlit on port 8501 - this is what Render exposes
exec streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
