#!/bin/bash

# Bind FastAPI to localhost only - Render cannot detect this port
# This forces Render to always find Streamlit on 8501
uvicorn app.main:app --host 127.0.0.1 --port 8000 &

# Wait for FastAPI to be ready
sleep 10

# Start Streamlit on 0.0.0.0 - this is what Render detects and exposes
exec streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
