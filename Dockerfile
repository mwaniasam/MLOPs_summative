FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

COPY . .

RUN mkdir -p models data/uploads

RUN chmod +x start.sh

ENV PORT=8501

EXPOSE 8000 8501

CMD ["./start.sh"]
