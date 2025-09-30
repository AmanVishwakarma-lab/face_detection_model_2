# ===== Base Image =====
FROM python:3.10-slim

# ===== System dependencies for dlib, OpenCV, DeepFace =====
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget unzip \
    libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev \
    python3-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

# ===== Set working directory =====
WORKDIR /app

# ===== Copy requirements =====
COPY requirements.txt .

# ===== Install Python dependencies =====
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ===== Copy project code =====
COPY . .

# ===== Optional uploads folder (if you keep disk saves) =====
RUN mkdir -p uploads

# ===== Expose port for Railway =====
ENV PORT=10000
EXPOSE $PORT

# ===== Start Flask with Gunicorn, use Railway PORT =====
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT server:app"]
