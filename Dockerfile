# Use official Python slim image with build tools
FROM python:3.10-slim

# Install system dependencies for dlib and opencv
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget unzip \
    libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev \
    python3-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of your code
COPY . .

# Create uploads folder
RUN mkdir uploads

# Expose port (Railway uses $PORT)
ENV PORT=10000
EXPOSE $PORT

# Start Flask with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "server:app"]
