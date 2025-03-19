#!/bin/bash
set -e

# Create necessary directories
mkdir -p uploads static cache credentials

# Install system dependencies
apt-get update
apt-get install -y tesseract-ocr poppler-utils ffmpeg

# Start the FastAPI application with Gunicorn for production
gunicorn --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT src.app.backend.main:app 