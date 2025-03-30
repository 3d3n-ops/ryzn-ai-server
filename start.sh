#!/bin/bash
set -e

# Create necessary directories
mkdir -p uploads static cache credentials

# Install system dependencies
apt-get update
apt-get install -y tesseract-ocr poppler-utils ffmpeg

# Start the FastAPI application with Gunicorn for production
gunicorn -c src/app/backend/gunicorn_config.py src.app.backend.main:app 