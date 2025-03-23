FROM python:3.9-slim

WORKDIR /app

# Install system dependencies and cleanup in one layer to reduce image size
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/uploads /app/static /app/cache /app/credentials
# Copy application code
COPY . .

# Set binary paths for Linux/Docker environment
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
ENV PYTESSERACT_CMD=/usr/bin/tesseract
ENV POPPLER_PATH=/usr/bin
ENV FFMPEG_PATH=/usr/bin

# Expose port
EXPOSE 8000
# Set the start command
CMD ["uvicorn", "src.app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]