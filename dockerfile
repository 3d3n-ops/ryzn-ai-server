FROM python:3.10-slim  

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata \
    PYTESSERACT_CMD=/usr/bin/tesseract \
    POPPLER_PATH=/usr/bin \
    FFMPEG_PATH=/usr/bin \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

CMD ["uvicorn", "src.app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]