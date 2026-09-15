FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVER_NAME=0.0.0.0 \
    SERVER_PORT=7860 \
    PORT=7860 \
    VECTOR_STORE_PATH=/app/data/vector_store \
    DOCUMENT_STORE_PATH=/app/data/documents

WORKDIR /app

# Install system dependencies for OCR, PDF processing, and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    poppler-utils \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to keep image compact
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache embedding model to ensure instant, offline-capable startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application source code
COPY . .

# Ensure data directories exist
RUN mkdir -p /app/data/vector_store /app/data/documents

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "app.py"]
