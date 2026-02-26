FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements-webapp.txt .
RUN pip install --no-cache-dir -r requirements-webapp.txt

# App code
COPY webapp/ webapp/
COPY run_webapp.py .

# Data directory (mount or copy predictions at deploy time)
RUN mkdir -p /data/models
ENV MLB_DATA_DIR=/data
ENV MLB_HOST=0.0.0.0
ENV MLB_PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "8000"]
