# Deep Learning Finance — API container
#
# Build: docker build -t dl-finance .
# Run:   docker compose up

FROM python:3.10-slim

# --- Python runtime flags ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# --- System dependencies (required to compile scipy/numpy C extensions) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies (separate layer — only rebuilds when requirements change) ---
COPY requirements-api.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

# --- Application source and configuration ---
COPY src/ ./src/
COPY config/ ./config/
COPY dashboard.html ./dashboard.html

# --- Entrypoint script (creates artifact subdirs inside mounted volume) ---
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# --- API port ---
EXPOSE 8000

# --- Startup: init artifact dirs then launch uvicorn ---
ENTRYPOINT ["./entrypoint.sh"]
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
