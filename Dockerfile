FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch keeps the image ~2 GB instead of ~6 GB; swap index URL for GPU builds
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/gpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
