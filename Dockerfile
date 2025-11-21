FROM python:3.12-slim

WORKDIR /app

# System deps (optional, kept minimal here)
RUN apt-get update && apt-get install -y \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY app.py /app/app.py
COPY config.yaml /app/config.yaml

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn APScheduler pandas networkx py2neo pyyaml

ENV PCC_CONFIG_PATH=/app/config.yaml

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
