FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN mkdir -p /data

# one process (one photo at a time keeps memory predictable), many threads for the web pages
CMD gunicorn app:app --workers 1 --threads 8 --timeout 600 --bind 0.0.0.0:${PORT:-8080}
