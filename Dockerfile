FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# system deps for common CV libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# copy project files
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PORT=8080

EXPOSE 8080

# Bind to the platform-provided port so the same image works on Render, HF Spaces, and local Docker runs.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} web_app:app --workers 1 --threads 4 --timeout 120 --graceful-timeout 30 --access-logfile - --log-level debug"]
