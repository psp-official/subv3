FROM python:3.11-slim

# Install FFmpeg and system dependencies required for Faster-Whisper
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Faster-Whisper 'small' model during Docker build so Render starts fast!
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8', download_root='/tmp/whisper_models')"

COPY . .

ENV PORT=10000
ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=small
ENV WHISPER_THREADS=4

EXPOSE 10000

# IMPORTANT: Use 1 worker + 8 threads so in-memory background job state is shared across requests
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "8", "--timeout", "0"]
