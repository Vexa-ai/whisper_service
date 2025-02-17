FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg

# Install Python packages
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    fastapi==0.103.1 \
    "ray[serve]"==2.7.1 \
    faster-whisper==0.9.0 \
    python-dotenv==1.0.0 \
    websockets==12.0 \
    redis==4.5.5

# Copy application code
COPY . /app
WORKDIR /app 