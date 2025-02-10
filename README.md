# Whisper Service

A GPU-accelerated speech-to-text service using OpenAI's Whisper model (large-v3), implemented with Ray Serve and FastAPI.

## Features

- Fast speech-to-text transcription using Whisper large-v3 model
- GPU acceleration with CUDA support
- Streaming capability
- Containerized deployment with Docker

## Prerequisites

- Docker with NVIDIA Container Runtime installed
- NVIDIA GPU with appropriate drivers
- Docker Compose
- At least 10GB of GPU memory recommended

## Quick Start

1. Pull the Docker image:
```bash
docker pull dimadgo/ray_whisper:1
```

2. Start the service:
```bash
docker-compose up
```

The service will be available at `http://localhost:8033`

## API Usage

### Transcribe Audio

```bash
curl -X POST -H "Content-Type: application/octet-stream" --data-binary @your_audio_file.webm http://localhost:8033/
```

The service accepts audio files in various formats (webm, wav, mp3, etc.) and returns a JSON response with the transcription:

```json
{
    "transcription": "Your transcribed text will appear here"
}
```

## Testing

A test client and sample audio file are provided in the repository:

```bash
python test_client.py
```

## Notes

- The service uses the Whisper large-v3 model for optimal transcription quality
- GPU acceleration is required for reasonable performance
- The service uses shared memory of 10.24GB to handle large audio files
- Multiple GPU devices can be utilized by adjusting the `count` parameter in docker-compose.yml 