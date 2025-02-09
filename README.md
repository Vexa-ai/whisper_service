# Whisper Transcription Service

A distributed Whisper model serving system using Ray Serve and FastAPI. This service provides a scalable solution for audio transcription using OpenAI's Whisper model.

## Features

- Distributed model serving using Ray Serve
- GPU acceleration support
- Health checking and status monitoring
- RESTful API interface
- Docker containerization with CUDA support

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd whisper_service
```

2. Build and start the service:
```bash
docker-compose up --build
```

The service will be available at `http://localhost:8000`

## API Endpoints

### Transcribe Audio
```
POST /api/v1/transcribe
```
Upload an audio file for transcription.

### Check Status
```
GET /api/v1/status
```
Get the current status of the transcription service.

### Health Check
```
GET /api/v1/health
```
Check if the service is healthy.

## Example Usage

Using curl to transcribe an audio file:
```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@path/to/audio.mp3"
```

## Configuration

The service can be configured through environment variables:
- `RAY_ADDRESS`: Ray cluster address (default: "local")
- GPU settings can be adjusted in the docker-compose.yml file

## License

[MIT License](LICENSE) 