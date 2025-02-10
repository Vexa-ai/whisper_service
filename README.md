# Whisper Service

A GPU-accelerated speech-to-text service leveraging [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and Ray Serve. This service utilizes OpenAI's Whisper model (large-v3) optimized with faster-whisper for significantly improved transcription speeds and reduced resource consumption, while Ray Serve and FastAPI enable scalable and low-latency real-time processing.

## Features

- Fast speech-to-text transcription using Whisper large-v3 model
- Optimized speech-to-text transcription using Whisper large-v3 model with faster-whisper enhancements
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

## Benefits of Integrating faster-whisper with Ray Serve

By combining [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with Ray Serve, this service achieves:

- **High Performance:** faster-whisper employs CTranslate2 to re-implement OpenAI's Whisper, offering up to 4x faster transcription speeds while reducing memory usage.
- **Scalability:** Ray Serve provides robust scalability and flexible deployment options, supporting load balancing, parallel processing, and efficient resource utilization.
- **Real-time Transcription:** The synergy enables low-latency processing ideal for live transcription applications.
- **Ease of Deployment:** Leveraging Docker with Ray Serve simplifies containerization, orchestration, and maintenance of the service.
- **Flexibility:** Both faster-whisper and Ray Serve support GPU acceleration and inference optimizations like quantization, making them suitable for diverse environments.

For more details, please refer to:
- [faster-whisper GitHub repository](https://github.com/SYSTRAN/faster-whisper)
- [Ray Serve Advanced Guides](https://docs.ray.io/en/latest/serve/advanced-guides/multi-app-container.html) 