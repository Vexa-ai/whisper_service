import ray
from ray import serve
from starlette.requests import Request
from fastapi import FastAPI, File, HTTPException, Security, Depends, WebSocket
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from faster_whisper import WhisperModel
import os
import io
import logging
import time
from logging.handlers import RotatingFileHandler
import json
from starlette.websockets import WebSocketDisconnect
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute, Route
from starlette.responses import JSONResponse

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Create FastAPI app for better WebSocket support
app = FastAPI()
security = HTTPBearer()

# Get configurations from environment variables
API_TOKEN = os.getenv("WHISPER_API_TOKEN", "default_token_change_me")
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
NUM_CPUS = float(os.getenv("WHISPER_NUM_CPUS", "1"))
NUM_GPUS = float(os.getenv("WHISPER_NUM_GPUS", "0.1"))
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "True").lower() == "true"
VAD_THRESHOLD = float(os.getenv("WHISPER_VAD_THRESHOLD", "0.9"))
NUM_REPLICAS = os.getenv("NUM_REPLICAS", "1")

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0, "num_gpus": 1})
@serve.ingress(app)
class Transcriber:
    def __init__(self):
        # Set up logging
        self.logger = logging.getLogger("transcriber")
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_directory = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_directory, exist_ok=True)
        log_file = os.path.join(log_directory, 'transcriber.log')
        
        # Create handlers
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,
            backupCount=5,
            mode='a'
        )
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Initializing Transcriber service")
        
        # Initialize model
        self.model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )
        self.logger.info(f"Initialized Whisper model: {MODEL_SIZE} on {DEVICE}")

    @app.post("/")
    async def handle_http(self, request: Request) -> JSONResponse:
        # Verify authentication
        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header.split()[1] != API_TOKEN:
            return JSONResponse(
                {"error": "Invalid authentication token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        request_id = str(time.time())
        start_time = time.time()
        
        try:
            # Read raw bytes from request body
            self.logger.info(f"[{request_id}] Starting request processing")
            audio_bytes = await request.body()
            body_read_time = time.time()
            audio_size = len(audio_bytes) / 1024  # Size in KB
            
            self.logger.info(
                f"[{request_id}] Read request body: {audio_size:.2f}KB in "
                f"{(body_read_time - start_time)*1000:.2f}ms"
            )

            # Get transcription
            segments, info = self.model.transcribe(
                io.BytesIO(audio_bytes),
                beam_size=BEAM_SIZE,
                vad_filter=VAD_FILTER,
                word_timestamps=True,
                vad_parameters={"threshold": VAD_THRESHOLD},
            )
            transcribe_end = time.time()
            transcribe_duration = transcribe_end - body_read_time
            
            result = {
                "segments": [{
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "words": [
                        {"word": word.word, "start": word.start, "end": word.end}
                        for word in segment.words
                    ] if segment.words else []
                } for segment in segments],
                "timing": {
                    "request_id": request_id,
                    "transcribe_time_ms": transcribe_duration * 1000,
                    "body_read_time_ms": (body_read_time - start_time) * 1000,
                    "total_time_ms": (time.time() - start_time) * 1000,
                    "realtime_factor": transcribe_duration/info.duration if info.duration else 0
                }
            }
            
            self.logger.info(
                f"[{request_id}] Transcription completed:\n"
                f"  - Model inference time: {result['timing']['transcribe_time_ms']:.2f}ms\n"
                f"  - Total request time: {result['timing']['total_time_ms']:.2f}ms\n"
                f"  - Audio/Processing ratio: {result['timing']['realtime_factor']:.2f}x realtime"
            )
            
            return JSONResponse(result)
            
        except Exception as e:
            error_time = time.time()
            self.logger.error(
                f"[{request_id}] Error during transcription after "
                f"{(error_time - start_time)*1000:.2f}ms: {str(e)}"
            )
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.websocket("/ws/transcribe")
    async def handle_websocket(self, websocket: WebSocket):
        self.logger.info("WebSocket connection request received")
        try:
            await websocket.accept()
            self.logger.info("WebSocket connection accepted")
            
            buffer = bytearray()
            try:
                while True:
                    try:
                        data = await websocket.receive_bytes()
                        self.logger.info(f"Received {len(data)} bytes")
                        buffer.extend(data)
                        
                        # Process when buffer reaches chunk size
                        if len(buffer) >= 16000:  # 1s of 16kHz audio
                            self.logger.info("Processing audio chunk")
                            segments, _ = self.model.transcribe(
                                audio=io.BytesIO(buffer),
                                beam_size=BEAM_SIZE,
                                word_timestamps=True,
                                vad_filter=VAD_FILTER
                            )
                            
                            result = [{
                                "text": segment.text,
                                "start": segment.start,
                                "end": segment.end,
                                "words": [
                                    {"word": word.word, "start": word.start, "end": word.end}
                                    for word in segment.words
                                ] if segment.words else []
                            } for segment in segments]
                            
                            await websocket.send_text(json.dumps(result))
                            buffer = bytearray()
                            
                    except WebSocketDisconnect:
                        self.logger.info("Client disconnected")
                        break
                    except Exception as e:
                        self.logger.error(f"Error processing message: {str(e)}", exc_info=True)
                        await websocket.close(code=1011)
                        break
                        
            except Exception as e:
                self.logger.error(f"Error in websocket loop: {str(e)}", exc_info=True)
                await websocket.close(code=1011)
                    
        except Exception as e:
            self.logger.error(f"Error accepting websocket: {str(e)}", exc_info=True)
            # Don't try to close the websocket here as it hasn't been accepted yet

transcriber_app = Transcriber.bind()

if __name__ == "__main__":
    ray.init(address="auto", namespace="serve")
    serve.run(transcriber_app)