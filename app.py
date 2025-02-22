import ray
from ray import serve
from starlette.requests import Request
from fastapi import FastAPI, File, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from faster_whisper import WhisperModel
import os
import io
import logging
import time
from logging.handlers import RotatingFileHandler

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0, "num_gpus": 1})
class Transcriber:
    def __init__(self):
        # Set up logging
        self.logger = logging.getLogger("transcriber")
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Create logs directory if it doesn't exist
        log_directory = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_directory, exist_ok=True)
        log_file = os.path.join(log_directory, 'transcriber.log')
        
        # Create a file handler with proper permissions
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            mode='a'
        )
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Also add a stream handler for console output
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Initializing Transcriber service")
        
        self.model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )
        self.logger.info(f"Initialized Whisper model: {MODEL_SIZE} on {DEVICE} with {COMPUTE_TYPE}")

    def transcribe(self, audio_bytes: bytes, request_id: str, initial_prompt: str = None, prefix: str = None, language: str = None, task: str = "transcribe") -> dict:
        self.logger.info(f"[{request_id}] Starting transcription")
        transcribe_start = time.time()
        segments, info = self.model.transcribe(
            io.BytesIO(audio_bytes),
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
            word_timestamps=True,
            vad_parameters={"threshold": VAD_THRESHOLD},
            initial_prompt=initial_prompt,
            prefix=prefix,
            language=language,
            task=task
        )
        transcribe_end = time.time()
        transcribe_duration = transcribe_end - transcribe_start
        
        return {
            "segments": segments,
            "timing": {
                "transcribe_time_ms": transcribe_duration * 1000,
                "realtime_factor": transcribe_duration/info.duration if info.duration else 0
            }
        }

    async def __call__(self, request: Request) -> dict:
        # Verify authentication
        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header.split()[1] != API_TOKEN:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        request_id = str(time.time())
        start_time = time.time()
        
        try:
            # Read raw bytes from request body
            self.logger.info(f"[{request_id}] Starting request processing")
            audio_bytes = await request.body()
            initial_prompt = request.query_params.get("initial_prompt")
            prefix = request.query_params.get("prefix")
            language = request.query_params.get("language")
            task = request.query_params.get("task", "transcribe")
            body_read_time = time.time()
            audio_size = len(audio_bytes) / 1024  # Size in KB
            
            self.logger.info(
                f"[{request_id}] Read request body: {audio_size:.2f}KB in "
                f"{(body_read_time - start_time)*1000:.2f}ms"
            )

            # Get transcription
            result = self.transcribe(audio_bytes, request_id, initial_prompt, prefix, language, task)
            
            # Add timing information
            result["timing"]["request_id"] = request_id
            result["timing"]["body_read_time_ms"] = (body_read_time - start_time) * 1000
            result["timing"]["total_time_ms"] = (time.time() - start_time) * 1000
            
            self.logger.info(
                f"[{request_id}] Transcription completed:\n"
                f"  - Model inference time: {result['timing']['transcribe_time_ms']:.2f}ms\n"
                f"  - Total request time: {result['timing']['total_time_ms']:.2f}ms\n"
                f"  - Audio/Processing ratio: {result['timing']['realtime_factor']:.2f}x realtime"
            )
            
            return result
            
        except Exception as e:
            error_time = time.time()
            self.logger.error(
                f"[{request_id}] Error during transcription after "
                f"{(error_time - start_time)*1000:.2f}ms: {str(e)}"
            )
            raise HTTPException(status_code=500, detail=str(e))

transcriber_app = Transcriber.bind()