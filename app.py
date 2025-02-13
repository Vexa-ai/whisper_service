import ray
from ray import serve
from fastapi import FastAPI, File, Request, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from faster_whisper import WhisperModel
import os
import io

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

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": NUM_CPUS, "num_gpus": NUM_GPUS})
@serve.ingress(app)
class Transcriber:
    def __init__(self):
        self.model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )

    @app.post("/")
    async def transcribe(self, request: Request, authenticated: bool = Depends(verify_token)) -> dict:
        # Read raw bytes from request body
        audio_bytes = await request.body()
        segments, info = self.model.transcribe(
            io.BytesIO(audio_bytes),
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
            word_timestamps=True,
            vad_parameters={"threshold": VAD_THRESHOLD},
        )
        return {"segments": segments}

transcriber_app = Transcriber.bind()


#ray start --head
#serve run app:transcriber_app