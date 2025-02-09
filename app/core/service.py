from ray import serve
from fastapi import FastAPI
from app.core.whisper_model import WhisperTranscriber

app = FastAPI(
    title="Whisper Transcription Service",
    description="Distributed Whisper model serving using Ray Serve",
    version="1.0.0"
)

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
    max_concurrent_queries=4
)
@serve.ingress(app)
class WhisperService:
    def __init__(self):
        self._model = None
        self.is_ready = False

    async def startup(self):
        """Initialize the model on startup"""
        if not self._model:
            self._model = WhisperTranscriber()
            self.is_ready = True
            
    def get_model(self):
        """Get the model instance"""
        return self._model

    @app.post("/transcribe")
    async def transcribe_audio(self, audio_data: bytes) -> dict:
        """Transcribe audio data"""
        if not self._model:
            await self.startup()
        return await self._model.transcribe(audio_data)

    @app.get("/status")
    async def get_status(self) -> dict:
        """Get service status"""
        if not self._model:
            return {"status": "initializing"}
        return self._model.get_status()

    @app.get("/health")
    async def health_check(self) -> dict:
        """Health check endpoint"""
        return {"status": "healthy"} 