import ray
from ray import serve
from fastapi import FastAPI, File, Request
from faster_whisper import WhisperModel
import os
import io

app = FastAPI()

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0.1})
@serve.ingress(app)
class Transcriber:
    def __init__(self):
        model_size = "large-v3"  # Using smaller model for testing
        self.model = WhisperModel(
            model_size,
            device="cuda",
            compute_type="float16",
        )

    @app.post("/")
    async def transcribe(self, request: Request) -> dict:
        # Read raw bytes from request body
        audio_bytes = await request.body()
        segments, info = self.model.transcribe(
            io.BytesIO(audio_bytes),
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
            vad_parameters={"threshold": 0.9},
        )
        # Convert segments to text
        text = " ".join([segment.text for segment in segments])
        return {"transcription": text}

transcriber_app = Transcriber.bind()


#ray start --head
#serve run app:transcriber_app