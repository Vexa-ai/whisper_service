import logging
from typing import Optional, Dict, Any
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self, model_size: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info(f"Initialized Whisper model: {model_size} on {device}")
        
    async def transcribe(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Transcribe audio data and return the results
        """
        try:
            segments, info = self.model.transcribe(audio_data)
            return {
                "segments": [
                    {
                        "text": segment.text,
                        "start": segment.start,
                        "end": segment.end,
                        "confidence": segment.confidence
                    }
                    for segment in segments
                ],
                "language": info.language,
                "language_probability": info.language_probability
            }
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise

    def get_status(self) -> Dict[str, Any]:
        """
        Return the current status of the model
        """
        return {
            "status": "ready",
            "model_size": self.model.model_size,
            "device": self.model.device,
            "compute_type": self.model.compute_type
        } 