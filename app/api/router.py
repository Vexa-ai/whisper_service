from fastapi import APIRouter, UploadFile, HTTPException, Depends
from typing import Dict, Any
from app.core.service import WhisperService
import io

router = APIRouter()

async def get_service():
    service = WhisperService.get_current()
    if not service.is_ready:
        await service.startup()
    return service

@router.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile,
    service: WhisperService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Endpoint to transcribe audio files
    """
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    try:
        contents = await audio_file.read()
        model = service.get_model()
        if not model:
            raise HTTPException(status_code=503, detail="Model not initialized")
        result = await model.transcribe(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_status(
    service: WhisperService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Get the current status of the transcription service
    """
    try:
        model = service.get_model()
        return model.get_status() if model else {"status": "initializing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint
    """
    return {"status": "healthy"} 