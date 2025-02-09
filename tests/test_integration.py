import pytest
import httpx
import os
from pathlib import Path

TEST_AUDIO_PATH = Path(__file__).parent / "test_audio.webm"
BASE_URL = "http://localhost:8033/api/v1"

@pytest.mark.asyncio
async def test_health_check():
    """Test the health check endpoint"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_service_status():
    """Test the service status endpoint"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/status")
        assert response.status_code == 200
        status = response.json()
        assert "status" in status
        assert "model_size" in status
        assert "device" in status
        assert "compute_type" in status

@pytest.mark.asyncio
async def test_transcribe_audio():
    """Test audio transcription with a real audio file"""
    if not TEST_AUDIO_PATH.exists():
        pytest.skip("Test audio file not found")

    async with httpx.AsyncClient() as client:
        with open(TEST_AUDIO_PATH, "rb") as f:
            files = {"audio_file": ("test_audio.webm", f, "audio/webm")}
            response = await client.post(
                f"{BASE_URL}/transcribe",
                files=files,
                timeout=300  # Longer timeout for transcription
            )
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify response structure
        assert "segments" in result
        assert "language" in result
        assert "language_probability" in result
        
        # Verify segments structure
        assert len(result["segments"]) > 0
        for segment in result["segments"]:
            assert "text" in segment
            assert "start" in segment
            assert "end" in segment
            assert "confidence" in segment 