import argparse
import asyncio
import os
import time
from dotenv import load_dotenv
import websockets
import json
import requests

# Load environment variables from .env file
load_dotenv()

def test_transcription(audio_file_path):
    # Get token from environment variable
    token = os.getenv("WHISPER_API_TOKEN", "default_token_change_me")
    
    # Read the audio file as bytes
    with open(audio_file_path, 'rb') as f:
        audio_bytes = f.read()
    
    # Set up headers with authentication token
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    # Send the bytes directly with authentication
    response = requests.post("http://127.0.0.1:8000/", data=audio_bytes, headers=headers)
    
    if response.status_code == 401:
        print("Authentication failed - check your WHISPER_API_TOKEN")
        return
    
    # Print the response
    if response.status_code == 200:
        print("Transcription result:", response.text)
    else:
        print(f"Error: {response.status_code}", response.text)

async def test_stream_transcription():
    """Test real-time streaming with simulated timing"""
    token = os.getenv("WHISPER_API_TOKEN", "default_token_change_me")
    
    try:
        # Use proper URI parameters for authentication
        async with websockets.connect(
            f"ws://localhost:8000/ws/transcribe"
        ) as ws:
            print("Connected to WebSocket endpoint")
            
            # Simulate real-time audio input
            with open("test_audio.webm", "rb") as f:
                start_time = time.time()
                chunk_count = 0
                
                while True:
                    chunk = f.read(16000)  # 1 second of 16kHz audio
                    if not chunk:
                        break
                    
                    # Send chunk with timestamp
                    await ws.send(chunk)
                    chunk_count += 1
                    
                    # Receive partial transcription
                    response = await asyncio.wait_for(ws.recv(), timeout=5)
                    result = json.loads(response)
                    
                    # Calculate real-time metrics
                    elapsed = time.time() - start_time
                    real_time_factor = chunk_count / elapsed
                    
                    print(f"\nReceived partial @ {elapsed:.2f}s (RTF: {real_time_factor:.2f}x):")
                    for seg in result:
                        print(f"{seg['start']:.2f}-{seg['end']:.2f}s: {seg['text']}")
                    
                    # Simulate real-time pacing
                    await asyncio.sleep(1.0)  # Wait for next "real-time" chunk

    except websockets.exceptions.InvalidStatusCode as e:
        print(f"Connection failed: {e.status_code=}")
        if e.status_code == 403:
            print("Invalid authentication token")
    except asyncio.TimeoutError:
        print("Server response timeout")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Whisper Service')
    parser.add_argument('--mode', choices=['batch', 'stream'], default='stream',
                        help='Test mode: batch (POST) or stream (WebSocket)')
    args = parser.parse_args()

    if args.mode == 'stream':
        print("Starting real-time streaming test...")
        asyncio.run(test_stream_transcription())
    else:
        print("Starting batch transcription test...")
        audio_file_path = "test_audio.webm"
        test_transcription(audio_file_path)

#ray start --head --node-ip-address=0.0.0.0 --port=6379
#serve deploy app:transcriber_app