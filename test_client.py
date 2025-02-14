import requests
import os
from dotenv import load_dotenv

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

if __name__ == "__main__":
    # Use the webm test file
    audio_file_path = "test_audio.webm"
    test_transcription(audio_file_path)
    
    
#ray start --head --node-ip-address=0.0.0.0 --port=6379
#serve deploy app:transcriber_app