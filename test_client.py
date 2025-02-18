import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def process_audio(audio_file_path, task="transcribe", language=None, initial_prompt=None, prefix=None):
    """
    Process audio file with specified parameters for either transcription or translation
    
    Args:
        audio_file_path (str): Path to the audio file
        task (str): Either "transcribe" or "translate"
        language (str): Target language code (e.g., "en", "es", "fr")
        initial_prompt (str): Optional initial prompt for the model
        prefix (str): Optional prefix for the model
    """
    # Get token from environment variable
    token = os.getenv("WHISPER_API_TOKEN", "default_token_change_me")
    
    # Read the audio file as bytes
    with open(audio_file_path, 'rb') as f:
        audio_bytes = f.read()
    
    # Set up headers with authentication token
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    # Build query parameters
    params = {}
    if task:
        params["task"] = task
    if language:
        params["language"] = language
    if initial_prompt:
        params["initial_prompt"] = initial_prompt
    if prefix:
        params["prefix"] = prefix
    
    print(f"\nSending request with parameters:")
    print(f"Task: {task}")
    print(f"Language: {language}")
    print(f"Initial Prompt: {initial_prompt}")
    print(f"Prefix: {prefix}\n")
    
    # Send the request
    response = requests.post(
        "http://127.0.0.1:8000/",
        params=params,
        data=audio_bytes,
        headers=headers
    )
    
    if response.status_code == 401:
        print("Authentication failed - check your WHISPER_API_TOKEN")
        return
    
    # Print the response
    if response.status_code == 200:
        print("Result:", response.json())
    else:
        print(f"Error: {response.status_code}", response.text)

def run_tests():
    audio_file_path = "test_audio.webm"
    
    # Test 1: Basic transcription
    print("\n=== Test 1: Basic Transcription ===")
    process_audio(
        audio_file_path,
        task="transcribe"
    )
    
    # Test 2: Translation to English
    print("\n=== Test 2: Translation to English ===")
    process_audio(
        audio_file_path,
        task="translate",
        language="en"
    )
    
    # Test 3: Transcription with initial prompt
    print("\n=== Test 3: Transcription with Initial Prompt ===")
    process_audio(
        audio_file_path,
        task="transcribe",
        initial_prompt="This is a test recording"
    )
    
    # Test 4: Translation to Spanish
    print("\n=== Test 4: Translation to Spanish ===")
    process_audio(
        audio_file_path,
        task="translate",
        language="es"
    )

if __name__ == "__main__":
    run_tests()
    
    
#ray start --head --node-ip-address=0.0.0.0 --port=6379
#serve deploy app:transcriber_app