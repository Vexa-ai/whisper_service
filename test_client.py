import requests

def test_transcription(audio_file_path):
    # Read the audio file as bytes
    with open(audio_file_path, 'rb') as f:
        audio_bytes = f.read()
    
    # Send the bytes directly
    response = requests.post("http://127.0.0.1:8000/", data=audio_bytes)
    
    # Print the response
    print("Transcription result:", response.text)

if __name__ == "__main__":
    # Use the webm test file
    audio_file_path = "test_audio.webm"
    test_transcription(audio_file_path)