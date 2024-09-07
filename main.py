import time
from pathlib import Path

import os
import sys
import sounddevice as sd
import numpy as np
import tempfile
import wave
import threading
from flask import Flask, jsonify
import simpleaudio as sa
import whisper

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    flask_port: int = 5006
    base_folder_for_recordings: str
    use_local_model: bool = False

    class Config:
        env_file = ".env"  # Optional: Load environment variables from a .env file
        env_file_encoding = "utf-8"

# Load settings

settings = Settings()
# Make sure the folder for recordings exists, if not create it, and print that it was created
if not os.path.exists(settings.base_folder_for_recordings):
    os.makedirs(settings.base_folder_for_recordings)
    print(f"Folder for recordings created at: {settings.base_folder_for_recordings}")

# Access the settings
api_key = settings.openai_api_key
flask_port = settings.flask_port

# Use the settings in your application
print(f"API Key: {api_key}")
print(f"Flask Port: {flask_port}")

# Load the appropriate model or client
if settings.use_local_model:
    import whisper
    model = whisper.load_model("medium")
else:
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)

app = Flask(__name__)

# Global variables to control the recording state
is_recording = False
audio_file_path = None
recording_thread = None
audio_data = []

def play_wav(file_path):
    # Load the WAV file
    wave_obj = sa.WaveObject.from_wave_file(file_path)
    # Play the sound
    play_obj = wave_obj.play()
    # Wait for playback to finish before exiting
    play_obj.wait_done()

## Purely useful for debugging microphones.
# def list_available_devices():
#     devices = sd.query_devices()
#     for i, device in enumerate(devices):
#         print(f"Device {i}: {device['name']} (Input Channels: {device['max_input_channels']})")

# list_available_devices()

def get_preferred_device(preferred_names):
    devices = sd.query_devices()
    # Check for preferred devices
    for name in preferred_names:
        for i, device in enumerate(devices):
            if name.lower() in device['name'].lower() and device['max_input_channels'] > 0:
                print(f"Using device: {device['name']} (Index: {i})")
                return i  # Return the index of the preferred device
    # Fallback to default device
    print("No preferred device found, using default.")
    return sd.default.device[0]  # Return the default device index

# List of preferred microphones in order of preference
preferred_microphones = [
    "C03U multi-pattern microphone",
    # Add more preferred microphone names here
]

# Get the preferred device index
device_index = get_preferred_device(preferred_microphones)

def record_audio_continuously(max_duration=999999999999999999, device_index=None):
    global is_recording, audio_data

    samplerate = 16000
    is_recording = True

    print("Recording started...")
    with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16', device=device_index) as stream:
        start_time = time.time()
        while is_recording:
            data, _ = stream.read(1024)
            audio_data.append(data)
            if time.time() - start_time >= max_duration:
                break

    print("Recording stopped.")


def save_to_wav(audio_data, file_path=None, samplerate=16000):
    if file_path is None:
        file_path = os.path.join(tempfile.gettempdir(), "recording.wav")

    # save audio to path and return path
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes = 16 bits
        wf.setframerate(samplerate)
        wf.writeframes(np.concatenate(audio_data).tobytes())
    return file_path



# Read the prompt from a file
def read_prompt_file(file_path='prompt.txt'):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using default prompt.")
        return ""

# Load the prompt
prompt = read_prompt_file()

def transcribe_audio(wav_file):
    try:
        if settings.use_local_model:
            # Local model transcription
            result = model.transcribe(wav_file)
            transcription = result["text"]
        else:
            # Remote OpenAI API transcription
            with open(wav_file, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file, response_format="text"
                )
                transcription = transcript
        print(transcription)
        return transcription
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""  # Return an empty string instead of None

# Test API connection with recording
def test_api_connection_with_recording():
    global is_recording, audio_data

    try:
        # Perform a 1-second test recording
        record_audio_continuously(max_duration=2)
        wav_file = save_to_wav(audio_data)
        is_recording = False

        # Test transcription
        transcription = transcribe_audio(wav_file)
        os.remove(wav_file)  # Clean up the temporary file

        if transcription:
            print("API connection and transcription test successful!")
            print(f"Transcription: {transcription}")
        else:
            print("API connection test failed or no transcription returned.")
            sys.exit(1)
    except Exception as e:
        print(f"Error during API connection test: {e}")
        sys.exit(1)


@app.route("/start", methods=["POST"])
def start_recording():
    global is_recording, recording_thread, audio_data

    if is_recording:
        return jsonify({"message": "Recording is already in progress!"}), 400
    audio_data = []
    # Start recording in a background thread
    is_recording = True
    recording_thread = threading.Thread(target=record_audio_continuously)
    recording_thread.start()

    return jsonify({"message": "Recording started!"})


@app.route("/stop", methods=["POST"])
def stop_recording():
    global is_recording, recording_thread, audio_file_path

    if not is_recording:
        return jsonify({"message": "No recording is currently in progress!"}), 400

    # Stop the recording
    is_recording = False
    recording_thread.join()  # Wait for the recording thread to finish

    # Save the recorded audio to a WAV file
    wav_file_path = Path(settings.base_folder_for_recordings) / f"{time.strftime('%Y-%m-%d-%H:%M:.%S')}-{os.urandom(4).hex()}.wav"
    txt_file_path = wav_file_path.with_suffix(".txt")
    audio_file_path = save_to_wav(audio_data, file_path=str(wav_file_path))

    print(f"Audio file saved to: {audio_file_path}")

    # Transcribe the saved audio file
    transcription = transcribe_audio(audio_file_path)

    # Save the transcription to a text file
    with open(txt_file_path, "w") as f:
        f.write(transcription)

    if transcription:
        return jsonify({"transcription": transcription})
    else:
        return jsonify({"message": "Transcription was empty or failed."}), 500


if __name__ == "__main__":
    # # Test the API connection before starting the server
    # test_api_connection_with_recording()

    # Start the Flask server
    app.run(host="0.0.0.0", port=int(flask_port))
