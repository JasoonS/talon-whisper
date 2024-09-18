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
from faster_whisper import WhisperModel
import torch  # Add this import
from pydub import AudioSegment

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    flask_port: int = 5006
    base_folder_for_recordings: str
    use_local_model: bool = False
    language: str = "en"
    use_faster_whisper: bool = False
    faster_whisper_model: str = "large-v3"
    whisper_model: str = "medium"

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

def print_gpu_memory_info():
    if torch.cuda.is_available():
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Cached GPU memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"Free GPU memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated() - torch.cuda.memory_reserved()) / 1e9:.2f} GB")
    else:
        print("CUDA is not available. Running on CPU.")

# Load the appropriate model or client
if settings.use_local_model:
    if settings.use_faster_whisper:
        print(f"Using faster-whisper model: {settings.faster_whisper_model}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        model = WhisperModel(settings.faster_whisper_model, device=device, compute_type=compute_type)
        print_gpu_memory_info()
    else:
        import torch
        print(f"Using Whisper model: {settings.whisper_model}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(torch.__version__)
        print(torch.cuda.is_available())
        import whisper
        
        print("GPU memory before loading model:")
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisper.load_model(settings.whisper_model).to(device)
        except torch.cuda.OutOfMemoryError:
            print("GPU memory insufficient, falling back to CPU")
            device = "cpu"
            model = whisper.load_model("small").to(device)
        print(f"Using device: {device}")
        
        print("GPU memory after loading model:")
        print_gpu_memory_info()

# else: # always use openai for longer transcription
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
    # Print all available audio devices
    for i, device in enumerate(devices):
        print(f"Device {i}: {device['name']} (Input Channels: {device['max_input_channels']})")
    # Check for preferred devices
    for name in preferred_names:
        for i, device in enumerate(devices):
            if name.lower() in device['name'].lower(): 
                # if device['max_input_channels'] > 0: ## My mic was sometimes showing as 0 input channels so removed this.
                  print(f"Using device: {device['name']} (Index: {i})")
                  return i  # Return the index of the preferred device
                # else:
                #     print(f"Device {device['name']} has no input channels.")
    # Fallback to default device
    print("No preferred device found, using default.")
    return sd.default.device[0]  # Return the default device index

# List of preferred microphones in order of preference
preferred_microphones = [
    # "C03U multi-pattern microphone",
    # "Samson C03U",
    "C03U",
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


def save_to_mp3(audio_data, file_path=None, samplerate=16000):
    if file_path is None:
        file_path = os.path.join(tempfile.gettempdir(), "recording.mp3")

    # Save audio to a temporary WAV file first
    temp_wav_path = file_path.replace(".mp3", ".wav")
    with wave.open(temp_wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes = 16 bits
        wf.setframerate(samplerate)
        wf.writeframes(np.concatenate(audio_data).tobytes())

    # Convert WAV to MP3
    audio = AudioSegment.from_wav(temp_wav_path)
    audio.export(file_path, format="mp3")

    # Remove the temporary WAV file
    os.remove(temp_wav_path)

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

def transcribe_audio(audio_file, use_openai=False):
    try:
        if settings.use_local_model and not use_openai:
            print(f"Using {settings.use_faster_whisper} for transcription")
            if settings.use_faster_whisper:
                # Faster-whisper transcription
                segments, info = model.transcribe(
                    audio_file, 
                    beam_size=5, 
                    language=settings.language,
                    initial_prompt=prompt
                )
                transcription = " ".join([segment.text for segment in segments])
            else:
                # Local model transcription
                audio = whisper.load_audio(audio_file)
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio).to(model.device)
                _, probs = model.detect_language(mel)
                options = whisper.DecodingOptions(
                    language=settings.language, 
                    fp16=torch.cuda.is_available(),
                    prompt=prompt
                )
                result = whisper.decode(model, mel, options)
                transcription = result.text
        else:
            # Remote OpenAI API transcription
            print("Using OpenAI API for transcription")
            with open(audio_file, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file, 
                    response_format="text", 
                    language=settings.language,
                    prompt=prompt
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

    # Save the recorded audio to an MP3 file
    mp3_file_path = Path(settings.base_folder_for_recordings) / f"{time.strftime('%Y-%m-%d-%H:%M:%S')}-{os.urandom(4).hex()}.mp3"
    txt_file_path = mp3_file_path.with_suffix(".txt")
    audio_file_path = save_to_mp3(audio_data, file_path=str(mp3_file_path))

    print(f"Audio file saved to: {audio_file_path}")

    # Calculate the duration of the recording
    duration = len(np.concatenate(audio_data)) / 16000  # Assuming 16000 Hz sample rate

    # Decide whether to use the local model or the OpenAI API
    use_openai = duration > 30 ## I get 'out of memory' errors for anything longer than ~30 seconds. So send them to openai for now.

    # Transcribe the saved audio file
    transcription = transcribe_audio(audio_file_path, use_openai=use_openai)

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
