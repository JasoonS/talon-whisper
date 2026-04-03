import time
from pathlib import Path
import os
import sys
import subprocess
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
import logging

# Setup logging so that errors are recorded
logging.basicConfig(level=logging.INFO)

class Settings(BaseSettings):
    openai_api_key: str
    flask_port: int = 5006
    base_folder_for_recordings: str
    use_local_model: bool = False
    language: str = "en"
    use_faster_whisper: bool = False
    faster_whisper_model: str = "large-v3"
    whisper_model: str = "medium"
    max_duration_for_local_model: int = 60
    max_duration_for_openai: int = 200

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Load settings
settings = Settings()

if not os.path.exists(settings.base_folder_for_recordings):
    os.makedirs(settings.base_folder_for_recordings)
    logging.info(f"Folder for recordings created at: {settings.base_folder_for_recordings}")

api_key = settings.openai_api_key
flask_port = settings.flask_port

logging.info(f"API Key: {api_key}")
logging.info(f"Flask Port: {flask_port}")

def get_best_device():
    """Return the best available torch device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def print_gpu_memory_info():
    if torch.cuda.is_available():
        logging.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logging.info(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logging.info(f"Cached GPU memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        logging.info(f"Free GPU memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated() - torch.cuda.memory_reserved()) / 1e9:.2f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logging.info("Using Apple Silicon GPU (MPS).")
    else:
        logging.info("No GPU available. Running on CPU.")

# Global lock for protecting shared audio_data
audio_data_lock = threading.Lock()

# Global variables to control the recording state
is_recording = False
audio_file_path = None
recording_thread = None
audio_data = []  # This list will hold chunks of recorded data

# Max recording length safeguard (seconds)
MAX_RECORDING_SECONDS = 120

# Auto-stop and pre-saved artifacts
auto_stop_triggered = False
auto_saved_mp3_path = None
auto_saved_txt_path = None
auto_saved_transcription = None

def send_notification(title: str, message: str) -> None:
    """Send a desktop notification (macOS via osascript, Linux via notify-send)."""
    try:
        if sys.platform == "darwin":
            escaped_msg = message.replace('"', '\\"')
            escaped_title = title.replace('"', '\\"')
            subprocess.run(
                ["osascript", "-e", f'display notification "{escaped_msg}" with title "{escaped_title}"'],
                check=False,
            )
        else:
            subprocess.run(["notify-send", title, message], check=False)
    except Exception as e:
        logging.warning(f"Notification failed: {e}")
    # Also log and print for visibility when notifications are off
    logging.info(f"[NOTIFICATION] {title}: {message}")
    print(f"[NOTIFICATION] {title}: {message}")

# Load the appropriate model or client
if settings.use_local_model:
    if settings.use_faster_whisper:
        logging.info(f"Using faster-whisper model: {settings.faster_whisper_model}")
        # faster-whisper (CTranslate2) only supports cuda and cpu, not mps
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        logging.info(f"Using device: {device}")
        
        try:
            # Try to load the model with offline mode first
            logging.info("Attempting to load model in offline mode...")
            model = WhisperModel(
                settings.faster_whisper_model, 
                device=device, 
                compute_type=compute_type,
                local_files_only=True  # Force offline mode
            )
            logging.info(f"Model loaded successfully in offline mode")
        except Exception as e:
            logging.warning(f"Failed to load model in offline mode: {e}")
            logging.info("Attempting to load model with online verification...")
            try:
                model = WhisperModel(
                    settings.faster_whisper_model, 
                    device=device, 
                    compute_type=compute_type
                )
                logging.info(f"Model loaded successfully with online verification")
            except Exception as e2:
                logging.error(f"Failed to load model: {e2}")
                logging.info("Falling back to smaller model...")
                model = WhisperModel(
                    "base", 
                    device=device, 
                    compute_type=compute_type,
                    local_files_only=True
                )
                logging.info(f"Fallback model loaded successfully")
        
        print_gpu_memory_info()
    else:
        import whisper
        logging.info(f"Using Whisper model: {settings.whisper_model}")
        device = get_best_device()
        logging.info(f"Using device: {device}")
        try:
            model = whisper.load_model(settings.whisper_model).to(device)
        except torch.cuda.OutOfMemoryError:
            logging.info("GPU memory insufficient, falling back to CPU")
            device = "cpu"
            model = whisper.load_model("small").to(device)
        logging.info(f"Using device: {device}")
        print_gpu_memory_info()
# else: # always use openai for longer transcription
from openai import OpenAI
client = OpenAI(api_key=settings.openai_api_key)

app = Flask(__name__)


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
    "MVX2U",
    "Shure MVX2U Mono",
    "C03U",
    # Add more preferred microphone names here
]

# Get the preferred device index
device_index = get_preferred_device(preferred_microphones)

# Dummy implementation; replace with your actual audio recording logic.
def record_audio_continuously(max_duration=999999, device_index=None):
    global is_recording, audio_data, auto_stop_triggered, auto_saved_mp3_path, auto_saved_txt_path, auto_saved_transcription

    samplerate = 16000
    is_recording = True
    auto_limit_hit = False

    print("Recording started...")
    with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16', device=device_index) as stream:
        start_time = time.time()
        while is_recording:
            data, _ = stream.read(1024)
            audio_data.append(data)
            elapsed = time.time() - start_time
            if elapsed >= max_duration:
                break
            if elapsed >= MAX_RECORDING_SECONDS:
                auto_limit_hit = True
                is_recording = False
                break

    print("Recording stopped.")

    # If we hit the auto limit, process and save immediately
    if auto_limit_hit:
        try:
            with audio_data_lock:
                data_copy = list(audio_data)
                audio_data.clear()

            if not data_copy:
                logging.error("No audio data captured at auto-stop; nothing to save.")
                send_notification("Recording auto-stopped", "No audio data captured; nothing saved.")
                return

            mp3_file_path = Path(settings.base_folder_for_recordings) / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}-{os.urandom(4).hex()}.mp3"
            txt_file_path = mp3_file_path.with_suffix(".txt")

            try:
                audio_file_path_local = save_to_mp3(data_copy, file_path=str(mp3_file_path))
                duration = len(np.concatenate(data_copy)) / 16000
            except Exception as e:
                logging.error(f"Error saving MP3 file at auto-stop: {e}")
                send_notification("Recording auto-stopped", "Failed to save audio file.")
                return

            use_openai = duration > settings.max_duration_for_local_model
            transcription = ""
            try:
                transcription = transcribe_audio(audio_file_path_local, use_openai=use_openai)
            except Exception as e:
                logging.error(f"Error during transcription at auto-stop: {e}")

            try:
                with open(txt_file_path, "w") as f:
                    f.write(transcription if transcription is not None else "")
            except Exception as e:
                logging.error(f"Error saving transcription file at auto-stop: {e}")

            auto_stop_triggered = True
            auto_saved_mp3_path = str(mp3_file_path)
            auto_saved_txt_path = str(txt_file_path)
            auto_saved_transcription = transcription

            send_notification(
                "Recording auto-stopped at 120s",
                "Audio and transcription saved. Press stop to retrieve."
            )
        except Exception as e:
            logging.error(f"Unhandled error in auto-stop processing: {e}")


def save_to_mp3(audio_data, file_path=None, samplerate=16000):
    # Validate that there is audio data
    if not audio_data:
        raise ValueError("No audio data recorded; aborting save.")
    # If no file_path specified, use a temporary file
    if file_path is None:
        file_path = os.path.join(tempfile.gettempdir(), "recording.mp3")
    audio_concatenated = np.concatenate(audio_data)
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)  # Assuming mono; adjust as needed
        wf.setsampwidth(2)  # Assuming 16-bit audio; adjust if needed
        wf.setframerate(samplerate)
        wf.writeframes(audio_concatenated.tobytes())
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
            if settings.use_faster_whisper:
                logging.info("Using faster-whisper batched transcription for improved speed")
                from faster_whisper import BatchedInferencePipeline
                batched_model = BatchedInferencePipeline(model=model)
                segments, info = batched_model.transcribe(
                    audio_file,
                    beam_size=5,
                    language=settings.language,
                    initial_prompt=prompt,
                    batch_size=16
                )
                transcription = " ".join([segment.text for segment in segments])
            else:
                import whisper
                audio = whisper.load_audio(audio_file)
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio).to(model.device)
                _, probs = model.detect_language(mel)
                options = whisper.DecodingOptions(
                    language=settings.language,
                    fp16=(device == "cuda"),
                    prompt=prompt
                )
                result = whisper.decode(model, mel, options)
                transcription = result.text
        else:
            logging.info("Using OpenAI API for transcription")
            with open(audio_file, "rb") as audio_file_obj:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file_obj,
                    response_format="text",
                    language=settings.language,
                    prompt=prompt
                )

        transcription = transcription.lstrip()
        if not transcription.endswith(" "):
            ## I added this so I can continuously dictate, and spaces are added between sentences without extra keyboard input.
            transcription += " "
        return transcription
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return ""

# # If a transcription is needed, uncomment this and run the script with the correct path to the audio file.
# transcription = transcribe_audio("/home/jasoons/Music/recordings/2025-05-22-17:35:00-a4a73bc0.mp3", use_openai=False)
# print(transcription)

def test_api_connection_with_recording():
    global is_recording, audio_data
    try:
        record_audio_continuously(max_duration=2)
        wav_file = save_to_mp3(audio_data)
        is_recording = False
        transcription = transcribe_audio(wav_file)
        os.remove(wav_file)
        if transcription:
            logging.info("API connection and transcription test successful!")
            logging.info(f"Transcription: {transcription}")
        else:
            logging.error("API connection test failed or no transcription returned.")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Error during API connection test: {e}")
        sys.exit(1)

@app.route("/start", methods=["POST"])
def start_recording():
    global is_recording, recording_thread, audio_data
    global auto_stop_triggered, auto_saved_mp3_path, auto_saved_txt_path, auto_saved_transcription
    if is_recording:
        return jsonify({"message": "Recording is already in progress!"}), 400
    with audio_data_lock:
        audio_data.clear()  # Clear any previous data safely
    # Reset auto-stop artifacts for a fresh session
    auto_stop_triggered = False
    auto_saved_mp3_path = None
    auto_saved_txt_path = None
    auto_saved_transcription = None
    is_recording = True
    recording_thread = threading.Thread(target=record_audio_continuously)
    recording_thread.start()
    return jsonify({"message": "Recording started!"})

@app.route("/stop", methods=["POST"])
def stop_recording():
    ### For testing, uncomment the following and comment out the rest of the function.
    # result = """abcdefghijklmnopqrstuvwxyz 1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ ABCDEFGHIJKLMNOPQRSTUVWXYZ ABCDEFGHIJKLMNOPQRSTUVWXYZ ABCDEFGHIJKLMNOPQRSTUVWXYZ ABCDEFGHIJKLMNOPQRSTUVWXYZ ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890  1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890"""

    global is_recording, recording_thread, audio_file_path
    global auto_stop_triggered, auto_saved_mp3_path, auto_saved_txt_path, auto_saved_transcription
    if not is_recording:
        # If an auto-stop occurred, return the pre-saved result immediately
        if auto_stop_triggered and auto_saved_transcription is not None:
            response = {"transcription": auto_saved_transcription, "auto_stopped": True}
            # Reset state after serving the result
            auto_stop_triggered = False
            return jsonify(response)
        return jsonify({"message": "No recording is currently in progress!"}), 400

    # Stop the recording and wait for the thread to finish
    is_recording = False
    if recording_thread is not None and recording_thread.is_alive():
        recording_thread.join()

    # Prepare paths for saving the audio and transcription files
    mp3_file_path = Path(settings.base_folder_for_recordings) / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}-{os.urandom(4).hex()}.mp3"
    txt_file_path = mp3_file_path.with_suffix(".txt")

    # If auto-stop already processed and saved artifacts, reuse them
    if auto_stop_triggered and auto_saved_mp3_path and auto_saved_txt_path is not None:
        mp3_file_path = Path(auto_saved_mp3_path)
        txt_file_path = Path(auto_saved_txt_path)
        try:
            with open(txt_file_path, "r") as f:
                transcription = f.read()
        except Exception as e:
            logging.error(f"Error reading pre-saved transcription: {e}")
            transcription = auto_saved_transcription or ""
        # Reset auto-stop state after use
        auto_stop_triggered = False
        print(f"{transcription}")
        if transcription:
            return jsonify({"transcription": transcription, "auto_stopped": True})
        else:
            return jsonify({"message": "No transcription result produced."}), 200

    # Safely copy and then clear audio_data using the lock
    with audio_data_lock:
        data_copy = list(audio_data)
        audio_data.clear()

    # Save the recorded audio to an MP3 file (if data exists)
    try:
        audio_file_path = save_to_mp3(data_copy, file_path=str(mp3_file_path))
    except Exception as e:
        logging.error(f"Error saving MP3 file: {e}")
        return jsonify({"message": str(e)}), 400

    logging.info(f"Audio file saved to: {audio_file_path}")

    # Calculate the recording duration
    try:
        duration = len(np.concatenate(data_copy)) / 16000  # Assuming 16000 Hz sample rate
    except Exception as e:
        logging.error(f"Error calculating duration: {e}")
        return jsonify({"message": "Error processing audio data."}), 400

    logging.info(f"Recording duration: {duration:.2f} seconds")

    # Decide whether to use the local model or the OpenAI API
    use_openai = duration > settings.max_duration_for_local_model

    # If the recording is longer than 2 minutes while using OpenAI, prompt for confirmation
    if use_openai and duration > settings.max_duration_for_openai:
        logging.warning("The recording is longer than 2 minutes and can be costly with OpenAI API.")
        user_input = input("Did you intend to record such a long audio? Type 'yes' to proceed, or any other key to abort transcription: ")
        if user_input.strip().lower() not in ["yes", "y"]:
            logging.info("Transcription aborted by user.")
            return jsonify({"message": "Transcription aborted by user due to long recording duration."}), 400

    # Attempt transcription with proper exception handling
    transcription = ""
    try:
        transcription = transcribe_audio(audio_file_path, use_openai=use_openai)
    except Exception as e:
        logging.error(f"Error during transcription process: {e}")

    print(f"{transcription}")

    # Save the transcription to a text file
    try:
        with open(txt_file_path, "w") as f:
            f.write(transcription if transcription is not None else "")
    except Exception as e:
        logging.error(f"Error saving transcription file: {e}")

    # Instead of returning 500 if no transcription result is found, return a 200 with a friendly message.
    if transcription:
        return jsonify({"transcription": transcription})
    else:
        logging.info("No transcription result produced.")
        return jsonify({"message": "No transcription result produced."}), 200


if __name__ == "__main__":
    # Optionally, you can test the connection before starting the server:
    # test_api_connection_with_recording()
    app.run(host="0.0.0.0", port=int(flask_port))
