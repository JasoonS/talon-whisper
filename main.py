from openai import OpenAI
import os
import sys
import sounddevice as sd
import numpy as np
import tempfile
import wave
import threading
from flask import Flask, jsonify

# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("API key is missing! Set the environment variable OPENAI_API_KEY.")
    sys.exit(1)

client = OpenAI(api_key=api_key)

app = Flask(__name__)

# Global variables to control the recording state
is_recording = False
audio_file_path = None
recording_thread = None
audio_data = []

def record_audio(duration=1, samplerate=16000):
    print("Recording started for test...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()  # Wait until recording is finished
    return recording

def save_to_wav(audio_data, samplerate=16000):
    # Save the recorded audio data to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        with wave.open(temp_wav.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes = 16 bits
            wf.setframerate(samplerate)
            wf.writeframes(np.concatenate(audio_data).tobytes())
        return temp_wav.name

def transcribe_audio(wav_file):
    # Transcribe the audio using OpenAI Whisper API
    try:
        with open(wav_file, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="text"
            )
            print(transcript)
            print(type(transcript))
            return transcript
            # return transcript.text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def test_api_connection_with_recording():
    try:
        # Perform a 1-second test recording
        audio_data = record_audio(duration=1)
        wav_file = save_to_wav(audio_data)

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
    global is_recording, recording_thread

    if is_recording:
        return jsonify({"message": "Recording is already in progress!"}), 400

    # Start recording in a background thread
    is_recording = True
    recording_thread = threading.Thread(target=record_audio)
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

    # Save the recorded audio to a temporary WAV file
    audio_file_path = save_to_wav(audio_data)

    # Transcribe the saved audio file
    transcription = transcribe_audio(audio_file_path)
    os.remove(audio_file_path)  # Delete the temporary file after transcription
    audio_file_path = None

    if transcription:
        return jsonify({"transcription": transcription})
    else:
        return jsonify({"message": "Error during transcription!"}), 500

if __name__ == "__main__":
    # Test the API connection before starting the server
    test_api_connection_with_recording()

    # Start the Flask server
    app.run(host="0.0.0.0", port=5005)