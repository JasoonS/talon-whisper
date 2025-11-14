#!/usr/bin/env python3
"""
Simple CLI utility to transcribe audio files from the 'input-sounds' folder.
Processes all audio files and creates corresponding text files in the 'output-text' folder.
"""

import sys
import logging
from pathlib import Path
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline
from pydantic_settings import BaseSettings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Settings(BaseSettings):
    openai_api_key: str = ""
    use_local_model: bool = True
    language: str = "en"
    use_faster_whisper: bool = True
    faster_whisper_model: str = "large-v3"
    whisper_model: str = "medium"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields from .env file

# Load settings
settings = Settings()

# Directories
SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR / "input-sounds"
OUTPUT_DIR = SCRIPT_DIR / "output-text"

# Ensure directories exist
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Supported audio formats
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.opus', '.webm', '.mp4'}

def read_prompt_file(file_path='prompt.txt'):
    """Read the prompt file for transcription context."""
    prompt_path = SCRIPT_DIR / file_path
    try:
        with open(prompt_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        logging.warning(f"{file_path} not found. Using empty prompt.")
        return ""

def load_model():
    """Load the appropriate Whisper model."""
    if settings.use_local_model and settings.use_faster_whisper:
        logging.info(f"Loading faster-whisper model: {settings.faster_whisper_model}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        logging.info(f"Using device: {device}, compute_type: {compute_type}")

        # Try different models in order of preference
        models_to_try = [
            (settings.faster_whisper_model, device, compute_type, True),  # Requested model, offline
            (settings.faster_whisper_model, device, compute_type, False), # Requested model, online
            (settings.faster_whisper_model, "cpu", "int8", True),         # Requested model, CPU, offline
            (settings.faster_whisper_model, "cpu", "int8", False),        # Requested model, CPU, online
            ("small", device, compute_type, True),                        # Smaller model, offline
            ("small", "cpu", "int8", True),                               # Smaller model, CPU, offline
        ]

        for model_name, dev, comp_type, offline in models_to_try:
            try:
                mode_str = "offline" if offline else "online"
                logging.info(f"Trying {model_name} on {dev} ({comp_type}) - {mode_str} mode...")
                model = WhisperModel(
                    model_name,
                    device=dev,
                    compute_type=comp_type,
                    local_files_only=offline
                )
                logging.info(f"✓ Model loaded successfully: {model_name} on {dev}")
                return model
            except Exception as e:
                logging.warning(f"✗ Failed: {e}")
                continue

        # If all else fails, use OpenAI API
        logging.error("Could not load any local model. Falling back to OpenAI API.")
        from openai import OpenAI
        if not settings.openai_api_key:
            logging.error("OpenAI API key not found. Please set it in .env file.")
            sys.exit(1)
        logging.info("Using OpenAI API")
        return OpenAI(api_key=settings.openai_api_key)
    else:
        # Fall back to OpenAI API if needed
        from openai import OpenAI
        if not settings.openai_api_key:
            logging.error("OpenAI API key not found. Please set it in .env file or use local model.")
            sys.exit(1)
        return OpenAI(api_key=settings.openai_api_key)

def transcribe_audio(audio_file: Path, model, prompt: str) -> str:
    """Transcribe an audio file using the loaded model."""
    try:
        # Check if model is a WhisperModel (local) or OpenAI client
        if isinstance(model, WhisperModel):
            logging.info(f"Transcribing with faster-whisper: {audio_file.name}")
            batched_model = BatchedInferencePipeline(model=model)
            segments, info = batched_model.transcribe(
                str(audio_file),
                beam_size=5,
                language=settings.language,
                initial_prompt=prompt,
                batch_size=16
            )
            transcription = " ".join([segment.text for segment in segments])
        else:
            # Using OpenAI API
            logging.info(f"Transcribing with OpenAI API: {audio_file.name}")
            with open(audio_file, "rb") as audio_file_obj:
                transcription = model.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file_obj,
                    response_format="text",
                    language=settings.language,
                    prompt=prompt
                )

        return transcription.strip()
    except Exception as e:
        logging.error(f"Error transcribing {audio_file.name}: {e}")
        return ""

def get_audio_files():
    """Get all audio files from the input directory."""
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(INPUT_DIR.glob(f"*{ext}"))
    return sorted(audio_files)

def process_files():
    """Process all audio files in the input directory."""
    audio_files = get_audio_files()

    if not audio_files:
        logging.info(f"No audio files found in {INPUT_DIR}")
        logging.info(f"Supported formats: {', '.join(AUDIO_EXTENSIONS)}")
        return

    logging.info(f"Found {len(audio_files)} audio file(s) to process")

    # Load model once
    logging.info("Loading transcription model...")
    model = load_model()
    prompt = read_prompt_file()

    # Process each file
    processed = 0
    skipped = 0
    failed = 0

    for audio_file in audio_files:
        output_file = OUTPUT_DIR / f"{audio_file.stem}.txt"

        # Skip if already transcribed
        if output_file.exists():
            logging.info(f"Skipping {audio_file.name} (already transcribed)")
            skipped += 1
            continue

        logging.info(f"Processing: {audio_file.name}")
        transcription = transcribe_audio(audio_file, model, prompt)

        if transcription:
            # Save transcription
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                logging.info(f"Saved: {output_file.name}")
                processed += 1
            except Exception as e:
                logging.error(f"Error saving transcription for {audio_file.name}: {e}")
                failed += 1
        else:
            logging.error(f"Failed to transcribe: {audio_file.name}")
            failed += 1

    # Summary
    logging.info("=" * 60)
    logging.info(f"Summary: {processed} processed, {skipped} skipped, {failed} failed")
    logging.info("=" * 60)

def main():
    """Main entry point."""
    logging.info("=" * 60)
    logging.info("Audio Transcription Utility")
    logging.info("=" * 60)
    logging.info(f"Input directory: {INPUT_DIR}")
    logging.info(f"Output directory: {OUTPUT_DIR}")
    logging.info("=" * 60)

    try:
        process_files()
    except KeyboardInterrupt:
        logging.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
