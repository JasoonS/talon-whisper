# Use the NVIDIA CUDA runtime base image

#### This doesn't work unfortunately.

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install Python, pip, and necessary development libraries
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev build-essential libasound2-dev portaudio19-dev alsa-base alsa-utils && \
    pip3 install --upgrade pip

# Copy the requirements file
COPY requirements.txt .

# Install the required Python packages
RUN pip3 install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV FLASK_PORT=5006
ENV BASE_FOLDER_FOR_RECORDINGS=/app/recordings

# Add the container user to the audio group - so that it can detect usb microphones too.
RUN usermod -aG audio root

# Expose the Flask port
EXPOSE 5006

# RUN apt-get install -y alsa-utils

# COPY .asoundrc /root/.asoundrc

# Set environment variables
ENV OPENAI_API_KEY=REMOVED
ENV FLASK_PORT=5006
ENV BASE_FOLDER_FOR_RECORDINGS=/app/recordings
ENV USE_LOCAL_MODEL=true
ENV LANGUAGE=en
ENV WHISPER_CACHE_DIR=/root/.cache/whisper

# Run the application
CMD ["python3", "main.py"]
