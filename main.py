import pyaudio
import wave
import numpy as np
import whisper
import time
import concurrent.futures
import pyperclip
import argparse
import sys
import tempfile
import os
import subprocess

# Settings
SAMPLE_RATE = 16000  # Sample rate in Hz
CHUNK_SIZE = 1024  # Buffer size
SILENCE_THRESHOLD = 500  # Adjust this threshold based on your environment
OUTPUT_FILE = "recorded_audio.wav"

# Load the model in a separate thread
model = None

def load_whisper_model():
    global model
    model = whisper.load_model("turbo")
    print("Model loaded.", file=sys.stderr)

def is_silent(data):
    audio_data = np.frombuffer(data, dtype=np.int16)
    return np.max(audio_data) < SILENCE_THRESHOLD

def record_audio(silence_duration):
    audio = pyaudio.PyAudio()

    # Open a stream for recording
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                        input=True, frames_per_buffer=CHUNK_SIZE)

    print("Recording...", file=sys.stderr)

    frames = []
    silence_start_time = None

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)

            if is_silent(data):
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time >= silence_duration:
                    print("Silence detected. Stopping recording and preparing transcription.", file=sys.stderr)
                    break
            else:
                silence_start_time = None
    finally:
        print("Finished recording.", file=sys.stderr)
        stream.stop_stream()
        stream.close()
        audio.terminate()

    # Save the recorded audio to a file
    with wave.open(OUTPUT_FILE, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

    return OUTPUT_FILE

def transcribe_audio(file_path, to_clipboard=False, to_edit=False):
    if model is None:
        print("Model is still loading. Please wait...", file=sys.stderr)
        # Ensure the model is loaded
        time.sleep(1)
    result = model.transcribe(file_path)
    
    transcription_text = result["text"]
    print(transcription_text)
    
    if to_edit:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(transcription_text)
            temp_file.flush()

        temp_file.close()
        # Get the file's last modified time before opening
        last_modified_time = os.path.getmtime(temp_file_name)

        # Open the temporary file in a new VS Code window and wait for the process to finish
        print(f"Opening {temp_file_name} in a new VS Code window...", file=sys.stderr)
        subprocess.run(["code", "-n", "--wait", temp_file_name], check=True)  # Open in a new VS Code window and wait

        # Check the modified time again after VS Code is closed
        new_modified_time = os.path.getmtime(temp_file_name)

        # If the file was saved (even if unchanged), the modified time will differ
        if new_modified_time != last_modified_time:
            # Read the content and copy it to the clipboard
            with open(temp_file_name, 'r') as file:
                new_content = file.read()
            pyperclip.copy(new_content)
            print("Transcription copied to clipboard.", file=sys.stderr)
        else:
            print("File closed without saving, nothing copied.", file=sys.stderr)

        # Clean up temporary file
        os.remove(temp_file_name)
        print(f"Temporary file {temp_file_name} removed.", file=sys.stderr)
    elif to_clipboard:
        pyperclip.copy(transcription_text)
        print("Transcription copied to clipboard.", file=sys.stderr)
    else:
        print("Transcription:")
        print(transcription_text)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Record audio and transcribe when silence is detected.")
    parser.add_argument(
        "--silence_duration",
        type=int,
        default=5,
        help="Number of seconds of silence required to stop recording and transcribe (default: 5 seconds)"
    )
    parser.add_argument(
        "-c", "--clipboard", action="store_true", help="Copy the transcription to clipboard."
    )
    parser.add_argument(
        "-e", "--edit", action="store_true", help="Open the transcription in VS Code for editing before copying to clipboard."
    )
    
    args = parser.parse_args()
    
    # Use the silence_duration argument
    silence_duration = args.silence_duration
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Start model loading and audio recording simultaneously
        model_future = executor.submit(load_whisper_model)
        audio_file = record_audio(silence_duration)
        
        # Wait for the model to load, if it hasn't finished by the time recording is done
        model_future.result()
        transcribe_audio(audio_file, to_clipboard=args.clipboard, to_edit=args.edit)