"""
Audio recording and processing functionality for the voice to text application.
"""

import os
import wave
import time
import pyaudio
import tempfile
import numpy as np
import logging

# -------------------------------------------------------------
# Logging
# -------------------------------------------------------------
logger = logging.getLogger(__name__)

def _info_print(*args, **kwargs):  # type: ignore[var-annotated]
    logger.info(" ".join(str(a) for a in args))

# Replace print within this file
print = _info_print  # type: ignore[assignment]

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper expects 16kHz audio
CHUNK = 1024 * 4

# Global audio variables
audio = None
stream = None

def initialize_audio():
    """Initialize the audio system"""
    global audio
    audio = pyaudio.PyAudio()
    return audio

def get_temp_dir():
    """Get the temporary directory for storing recordings"""
    temp_base_dir = os.path.join(tempfile.gettempdir(), "whisper_transcripts")
    os.makedirs(temp_base_dir, exist_ok=True)
    return temp_base_dir

def cleanup_old_files(temp_base_dir):
    """Clean up old audio and transcript files, keeping only the 5 most recent"""
    try:
        # Get all WAV files in the temp directory
        wav_files = [f for f in os.listdir(temp_base_dir) if f.endswith('.wav')]
        wav_files.sort(reverse=True)  # Sort newest first
        
        print(f"Checking for old files in {temp_base_dir}")
        
        # Keep only the 5 newest files
        for old_file in wav_files[5:]:
            file_path = os.path.join(temp_base_dir, old_file)
            os.remove(file_path)
            print(f"🧹 Cleaned up old file: {file_path}")
            
        # Do the same for transcript files
        txt_files = [f for f in os.listdir(temp_base_dir) if f.endswith('.txt')]
        txt_files.sort(reverse=True)  # Sort newest first
        
        # Keep only the 5 newest files
        for old_file in txt_files[5:]:
            file_path = os.path.join(temp_base_dir, old_file)
            os.remove(file_path)
            print(f"🧹 Cleaned up old file: {file_path}")
            
        if len(wav_files) <= 5 and len(txt_files) <= 5:
            print(f"No old files to clean up (keeping 5 most recent)")
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

def save_audio(frames, temp_base_dir):
    """Save recorded audio to a WAV file"""
    if not frames:
        print("⚠️ No frames to save")
        return None, int(time.time())
        
    timestamp = int(time.time())
    filename = os.path.join(temp_base_dir, f"recording_{timestamp}.wav")
    print(f"💾 Saving audio to {filename}...")
    
    try:
        # Join frames and check if we have data
        audio_bytes = b''.join(frames)
        if len(audio_bytes) < 1024:
            print(f"⚠️ Very short audio recording: {len(audio_bytes)} bytes")
            # Still save but warn the user
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(audio_bytes)
        return filename, timestamp
    except Exception as e:
        print(f"Error saving audio: {e}")
        import traceback
        traceback.print_exc()
        return None, timestamp

def save_transcript(text, temp_base_dir, timestamp):
    """Save transcript to a text file"""
    transcript_file = os.path.join(temp_base_dir, f"transcript_{timestamp}.txt")
    try:
        with open(transcript_file, 'w') as f:
            f.write(text)
        print(f"📄 Transcript saved to {transcript_file}")
        return transcript_file
    except Exception as e:
        print(f"Error saving transcript: {e}")
        return None

def frames_to_audio_data(frames):
    """Convert frames to numpy array for Whisper processing"""
    try:
        if not frames:
            print("⚠️ No audio frames recorded")
            return np.array([], dtype=np.float32)
            
        # Join frames and convert to numpy array
        audio_bytes = b''.join(frames)
        
        # Check if we have enough audio data
        if len(audio_bytes) < 1024:
            print(f"⚠️ Very short audio recording: {len(audio_bytes)} bytes")
            
        # Convert to numpy array
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Check if audio is mostly silence
        audio_max = abs(audio_np).max()
        if audio_max < 0.01:
            print(f"⚠️ Very quiet audio: peak amplitude {audio_max:.4f}")
        
        return audio_np
    except Exception as e:
        print(f"❌ Error converting audio frames: {e}")
        import traceback
        traceback.print_exc()
        # Return empty array as fallback
        return np.array([], dtype=np.float32)

def cleanup_audio_resources():
    """Clean up audio resources"""
    global audio, stream
    
    if stream:
        try:
            stream.stop_stream()
            stream.close()
            stream = None
        except Exception as e:
            print(f"Error cleaning up stream: {e}")
    
    if audio:
        try:
            audio.terminate()
            audio = None
        except Exception as e:
            print(f"Error terminating audio: {e}") 