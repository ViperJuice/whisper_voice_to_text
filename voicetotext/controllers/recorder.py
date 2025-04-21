"""
Recording controller for voice-to-text application.
Handles audio recording and error management.
"""

import time
import threading
import numpy as np
import pyaudio
import wave
import sys
import os
import ctypes
import logging

# Silence ALSA lib warnings to stderr
try:
    asound = ctypes.cdll.LoadLibrary('libasound.so')
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)

    def py_error_handler(filename, line, function, err, fmt):
        return

    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound.snd_lib_error_set_handler(c_error_handler)
except Exception:
    pass

logger = logging.getLogger(__name__)

class RecorderController:
    """Controller for recording audio from microphone"""
    
    def __init__(self, app=None):
        """Initialize the recorder controller"""
        # Store the app reference for callbacks
        self.app = app
        
        self.is_recording = False
        self.frames = []
        self.audio_data = None  # Store the raw audio data after recording finishes
        self.sample_rate = 16000  # Sample rate used for recording and processing
        
        # Audio recording parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Standard for speech recognition
        self.chunk = 1024
        self.target_file = "output.wav"  # Default output file name
        
        # PyAudio instance
        self.p = None
        
        self.stop_recording_flag = False
        self.config = {}
        
        # Print debug info about app reference
        if self.app is not None:
            print("RecorderController initialized with app reference")
        else:
            print("RecorderController initialized without app reference")
        
        # Set default recording config
        if "recording" not in self.config:
            self.config["recording"] = {
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "format": self.format,
                "chunk": self.chunk
            }
    
    
    def start_recording(self, target_file=None):
        """Start recording audio from microphone"""
        if self.is_recording:
            print("[Recorder] Already recording")
            return self.frames
            
        self.is_recording = True
        self.frames = []
        self.stop_recording_flag = False
        
        if target_file:
            self.target_file = target_file
            
        print(f"[Recorder] Starting audio capture")
        logger.debug("PyAudio FORMAT=%s, CHANNELS=%s, RATE=%s, CHUNK=%s", self.format, self.channels, self.rate, self.chunk)
        
        try:
            self.p = pyaudio.PyAudio()
            
            # Debug available audio devices
            info = self.p.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            print(f"[Recorder] *** DEBUG: Found {numdevices} audio devices:")
            for i in range(0, numdevices):
                device_info = self.p.get_device_info_by_host_api_device_index(0, i)
                if device_info.get('maxInputChannels') > 0:
                    print(f"[Recorder] *** DEBUG: Input Device id {i} - {device_info.get('name')}")

            # Try to open the default input device
            print(f"[Recorder] *** DEBUG: Opening default audio input device")
            stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            print("[Recorder] Audio stream opened. Capturing audio...")
            
            # Main recording loop - stop condition is now is_recording flag
            frame_count = 0
            try:
                while self.is_recording:
                    try:
                        data = stream.read(self.chunk, exception_on_overflow=False)
                        self.frames.append(data)
                        frame_count += 1
                        if frame_count % 10 == 0:  # Log every 10 frames to avoid excessive output
                            print(f"[Recorder] *** DEBUG: Captured frame {frame_count}, total audio length: {frame_count * self.chunk / self.rate:.2f}s")
                    except IOError as e:
                        if self.is_recording: # Only print errors if we expect to be recording
                             print(f"Warning: Audio stream read error - {e}")
                        continue # Try to continue if possible
                    except Exception as e:
                        print(f"Error during recording loop: {e}")
                        break
            except KeyboardInterrupt:
                print("\nKeyboard interrupt during recording loop")
                self.is_recording = False # Ensure flag is set
            finally:
                stream.stop_stream()
                stream.close()
                self.p.terminate()
                
            # Convert frames to raw audio data
            if self.frames:
                self.audio_data = b''.join(self.frames)
                print(f"[Recorder] Recording loop finished.")
                print(f"[Recorder] Stopped after {len(self.frames) * self.chunk / self.rate:.2f} seconds")
                print(f"[Recorder] Captured {len(self.frames)} audio frames")
                print(f"[Recorder] Final audio data size: {len(self.audio_data)} bytes")
                print("[Recorder] Audio capture complete.")
            else:
                print("[Recorder] No audio frames captured")
                print("[Recorder] *** DEBUG: is_recording flag was set to {self.is_recording} at end of loop")
                self.audio_data = None
                
        except Exception as e:
            print(f"[Recorder] Error during recording: {e}")
            import traceback
            traceback.print_exc()
            self.is_recording = False
            self.audio_data = None
            
        return self.frames

    def stop_recording(self):
        """Stop the recording process by setting the flag."""
        if self.is_recording:
             print("Recorder: Received stop signal.")
             self.is_recording = False # Signal the recording loop to stop 