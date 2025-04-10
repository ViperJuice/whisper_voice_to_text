#!/usr/bin/env python3
"""
Voice to Text with Dual Mode for Pop!_OS
- Super+Space: Simple transcription with basic cleanup
- Super+Alt+Space: Enhanced transcription with LLM refinement
"""
import os
import tempfile
import time
import threading
import subprocess
import re
import wave
import json
import numpy as np
import pyaudio
import torch
import whisper
import requests
from pynput import keyboard
from pynput.keyboard import Controller, Key, Listener

# Initialize keyboard controller for typing at cursor position
keyboard_controller = Controller()

# Load Whisper model with device detection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# PyAudio setup - using proper format for Whisper
CHUNK = 1024 * 4
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper expects 16kHz audio

p = pyaudio.PyAudio()

# Recording state
is_recording = False
is_running = True
frames = []
stop_recording_event = threading.Event()
use_llm_enhancement = False

# Currently pressed keys tracking
currently_pressed_keys = set()

# LLM API Configuration
# API keys for fallback options - leave empty if not using
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LOCAL_LLM_URL = "http://localhost:11434/api/chat"  # Ollama default API endpoint

# If using Ollama, set the model name
LOCAL_LLM_MODEL = "llama3"  # Standard model name in Ollama


def get_available_ollama_models():
    """Detect available Ollama models and select the best one"""
    try:
        # Check if Ollama server is responding
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        
        if response.status_code == 200:
            models_data = response.json()
            
            if not "models" in models_data or len(models_data["models"]) == 0:
                return None, []
            
            # Get all available models
            available_models = [model["name"] for model in models_data["models"]]
            
            # Define model preference order from most to least powerful
            preferred_models = [
                # Llama 3 variants (from largest to smallest)
                "llama3:70b", "llama3:instruct", "llama3",
                # Mistral variants
                "mistral:instruct", "mistral",
                # Llama 2 variants
                "llama2:70b", "llama2", 
                # Others
                "orca-mini", "phi", "phi3"
            ]
            
            # Find the first available model from our preference list
            selected_model = None
            for model in preferred_models:
                matches = [m for m in available_models if m.startswith(model)]
                if matches:
                    selected_model = matches[0]
                    break
            
            # If no preferred model found but there are models available, use the first one
            if not selected_model and available_models:
                selected_model = available_models[0]
                
            return selected_model, available_models
            
        return None, []
    except Exception as e:
        print(f"Error detecting Ollama models: {e}")
        return None, []


def detect_gpu_capabilities():
    """Detect GPU capabilities and recommend appropriate models"""
    whisper_model_recommendation = "base"
    gpu_description = "No GPU detected"
    vram_estimate = 0
    
    try:
        # Check for CUDA availability
        if not torch.cuda.is_available():
            return whisper_model_recommendation, "CPU only", vram_estimate
        
        # Get GPU count and info
        gpu_count = torch.cuda.device_count()
        gpu_names = []
        total_vram = 0
        
        for i in range(gpu_count):
            # Get GPU name
            gpu_name = torch.cuda.get_device_name(i)
            gpu_names.append(gpu_name)
            
            # Estimate VRAM based on known GPU models
            vram_gb = 0
            
            # NVIDIA RTX 40 series
            if "4090" in gpu_name:
                vram_gb = 24
            elif "4080" in gpu_name:
                vram_gb = 16
            elif "4070 Ti" in gpu_name:
                vram_gb = 12
            elif "4070" in gpu_name:
                vram_gb = 12
            elif "4060 Ti" in gpu_name:
                vram_gb = 8
            elif "4060" in gpu_name:
                vram_gb = 8
            
            # NVIDIA RTX 30 series
            elif "3090" in gpu_name:
                vram_gb = 24
            elif "3080 Ti" in gpu_name:
                vram_gb = 12
            elif "3080" in gpu_name:
                vram_gb = 10
            elif "3070 Ti" in gpu_name:
                vram_gb = 8
            elif "3070" in gpu_name:
                vram_gb = 8
            elif "3060 Ti" in gpu_name:
                vram_gb = 8
            elif "3060" in gpu_name:
                vram_gb = 12  # Unusual case where 3060 has more VRAM than 3060 Ti
            
            # NVIDIA RTX 20 series
            elif "2080 Ti" in gpu_name:
                vram_gb = 11
            elif "2080" in gpu_name:
                vram_gb = 8
            elif "2070" in gpu_name:
                vram_gb = 8
            elif "2060" in gpu_name:
                vram_gb = 6
                
            # Default case
            else:
                # Try to extract VRAM info from name (often includes xGB)
                vram_match = re.search(r'(\d+)\s*GB', gpu_name)
                if vram_match:
                    vram_gb = int(vram_match.group(1))
                else:
                    # Conservative estimate
                    vram_gb = 4
            
            total_vram += vram_gb
        
        # Create GPU description
        if gpu_count == 1:
            gpu_description = f"{gpu_names[0]} with approximately {total_vram}GB VRAM"
        else:
            gpu_description = f"{gpu_count} GPUs ({', '.join(gpu_names)}) with approximately {total_vram}GB total VRAM"
        
        # Make recommendations based on VRAM
        if total_vram >= 36:  # Multiple high-end GPUs or single very high-end GPU
            whisper_model_recommendation = "large"
        elif total_vram >= 16:  # Single high-end GPU
            whisper_model_recommendation = "medium"
        elif total_vram >= 8:  # Mid-range GPU
            whisper_model_recommendation = "small"
        else:  # Low-end GPU
            whisper_model_recommendation = "base"
        
        return whisper_model_recommendation, gpu_description, total_vram
        
    except Exception as e:
        print(f"Error detecting GPU capabilities: {e}")
        return "base", "GPU detection failed", 0


def check_ollama_status():
    """Check if Ollama server is running and find available models"""
    try:
        # Check if Ollama server is responding
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        
        if response.status_code == 200:
            # Get available models
            selected_model, available_models = get_available_ollama_models()
            
            if selected_model:
                print(f"✓ Ollama server is running with {len(available_models)} model(s)")
                print(f"  Selected model: {selected_model}")
                if len(available_models) > 1:
                    print(f"  Other available models: {', '.join([m for m in available_models if m != selected_model])}")
                return True, selected_model, available_models
            else:
                print("✗ Ollama server is running but no models are available")
                print("\nTo download a model, run:")
                print("  ollama pull llama3")
                return False, None, []
        else:
            print("✗ Ollama server responded with an error")
            return False, None, []
    except requests.exceptions.ConnectionError:
        print("✗ Ollama server is not running")
        print("\nTo start Ollama, open a new terminal and run:")
        print("  ollama serve")
        print("\nIf Ollama is not installed, install it with:")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
        return False, None, []
    except Exception as e:
        print(f"✗ Error checking Ollama status: {e}")
        return False, None, []


def extract_refined_text(text):
    """Helper function to extract just the refined text from LLM responses"""
    # Try to extract text between "REFINED_TEXT:" and end of line/paragraph
    refined_format = re.search(r'REFINED_TEXT:\s*(.+?)(?:\n\n|$)', text, re.DOTALL)
    if refined_format:
        return refined_format.group(1).strip()
    
    # Try to find text enclosed in quotes
    quoted_text = re.search(r'"([^"]+)"', text)
    if quoted_text:
        return quoted_text.group(1)
    
    # Try to get the first paragraph that's not an explanation
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) > 10 and not line.lower().startswith(('here', 'i ', 'this is', 'the ', 'refined')):
            return line
    
    # If all else fails, just return the input
    return text


def refine_with_llm(text):
    """
    Use an LLM to refine the transcribed text with cascading fallback:
    1. Try local LLM (Ollama) first
    2. Fall back to OpenAI if local fails and API key exists
    3. Fall back to Anthropic if OpenAI fails and API key exists
    4. Fall back to algorithmic cleanup if all LLMs fail
    """
    if not text or len(text.strip()) < 5:
        return text
    
    # Start with local LLM
    try:
        print("Attempting to refine with local LLM...")
        refined_text = refine_with_local_llm(text)
        refined_text = extract_refined_text(refined_text)  # Extract just the refined part
        if refined_text and refined_text != text:
            print("✓ Local LLM refinement successful")
            return refined_text
        else:
            print("Local LLM returned incomplete or identical text, trying next option...")
    except Exception as e:
        print(f"Local LLM error: {e}")
        print("Falling back to next option...")
    
    # Try OpenAI if available
    if OPENAI_API_KEY:
        try:
            print("Attempting to refine with OpenAI API...")
            refined_text = refine_with_openai(text)
            refined_text = extract_refined_text(refined_text)  # Extract just the refined part
            if refined_text and refined_text != text:
                print("✓ OpenAI API refinement successful")
                return refined_text
            else:
                print("OpenAI API returned incomplete or identical text, trying next option...")
        except Exception as e:
            print(f"OpenAI API error: {e}")
            print("Falling back to next option...")
    else:
        print("OpenAI API key not set, skipping...")
    
    # Try Anthropic if available
    if ANTHROPIC_API_KEY:
        try:
            print("Attempting to refine with Anthropic API...")
            refined_text = refine_with_anthropic(text)
            refined_text = extract_refined_text(refined_text)  # Extract just the refined part
            if refined_text and refined_text != text:
                print("✓ Anthropic API refinement successful")
                return refined_text
            else:
                print("Anthropic API returned incomplete or identical text, falling back...")
        except Exception as e:
            print(f"Anthropic API error: {e}")
            print("Falling back to algorithmic cleanup...")
    else:
        print("Anthropic API key not set, skipping...")
    
    # If all LLM options failed, use algorithmic cleanup
    print("All LLM options exhausted or failed, using algorithmic cleanup...")
    return algorithmic_cleanup(text)


def refine_with_local_llm(text):
    """Use a local Ollama LLM server to refine text"""
    try:
        url = LOCAL_LLM_URL
        headers = {
            "Content-Type": "application/json"
        }
        
        # Use a more direct prompt structure that forces a specific format
        prompt = f"""Your task is to refine this speech-to-text transcription to be more concise and clear.
Remove filler words, repetition, and unnecessary phrases, but maintain the personal style.

Original transcription: "{text}"

Format your response EXACTLY like this: 
REFINED_TEXT: [your refined version here]

Do not include anything else in your response."""
        
        payload = {
            "model": LOCAL_LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a text refinement assistant. You will ONLY reply with 'REFINED_TEXT:' followed by your refined version."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response_data = response.json()
        
        if "message" in response_data and "content" in response_data["message"]:
            raw_text = response_data["message"]["content"].strip()
            
            # Look for the specific format marker
            if "REFINED_TEXT:" in raw_text:
                parts = raw_text.split("REFINED_TEXT:", 1)
                refined = parts[1].strip()
                # Remove any additional commentary after the refined text
                refined = refined.split("\n\n", 1)[0]
                refined = refined.strip('[]" \t\n')
                return refined
            
            # If the format marker isn't found, try to extract quoted text
            quoted_text = re.search(r'"([^"]+)"', raw_text)
            if quoted_text:
                return quoted_text.group(1)
                
            # Last resort: first paragraph
            first_paragraph = raw_text.split("\n\n")[0].strip()
            if len(first_paragraph) > 10:  # Reasonable text length
                return first_paragraph
            
            # If all extraction methods fail, just take the first 150 chars of the response
            return raw_text[:150].strip()
            
        return text
    
    except Exception as e:
        print(f"Local LLM error: {e}")
        return text


def refine_with_openai(text):
    """Use OpenAI API to refine text"""
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        prompt = f"""Your task is to refine this speech-to-text transcription to be more concise and clear.
Remove filler words, repetition, and unnecessary phrases, but maintain the personal style.

Original transcription: "{text}"

Format your response EXACTLY like this: 
REFINED_TEXT: [your refined version here]

Do not include anything else in your response."""
        
        payload = {
            "model": "o3-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a text refinement assistant. You will ONLY reply with 'REFINED_TEXT:' followed by your refined version."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 150,
            "temperature": 0.2
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        
        if "choices" in response_data and len(response_data["choices"]) > 0:
            refined_text = response_data["choices"][0]["message"]["content"].strip()
            return refined_text
        
        return text
    
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return text


def refine_with_anthropic(text):
    """Use Anthropic API to refine text"""
    try:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        }
        
        prompt = f"""Your task is to refine this speech-to-text transcription to be more concise and clear.
Remove filler words, repetition, and unnecessary phrases, but maintain the personal style.

Original transcription: "{text}"

Format your response EXACTLY like this: 
REFINED_TEXT: [your refined version here]

Do not include anything else in your response."""
        
        payload = {
            "model": "claude-instant-1",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 150,
            "temperature": 0.2
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        
        if "content" in response_data and len(response_data["content"]) > 0:
            refined_text = response_data["content"][0]["text"].strip()
            return refined_text
        
        return text
    
    except Exception as e:
        print(f"Anthropic API error: {e}")
        return text


def algorithmic_cleanup(text):
    """Basic algorithmic cleanup for simple mode"""
    if not text:
        return ""
        
    # Initial cleanup - lowercase for processing
    processed_text = text.lower()
    
    # Remove filler words and phrases
    filler_words = [
        r'\buh\b', r'\bum\b', r'\ber\b', r'\blike\b(?! to)', r'\byou know\b', 
        r'\bactually\b', r'\bbasically\b', r'\bliterally\b', r'\bi mean\b'
    ]
    
    for filler in filler_words:
        processed_text = re.sub(filler, '', processed_text)
    
    # Remove stuttering and word repetition
    processed_text = re.sub(r'\b(\w+)\s+\1\b', r'\1', processed_text)  # "the the" → "the"
    
    # Clean up spacing and multiple spaces
    processed_text = re.sub(r'\s+', ' ', processed_text)
    processed_text = processed_text.strip()
    
    # Capitalize first letter of sentences
    if processed_text:
        processed_text = processed_text[0].upper() + processed_text[1:]
    
    # Fix common punctuation issues
    processed_text = re.sub(r'\s+([.,!?;:])', r'\1', processed_text)
    processed_text = re.sub(r'([.,!?;:])(\w)', r'\1 \2', processed_text)
    
    return processed_text


def start_recording(with_llm=False):
    """Start recording audio"""
    global is_recording, frames, use_llm_enhancement
    
    if is_recording:
        return
    
    is_recording = True
    frames = []
    use_llm_enhancement = with_llm
    stop_recording_event.clear()
    
    def recording_thread():
        global frames
        
        try:
            stream = p.open(format=FORMAT,
                           channels=CHANNELS,
                           rate=RATE,
                           input=True,
                           frames_per_buffer=CHUNK)
            
            if use_llm_enhancement:
                print("Recording started with LLM enhancement... Release keys to process.")
            else:
                print("Recording started with simple cleanup... Release keys to process.")
            
            # Record until the key combination is released
            while not stop_recording_event.is_set():
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            
            print("Processing speech...")
            
            # Only process if we've recorded something meaningful
            if len(frames) > 5:  # Ensure we have at least some data (avoid empty recordings)
                # Create a temporary WAV file with proper format
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_filename = temp_file.name
                
                # Save the recorded audio to a WAV file with proper headers
                try:
                    wf = wave.open(temp_filename, 'wb')
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                    wf.close()
                    
                    # Transcribe the recorded audio
                    try:
                        result = model.transcribe(temp_filename)
                        text = result["text"]
                        
                        if text:
                            print(f"Raw transcription: {text}")
                            
                            if use_llm_enhancement:
                                # Refine the text using LLM for enhanced mode
                                refined_text = refine_with_llm(text)
                                # Make sure we're only using the actual refined text
                                refined_text = extract_refined_text(refined_text)
                                print(f"LLM refined: {refined_text}")
                            else:
                                # Use simple algorithmic cleanup for basic mode
                                refined_text = algorithmic_cleanup(text)
                                print(f"Simple cleanup: {refined_text}")
                            
                            # Type the refined text at the cursor position
                            keyboard_controller.type(refined_text + " ")
                        else:
                            print("No speech detected.")
                    except Exception as e:
                        print(f"Transcription error: {e}")
                        
                        # Try with numpy array approach as fallback
                        try:
                            print("Trying alternative transcription method...")
                            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
                            result = model.transcribe(audio_data)
                            text = result["text"]
                            
                            if text:
                                print(f"Raw transcription: {text}")
                                
                                if use_llm_enhancement:
                                    # Refine the text using LLM for enhanced mode
                                    refined_text = refine_with_llm(text)
                                    # Make sure we're only using the actual refined text
                                    refined_text = extract_refined_text(refined_text)
                                    print(f"LLM refined: {refined_text}")
                                else:
                                    # Use simple algorithmic cleanup for basic mode
                                    refined_text = algorithmic_cleanup(text)
                                    print(f"Simple cleanup: {refined_text}")
                                
                                # Type the refined text at the cursor position
                                keyboard_controller.type(refined_text + " ")
                            else:
                                print("No speech detected.")
                        except Exception as e2:
                            print(f"Alternative transcription failed: {e2}")
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_filename):
                        try:
                            os.unlink(temp_filename)
                        except:
                            pass
            else:
                print("Recording too short, ignoring.")
            
            print("Ready for next recording. Press Super+Space for simple mode or Super+Alt+Space for LLM mode.")
            
        except Exception as e:
            print(f"Error during recording: {e}")
        finally:
            global is_recording
            is_recording = False
    
    # Start the recording thread
    threading.Thread(target=recording_thread).start()


def stop_recording():
    """Signal to stop recording"""
    global is_recording
    if is_recording:
        stop_recording_event.set()


def setup_pop_os_dependencies():
    """Check if required dependencies are installed for Pop!_OS"""
    try:
        # Check for xdotool (for X11) or wtype (for Wayland)
        session_type = os.environ.get('XDG_SESSION_TYPE', 'unknown')
        print(f"Session type: {session_type}")
        
        # Check dependencies without installing them automatically
        missing_deps = []
        
        # Check for ffmpeg
        try:
            subprocess.run(['which', 'ffmpeg'], check=True, stdout=subprocess.PIPE)
            print("✓ ffmpeg is installed")
        except subprocess.CalledProcessError:
            missing_deps.append("ffmpeg")
            print("✗ ffmpeg is missing (needed for audio processing)")
        
        if session_type == 'wayland':
            # For Wayland, check for wtype
            try:
                subprocess.run(['which', 'wtype'], check=True, stdout=subprocess.PIPE)
                print("✓ wtype is installed")
            except subprocess.CalledProcessError:
                missing_deps.append("wtype")
                print("✗ wtype is missing (needed for Wayland)")
        else:
            # For X11, check for xdotool
            try:
                subprocess.run(['which', 'xdotool'], check=True, stdout=subprocess.PIPE)
                print("✓ xdotool is installed")
            except subprocess.CalledProcessError:
                missing_deps.append("xdotool")
                print("✗ xdotool is missing (needed for X11)")
        
        # Check for PortAudio
        try:
            # A simple test to check if portaudio is available
            p = pyaudio.PyAudio()
            p.terminate()
            print("✓ PortAudio is installed")
        except:
            missing_deps.append("portaudio19-dev")
            print("✗ PortAudio is missing")
        
        # Check for requests library
        try:
            import requests
            print("✓ Requests library is installed")
        except ImportError:
            print("✗ Requests library is missing (needed for LLM API)")
            print("  Install with: pip install requests")
        
        # Check for ollama command
        try:
            subprocess.run(['which', 'ollama'], check=True, stdout=subprocess.PIPE)
            print("✓ Ollama is installed")
            # Check if ollama server is running
            ollama_running, detected_model, available_models = check_ollama_status()
            if ollama_running and detected_model:
                # Update the global model variable to use the detected model
                global LOCAL_LLM_MODEL
                LOCAL_LLM_MODEL = detected_model
        except subprocess.CalledProcessError:
            print("✗ Ollama is not installed (recommended for LLM enhancement)")
            print("  Install with: curl -fsSL https://ollama.com/install.sh | sh")
            print("  Then run: ollama pull llama3")
        
        if missing_deps:
            print("\nMissing dependencies detected. Please install:")
            print(f"sudo apt install -y {' '.join(missing_deps)}")
            return False
            
        return True
    except Exception as e:
        print(f"Error checking dependencies: {e}")
        return False


def on_press(key):
    """Handle key press events"""
    try:
        # Add the key to currently pressed keys
        if hasattr(key, 'char'):
            currently_pressed_keys.add(key.char)
        else:
            currently_pressed_keys.add(key)
        
        # Check for Super+Space combination for simple mode
        if (keyboard.Key.space in currently_pressed_keys and 
            is_super_pressed() and 
            keyboard.Key.alt not in currently_pressed_keys):
            start_recording(with_llm=False)
        
        # Check for Super+Alt+Space combination for LLM-enhanced mode
        elif (keyboard.Key.space in currently_pressed_keys and 
              is_super_pressed() and 
              keyboard.Key.alt in currently_pressed_keys):
            start_recording(with_llm=True)
            
    except Exception as e:
        print(f"Error in on_press: {e}")


def on_release(key):
    """Handle key release events"""
    try:
        # Remove the key from currently pressed keys
        if hasattr(key, 'char') and key.char in currently_pressed_keys:
            currently_pressed_keys.remove(key.char)
        elif key in currently_pressed_keys:
            currently_pressed_keys.remove(key)
        
        # If Super, Alt, or Space is released, stop recording
        if key in [keyboard.Key.space, keyboard.Key.cmd, keyboard.Key.cmd_l, 
                  keyboard.Key.cmd_r, keyboard.Key.alt]:
            if is_recording:
                stop_recording()
        
        # Exit on Escape key
        if key == keyboard.Key.esc:
            global is_running
            is_running = False
            stop_recording()
            return False
    except Exception as e:
        print(f"Error in on_release: {e}")


def is_super_pressed():
    """Check if the Super key (Windows/Command key) is pressed"""
    return (keyboard.Key.cmd in currently_pressed_keys or 
            keyboard.Key.cmd_l in currently_pressed_keys or 
            keyboard.Key.cmd_r in currently_pressed_keys)


def main():
    """Main function to set up and run the application"""
    # Detect GPU capabilities and recommend appropriate Whisper model
    whisper_recommendation, gpu_desc, vram = detect_gpu_capabilities()
    print(f"System detected: {gpu_desc}")
    print(f"Recommended Whisper model: {whisper_recommendation}")
    
    # Let the user confirm or change the model
    model_size = whisper_recommendation
    if vram >= 24:  # For high-end GPUs offer large
        print("\nYou have high-end GPU(s) with sufficient VRAM for the 'large' Whisper model.")
        print("This will provide the best possible transcription quality.")
    
    # Load the appropriate Whisper model
    print(f"\nLoading Whisper model: {model_size}")
    global model
    model = whisper.load_model(model_size, device=device)
    
    # Check dependencies
    if not setup_pop_os_dependencies():
        print("Failed to check dependencies. Some features may not work.")
    
    print("\n=== Voice-to-Text with Dual Mode and Cascading LLM Fallback ===")
    print("\nLLM Fallback Chain (for enhanced mode):")
    print(f"1. Local Ollama LLM (primary, using {LOCAL_LLM_MODEL})")
    if OPENAI_API_KEY:
        print("2. OpenAI API (fallback if local fails)")
    else:
        print("2. OpenAI API (not configured - set OPENAI_API_KEY to enable)")
    if ANTHROPIC_API_KEY:
        print("3. Anthropic API (fallback if OpenAI fails)")
    else:
        print("3. Anthropic API (not configured - set ANTHROPIC_API_KEY to enable)")
    print("4. Algorithmic cleanup (final fallback)")
    
    print("\nHotkeys:")
    print("⏺️  Super+Space: Simple transcription with basic cleanup")
    print("✨ Super+Alt+Space: Enhanced transcription with LLM refinement")
    print("❌ Esc: Exit the program")
    
    print("\nReady to use! Press the hotkeys to start recording...")
    
    # Set up the keyboard listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
    
    # Clean up
    p.terminate()
    print("Program terminated.")


if __name__ == "__main__":
    main()