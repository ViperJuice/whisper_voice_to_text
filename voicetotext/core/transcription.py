"""
Transcription functionality for the voice to text application.
"""

import os
import whisper
import torch
import numpy as np
from openai import OpenAI

def load_whisper_model(model_size="base"):
    """Load the Whisper model for transcription"""
    print(f"Loading Whisper model '{model_size}' (this may take a moment)...")
    
    # Use system_checks to determine device and model size
    try:
        from voicetotext.utils import system_checks
        sys_config = system_checks.system_check()
        if sys_config and "gpu_available" in sys_config and sys_config["gpu_available"]:
            device = "cuda"
            print("✅ Using GPU for Whisper transcription")
        else:
            device = "cpu"
            print("ℹ️ Using CPU for Whisper transcription")
            
        # Use recommended model size from system_checks if available
        if sys_config and "whisper_model" in sys_config:
            model_size = sys_config["whisper_model"]
            print(f"Using recommended model size: {model_size}")
    except ImportError:
        # Fallback to basic CUDA check if system_checks not available
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = whisper.load_model(model_size, device=device)
    print(f"Model loaded successfully! [TRANSCRIPTION: WHISPER_{model_size.upper()}]")
    return model, device

def transcribe_with_openai_api(audio_file, api_key, language=None, temperature=0.0, prompt=None):
    """Transcribe audio using OpenAI's API with whisper-1 model
    
    Args:
        audio_file (str): Path to the audio file to transcribe
        api_key (str): OpenAI API key
        language (str, optional): Language code of the audio (e.g., 'en', 'es', 'fr')
        temperature (float, optional): Controls randomness in the output. Range: 0.0 to 1.0
        prompt (str, optional): Optional text to guide the model's style or continue a previous audio segment
    
    Returns:
        str: Transcribed text or empty string if transcription fails
    """
    try:
        client = OpenAI(api_key=api_key)
        
        with open(audio_file, "rb") as audio_file:
            # Build the parameters dictionary
            params = {
                "model": "gpt-4o-mini-transcribe",
                "file": audio_file,
                "response_format": "text",
                "temperature": temperature
            }
            
            # Add optional parameters if provided
            if language:
                params["language"] = language
            if prompt:
                params["prompt"] = prompt
            
            # Make the API call
            response = client.audio.transcriptions.create(**params)
            
        return response.strip()
    except Exception as e:
        print(f"❌ Error during OpenAI API transcription: {e}")
        import traceback
        traceback.print_exc()
        return ""

def transcribe_audio(model, audio_data):
    """Transcribe audio data using the Whisper model"""
    try:
        # Validate audio data
        if audio_data is None:
            print("⚠️ Error: No audio data provided for transcription")
            return ""
            
        if len(audio_data) == 0:
            print("⚠️ Error: Empty audio data provided for transcription")
            return ""
            
        # Check if the audio is too short (less than 0.5 seconds at 16kHz)
        if len(audio_data) < 8000:  # 16000 * 0.5 = 8000 samples (0.5 sec)
            print(f"⚠️ Warning: Audio is very short ({len(audio_data)} samples, less than 0.5 seconds)")
            # Still try to transcribe, but warn the user
        
        # Check if audio is too quiet (mostly silence)
        if np.abs(audio_data).max() < 0.01:
            print("⚠️ Warning: Audio appears to be silent or very quiet")
            # Still try to transcribe but warn the user
            
        print(f"🎤 Transcribing with Whisper...")
        
        # Add a timeout to prevent hanging on problematic audio
        try:
            # Use FP16 if we're on GPU
            use_fp16 = model.device.type == 'cuda'
            result = model.transcribe(audio_data, fp16=use_fp16)
            raw_text = result["text"].strip()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("⚠️ CUDA out of memory error. Falling back to CPU...")
                # If we get a CUDA OOM error, try again with CPU
                import torch
                with torch.no_grad():
                    result = model.transcribe(audio_data, fp16=False)
                    raw_text = result["text"].strip()
            else:
                raise  # Re-raise the exception if it's not a CUDA OOM error
        
        # Check if transcription produced text
        if not raw_text:
            print("⚠️ Warning: Whisper did not detect any speech in the audio")
            return ""
            
        return raw_text
    except Exception as e:
        print(f"⚠️ Error during transcription: {e}")
        return "" 