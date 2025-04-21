#!/usr/bin/env python3
"""
System checks for voice-to-text application
This module handles:
1. OS detection and hotkey mapping
2. System capability assessment (CPU, RAM, GPU)
3. Ollama availability checks
4. GPU detection for Whisper model selection
"""
import os
import platform
import re
import sys
import subprocess
import torch
import logging

# ---------------------------------------------------------------------------
# Module‑level logger; legacy print calls are routed to logger.info so they
# still appear in normal runs but can be silenced by adjusting log level.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _info_print(*args, **kwargs):  # type: ignore[var-annotated]
    """Forward built‑in print output to logger.info."""
    logger.info(" ".join(str(a) for a in args))


# Shadow print within this module only.
print = _info_print  # type: ignore[assignment]

try:
    import psutil
except ImportError:
    psutil = None
try:
    import requests
except ImportError:
    requests = None

# Platform-specific key combinations
IS_WINDOWS = platform.system() == 'Windows'
IS_MACOS = platform.system() == 'Darwin'
IS_LINUX = platform.system() == 'Linux'

# Global hotkeys dictionary based on OS
def get_os_hotkeys():
    """Returns appropriate hotkey mapping based on operating system"""
    if IS_WINDOWS:
        return {
            'simple_mode': 'alt+win',
            'enhanced_mode': 'alt+win+ctrl',
            'toggle_mode': 'alt+win+o',
            'openai_enhanced': 'alt+win+ctrl+o',
            'gemini_enhanced': 'alt+win+ctrl+g',
            'deepseek_enhanced': 'alt+win+ctrl+d',
            'claude_enhanced': 'alt+win+ctrl+c',
            'stop': 'alt+win+ctrl+s',
            'exit': 'escape',
            'cleanup': 'alt+win+ctrl+x'
        }
    elif IS_MACOS:
        return {
            'simple_mode': 'option+command',
            'enhanced_mode': 'option+command+control',
            'toggle_mode': 'option+command+o',
            'openai_enhanced': 'option+command+control+o',
            'gemini_enhanced': 'option+command+control+g',
            'deepseek_enhanced': 'option+command+control+d',
            'claude_enhanced': 'option+command+control+c',
            'stop': 'option+command+control+s',
            'exit': 'escape',
            'cleanup': 'option+command+control+x'
        }
    else:  # Linux and others
        return {
            'simple_mode': 'alt+super',
            'enhanced_mode': 'alt+super+ctrl',
            'toggle_mode': 'alt+super+o',
            'openai_enhanced': 'alt+super+ctrl+o',
            'gemini_enhanced': 'alt+super+ctrl+g',
            'deepseek_enhanced': 'alt+super+ctrl+d',
            'claude_enhanced': 'alt+super+ctrl+c',
            'stop': 'alt+super+ctrl+s',
            'exit': 'escape',
            'cleanup': 'alt+super+ctrl+x'
        }

def check_ffmpeg():
    """Check if ffmpeg is installed and working"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✓ FFmpeg is installed and available")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("✗ FFmpeg is missing (needed for audio processing)")
        print("  Install with: winget install ffmpeg")
        if IS_WINDOWS:
            print("  If ffmpeg is installed but not detected, try:")
            print("  1. Close and reopen your terminal")
            print("  2. Restart your computer")
            print("  3. Verify ffmpeg is in your PATH: echo %PATH%")
        return False

def detect_gpu_capabilities():
    """Detect GPU capabilities and recommend appropriate Whisper model"""
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

def check_system_for_ollama():
    """Check if the system has sufficient resources for Ollama"""
    try:
        if not psutil:
            print("psutil not installed, skipping detailed system checks")
            return False, {}
            
        # Check GPU capabilities
        _, _, vram = detect_gpu_capabilities()
        
        # Check CPU cores
        cpu_cores = os.cpu_count()
        
        # Check available RAM
        total_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB
        
        # Minimum requirements for Ollama
        min_requirements = {
            'vram': 4,  # GB
            'cpu_cores': 4,
            'ram': 8,  # GB
        }
        
        # Check if system meets minimum requirements
        meets_requirements = (
            vram >= min_requirements['vram'] or
            (cpu_cores >= min_requirements['cpu_cores'] and total_ram >= min_requirements['ram'])
        )
        
        system_info = {
            'vram': vram,
            'cpu_cores': cpu_cores,
            'ram': total_ram
        }
        
        if meets_requirements:
            print("✓ System meets requirements for running Ollama")
        else:
            print("✗ System may not have sufficient resources for Ollama")
            print(f"  - CPU Cores: {cpu_cores} (minimum: {min_requirements['cpu_cores']})")
            print(f"  - RAM: {total_ram:.1f}GB (minimum: {min_requirements['ram']}GB)")
            print(f"  - VRAM: {vram}GB (minimum: {min_requirements['vram']}GB)")
        
        return meets_requirements, system_info
    except Exception as e:
        print(f"Error checking system capabilities: {e}")
        return False, {}

def check_ollama_status(local_llm_model="llama3"):
    """Check if Ollama server is running and find available models"""
    if not requests:
        print("requests module not installed, cannot check Ollama status")
        return False, None, []
        
    try:
        # Check if Ollama server is responding
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        
        if response.status_code == 200:
            # Get available models
            models_data = response.json()
            available_models = [model["name"] for model in models_data.get("models", [])]
            
            if local_llm_model in available_models:
                print(f"✓ Ollama server is running with {local_llm_model} model")
                return True, local_llm_model, available_models
            else:
                print(f"✗ Ollama model {local_llm_model} not found")
                if available_models:
                    print("Available models:", ", ".join(available_models))
                    print(f"Using first available model: {available_models[0]}")
                    return True, available_models[0], available_models
                else:
                    print("No models available in Ollama")
                    return False, None, []
        else:
            print("✗ Ollama server responded with an error")
            return False, None, []
    except requests.exceptions.ConnectionError:
        print("✗ Ollama server is not running")
        print("  To start Ollama, run: ollama serve")
        return False, None, []
    except Exception as e:
        print(f"✗ Error checking Ollama status: {e}")
        return False, None, []

def get_available_ollama_models():
    """Detect available Ollama models and select the best one"""
    if not requests:
        print("requests module not installed, cannot detect Ollama models")
        return None, []
        
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

def create_env_file():
    """Create .env file with placeholder API keys if it doesn't exist"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    if not os.path.exists(env_path):
        print("\nCreating .env file with placeholder API keys...")
        env_content = """# API Keys for Voice to Text with LLM Enhancement
# Replace the placeholders with your actual API keys

# OpenAI API Key (Required for transcription and enhancement if no GPU/Ollama)
# Get your key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_key_here

# Anthropic Claude API Key (Optional - for alternative enhancement)
# Get your key from: https://console.anthropic.com/settings/keys
ANTHROPIC_API_KEY=your_anthropic_key_here

# Google AI API Key (Optional - for alternative enhancement)
# Get your key from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_google_key_here

# DeepSeek API Key (Optional - for alternative enhancement)
# Get your key from: https://platform.deepseek.com/api-keys
DEEPSEEK_API_KEY=your_deepseek_key_here

# Whisper Model Configuration
# Options: tiny, base, small, medium, large
# Adjust based on your system's capabilities
WHISPER_MODEL=base
"""
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("Created .env file with placeholder API keys.")
        print("Please edit the file and replace the placeholders with your actual API keys.")
        print("You can find instructions for obtaining API keys in the comments above each key.")
        return True
    return False

def check_transcription_services():
    """Check availability of transcription services and return their status"""
    # Check local Whisper capability
    whisper_available = False
    whisper_status = "Local Whisper (CPU)"
    if torch.cuda.is_available():
        _, _, vram = detect_gpu_capabilities()
        if vram >= 4:
            whisper_status = "Local Whisper (GPU)"
            whisper_available = True
        else:
            whisper_status = "Local Whisper (CPU - GPU VRAM insufficient)"
            whisper_available = True
    else:
        whisper_status = "Local Whisper (CPU - No GPU)"
        whisper_available = True
    
    # Check OpenAI API availability
    openai_available = False
    openai_status = "OpenAI API (Not available)"
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key and openai_api_key != 'your_openai_key_here':
        openai_available = True
        openai_status = "OpenAI API (Available)"
    
    return {
        'whisper': {
            'available': whisper_available,
            'status': whisper_status
        },
        'openai': {
            'available': openai_available,
            'status': openai_status
        }
    }

def get_available_transcription_modes():
    """Get list of available transcription modes"""
    services = check_transcription_services()
    available_modes = []
    
    if services['whisper']['available']:
        available_modes.append('local')
    if services['openai']['available']:
        available_modes.append('openai')
    
    return available_modes

def system_check():
    """Run all system checks and return configuration"""
    print("\n=== System Capability Check ===")
    
    # Create .env file if it doesn't exist
    created_env = create_env_file()
    if created_env:
        print("\nPlease restart the application after adding your API keys.")
        return None
    
    # Check OS and determine hotkeys
    hotkeys = get_os_hotkeys()
    os_name = "Windows" if IS_WINDOWS else "macOS" if IS_MACOS else "Linux"
    print(f"Detected OS: {os_name}")
    
    # Check for FFMPEG
    ffmpeg_available = check_ffmpeg()
    if not ffmpeg_available:
        print("FFmpeg is required for audio processing. Please install it and try again.")
    
    # Check transcription services
    services = check_transcription_services()
    print("\nAvailable Transcription Services:")
    print(f"- {services['whisper']['status']}")
    print(f"- {services['openai']['status']}")
    
    # Check GPU capabilities
    whisper_model, gpu_desc, vram = detect_gpu_capabilities()
    print(f"GPU Status: {gpu_desc}")
    print(f"Recommended Whisper model: {whisper_model}")
    
    # Check system for Ollama capability
    can_run_ollama, system_info = check_system_for_ollama()
    
    # Check Ollama status
    ollama_running = False
    selected_model = None
    available_models = []
    
    # Only check Ollama if system meets requirements
    if can_run_ollama:
        ollama_running, selected_model, available_models = check_ollama_status()
    
    # If Ollama not available, check for models
    if not selected_model and can_run_ollama:
        selected_model, available_models = get_available_ollama_models()
        if selected_model:
            print(f"Selected Ollama model: {selected_model}")
    
    # Determine LLM availability
    can_use_llm = ollama_running and selected_model is not None
    
    # Check OpenAI API key availability
    openai_api_key = os.getenv('OPENAI_API_KEY')
    can_use_openai = openai_api_key is not None and openai_api_key != 'your_openai_key_here'
    
    # Determine transcription mode
    if torch.cuda.is_available() and vram >= 4:
        use_local_whisper = True
        print("Using local Whisper model for transcription (GPU)")
    else:
        use_local_whisper = True  # Always use local Whisper, just on CPU
        if torch.cuda.is_available():
            print("GPU VRAM insufficient - using local Whisper on CPU")
        else:
            print("No GPU available - using local Whisper on CPU")
        if can_use_openai:
            print("Note: OpenAI API is available as an alternative")
    
    # Determine enhancement mode
    if can_use_llm:
        print("Using local Ollama for text enhancement")
    elif can_use_openai:
        print("Ollama not available - falling back to OpenAI for text enhancement")
    else:
        print("WARNING: Neither Ollama nor OpenAI available for text enhancement")
        print("Please install Ollama or add your OpenAI API key to .env file")
        print("The application will run in transcription-only mode")
    
    print("=== System Check Complete ===\n")
    
    return {
        "os_name": os_name,
        "hotkeys": hotkeys,
        "whisper_model": whisper_model,
        "gpu_available": torch.cuda.is_available(),
        "vram": vram,
        "use_local_whisper": use_local_whisper,
        "can_run_ollama": can_run_ollama,
        "ollama_running": ollama_running,
        "selected_ollama_model": selected_model,
        "available_ollama_models": available_models,
        "can_use_llm": can_use_llm,
        "can_use_openai": can_use_openai,
        "system_info": system_info,
        "transcription_services": services,
        "available_transcription_modes": get_available_transcription_modes()
    }

if __name__ == "__main__":
    # Run system check when script is executed directly
    config = system_check()
    print("\nSystem Configuration Summary:")
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, list):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value}") 