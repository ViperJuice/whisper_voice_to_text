#!/usr/bin/env bash

# Voice to Text with LLM Enhancement - Launcher for Linux
# This script launches the voice-to-text application

echo "Voice to Text with LLM Enhancement - Launcher"
echo "============================================="
echo "(export VTT_DEBUG=1 for verbose logging)"

# Function to run install script
run_install() {
    echo "Running installation script..."
    chmod +x install-linux.sh
    if [ "$EUID" -ne 0 ]; then
        echo "Installation requires sudo privileges."
        sudo ./install-linux.sh
    else
        ./install-linux.sh
    fi
    if [ $? -ne 0 ]; then
        echo "Installation failed. Please check the error messages above."
        exit 1
    fi
}

# Function to check if a Python package is installed in the virtual environment
check_package() {
    local package=$1
    if [ -d ".venv" ]; then
        source .venv/bin/activate 2>/dev/null
        # Use importlib to support dotted module paths (e.g. google.generativeai)
        python3 - <<PY 2>/dev/null
import importlib, sys
try:
    importlib.import_module("${package}")
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
        return $?
    fi
    return 1
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Running installation..."
    run_install
fi

# Check if we're in the right directory
if [ ! -f "main.py" ] || [ ! -f "install-linux.sh" ]; then
    echo "ERROR: Required files not found. Please run this script from the project root directory."
    exit 1
fi

# Check for dependencies and run install script if needed
NEED_INSTALL=0

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv package installer not found."
    NEED_INSTALL=1
fi

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg not found."
    NEED_INSTALL=1
fi

# Check if virtual environment exists and is properly set up
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found."
    NEED_INSTALL=1
else
    # Check for required Python packages
    # Important runtime‑level imports required by the application. If any of these fail we
    # trigger a full installation/update cycle via install‑linux.sh.
    #
    # Notes:
    #   • Use the *import* module names here, not the PyPI package names. For example
    #     the PyPI package "openai‑whisper" installs the module "whisper".
    #   • Hyphens in package names are therefore NOT replaced – the full dotted import
    #     path is supplied so that `python -c "import <name>"` works correctly.
    #   • Keep this list in sync with requirements.txt and with any new integrations
    #     added to the codebase.
    REQUIRED_PACKAGES=(
        "pyaudio"              # microphone access
        "whisper"              # transcription (local + OpenAI Whisper API)
        "openai"               # OpenAI LLM / Whisper API client
        "anthropic"            # Claude (Anthropic) client
        "google.generativeai" # Gemini client
        "ollama"               # Local LLM client shim (optional but cheap to check)
        "requests"             # HTTP for several LLM helpers
        "rich"                 # colourful terminal output (used in helpers)
        "pyperclip"            # clipboard support
    )
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if ! check_package "$package"; then
            echo "Python package '${package}' not found."
            NEED_INSTALL=1
            break
        fi
    done
fi

# Warn if clipboard backend missing
if ! command -v xclip &> /dev/null && ! command -v wl-copy &> /dev/null; then
    echo "WARNING: No clipboard backend (xclip or wl-clipboard) detected. Copy-to-clipboard will not work."
    echo "Run sudo apt-get install xclip   or   sudo apt-get install wl-clipboard (Wayland) to enable it."
fi

# Run install script if any dependency is missing
if [ $NEED_INSTALL -eq 1 ]; then
    echo "Some dependencies are missing. Running installation..."
    run_install
fi

# Check if Ollama is installed and running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "Ollama is not running or not installed."
    echo "To use local LLM features, please install and start Ollama:"
    echo "1. Install: curl -fsSL https://ollama.ai/install.sh | sh"
    echo "2. Start: ollama serve"
    echo
    echo "The application will continue in API-only mode."
    echo
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Set PYTHONPATH to include the current directory
export PYTHONPATH=$PWD:$PYTHONPATH

# Ensure dependencies are up to date
echo "Checking for dependency updates..."
uv pip install -r requirements.txt > /dev/null 2>&1

# Start the application
echo "Starting Voice to Text application..."
echo
echo "HOTKEYS:"
echo "- Simple Mode: Alt"
echo "- Enhanced Mode: Alt+Windows"
echo "- Toggle Transcription: Alt+T"
echo "- Exit: Escape"
echo "- Cleanup Temp Files: Alt+K"
echo
echo "Note: Toggle Transcription will switch between local Whisper and OpenAI API"
echo "if both services are available. Check the system check output above"
echo "for available transcription services."
echo
echo "Note: The Windows key is also known as the Super key on Linux."
echo "Press Ctrl+C to quit..."
echo

python main.py
EXIT_CODE=$?

# If the application exited with an error
if [ $EXIT_CODE -ne 0 ]; then
    echo
    echo "Voice to Text application exited with an error (code: $EXIT_CODE)."
    echo "Running installation script to ensure all dependencies are properly set up..."
    run_install
    echo "Please try running the application again."
else
    echo
    echo "Voice to Text application has closed."
fi 