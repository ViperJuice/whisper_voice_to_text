#!/usr/bin/env bash

# Voice to Text with LLM Enhancement - Launcher for MacOS
# This script launches the voice-to-text application

echo "Voice to Text with LLM Enhancement - Launcher"
echo "============================================="
echo "(export VTT_DEBUG=1 for verbose logging)"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please run ./install-mac.sh first."
    exit 1
fi

# Check if uv is installed
if command -v uv &> /dev/null; then
    echo "uv package installer detected."
    USE_UV=1
else
    echo "uv not found, using standard pip."
    USE_UV=0
fi

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "ERROR: main.py not found. Please run this script from the project root directory."
    exit 1
fi

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "WARNING: FFmpeg not found. This is required for audio processing."
    echo "The application may not work correctly without it."
    
    if command -v brew &> /dev/null; then
        echo "Would you like to install FFmpeg using Homebrew? (y/n): "
        read -p "" INSTALL_FFMPEG
        if [[ $INSTALL_FFMPEG =~ ^[Yy]$ ]]; then
            echo "Installing FFmpeg via Homebrew..."
            brew install ffmpeg
        else
            read -p "Continue anyway? (y/n): " CONTINUE
            if [[ ! $CONTINUE =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    else
        echo "Install using: brew install ffmpeg (requires Homebrew)"
        read -p "Continue anyway? (y/n): " CONTINUE
        if [[ ! $CONTINUE =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Check if Ollama is installed and running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "Ollama is not running or not installed."
    read -p "Would you like to install Ollama? (y/n): " INSTALL_OLLAMA
    if [[ $INSTALL_OLLAMA =~ ^[Yy]$ ]]; then
        echo "Installing Ollama..."
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo "Homebrew not found. Please install Ollama manually:"
            echo "1. Download from https://ollama.ai/download"
            echo "2. Install the package"
            echo "3. Run 'ollama serve' in a terminal"
            read -p "Continue without Ollama? (y/n): " CONTINUE
            if [[ ! $CONTINUE =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
        echo "Starting Ollama service..."
        ollama serve &
        sleep 5
    else
        echo "Ollama is required for local LLM functionality."
        echo "The application will run in API-only mode."
        echo
    fi
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Running installation first..."
    ./install-mac.sh
    if [ $? -ne 0 ]; then
        echo "Installation failed. Please run ./install-mac.sh manually."
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Set PYTHONPATH to include the current directory
export PYTHONPATH=$PWD:$PYTHONPATH

# Ensure dependencies are installed
echo "Checking dependencies..."
if [ $USE_UV -eq 1 ]; then
    uv pip install -r requirements.txt > /dev/null 2>&1
else
    pip install -r requirements.txt > /dev/null 2>&1
fi

# Check for accessibility permissions on MacOS
echo "Note: On MacOS, you may need to grant accessibility permissions for keyboard monitoring."
echo "If the application doesn't respond to hotkeys, go to:"
echo "System Preferences → Security & Privacy → Privacy → Accessibility"
echo "and add Terminal (or your preferred terminal app) to the list."
echo

# Start the application
echo "Starting Voice to Text application..."
echo
echo "HOTKEYS:"
echo "- Simple Mode: Command (⌘) + Option (⌥)"
echo "- Enhanced Mode: Command (⌘) + Option (⌥) + Control (⌃)"
echo "- Toggle Transcription: Command (⌘) + Option (⌥) + T"
echo "- Exit: Escape"
echo "- Cleanup Temp Files: Control (⌃) + K"
echo
echo "Note: Toggle Transcription will switch between local Whisper and OpenAI API"
echo "if both services are available. Check the system check output above"
echo "for available transcription services."
echo
echo "Note: The Command key (⌘) is the primary modifier on macOS."
echo "Press Control+C to quit..."
echo

if [ $USE_UV -eq 1 ]; then
    uv run python main.py
else
    python main.py
fi
EXIT_CODE=$?

# If the application exited with an error
if [ $EXIT_CODE -ne 0 ]; then
    echo
    echo "Voice to Text application exited with an error (code: $EXIT_CODE)."
    echo "If you're experiencing issues, try running: ./install-mac.sh"
    echo "to ensure all dependencies are properly installed."
else
    echo
    echo "Voice to Text application has closed."
fi 