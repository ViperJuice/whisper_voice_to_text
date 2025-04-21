#!/usr/bin/env bash

# Voice to Text with LLM Enhancement - MacOS Setup
echo "Voice to Text with LLM Enhancement - MacOS Setup"
echo "==============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.8-3.11."
    echo "Visit https://www.python.org/downloads/ or use Homebrew (brew install python@3.11)"
    exit 1
fi

# Verify Python version is 3.11 or lower
PYTHON_VERSION=$(python3 --version | grep -oE '3\.[0-9]+')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -gt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -gt 11 ]); then
    echo "WARNING: Python version $PYTHON_VERSION detected."
    echo "This application is designed for Python 3.11 or older."
    echo "Some features may not work correctly."
    echo
    read -p "Continue anyway? (y/n): " CONTINUE
    if [[ ! $CONTINUE =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "WARNING: FFmpeg not found. This is required for audio processing."
    
    if command -v brew &> /dev/null; then
        echo "Would you like to install FFmpeg using Homebrew? (y/n): "
        read -p "" INSTALL_FFMPEG
        if [[ $INSTALL_FFMPEG =~ ^[Yy]$ ]]; then
            echo "Installing FFmpeg via Homebrew..."
            brew install ffmpeg
        else
            echo "Please install FFmpeg manually."
            read -p "Continue anyway? (y/n): " CONTINUE
            if [[ ! $CONTINUE =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    else
        echo "Please install FFmpeg manually."
        read -p "Continue anyway? (y/n): " CONTINUE
        if [[ ! $CONTINUE =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Check for Homebrew
echo "Checking system dependencies..."
PORTAUDIO_INSTALLED=false

if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. It's recommended for installing system dependencies."
    echo "Visit https://brew.sh/ to install Homebrew."
    read -p "Continue without Homebrew? (y/n): " CONTINUE
    if [[ ! $CONTINUE =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    # Install PortAudio via Homebrew if not already installed
    if brew list portaudio &> /dev/null; then
        echo "PortAudio already installed via Homebrew."
        PORTAUDIO_INSTALLED=true
    else
        echo "Installing PortAudio via Homebrew..."
        brew install portaudio
        PORTAUDIO_INSTALLED=true
    fi
fi

# Create virtual environment if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment."
        exit 1
    fi
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Update pip
echo "Updating pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing/updating dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Installing Whisper (this may take a moment)..."
pip install --exists-action i git+https://github.com/openai/whisper.git

echo "Checking for additional dependencies..."
pip install --exists-action i openai anthropic google-generativeai ollama retry rich

# Make run script executable
chmod +x run-mac.sh

echo
echo "Setup completed successfully!"

if [ "$PORTAUDIO_INSTALLED" = false ]; then
    echo
    echo "WARNING: PortAudio may not be installed."
    echo "If you encounter issues with audio recording, please install it manually using Homebrew:"
    echo "brew install portaudio"
fi

echo
echo "To run the application:"
echo "- Run ./run-mac.sh"
echo "- Or run 'python main.py' in the terminal after activating the environment"
echo
echo "Enjoy using Voice to Text with LLM Enhancement!" 