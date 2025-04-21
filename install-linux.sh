#!/usr/bin/env bash

# Voice to Text with LLM Enhancement - Linux Setup
echo "Voice to Text with LLM Enhancement - Linux Setup"
echo "==============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.8-3.11."
    echo "Visit https://www.python.org/downloads/ or use your distribution's package manager."
    exit 1
fi

# Verify Python version is 3.11 or lower
PYTHON_VERSION=$(python3 --version | grep -oP '(?<=Python )\d+\.\d+')
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
    
    # Suggest installation command based on distribution
    if command -v apt-get &> /dev/null; then
        echo "Install using: sudo apt-get install ffmpeg"
    elif command -v dnf &> /dev/null; then
        echo "Install using: sudo dnf install ffmpeg"
    elif command -v pacman &> /dev/null; then
        echo "Install using: sudo pacman -S ffmpeg"
    else
        echo "Please install ffmpeg using your distribution's package manager."
    fi
    
    read -p "Continue anyway? (y/n): " CONTINUE
    if [[ ! $CONTINUE =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for system dependencies
echo "Checking system dependencies..."
PORTAUDIO_INSTALLED=false

if command -v apt-get &> /dev/null; then
    echo "Debian/Ubuntu detected. Checking for required packages..."
    if dpkg -s portaudio19-dev &> /dev/null; then
        echo "PortAudio development package already installed."
        PORTAUDIO_INSTALLED=true
    else
        echo "Installing PortAudio development package..."
        sudo apt-get update
        sudo apt-get install -y portaudio19-dev python3-venv
        PORTAUDIO_INSTALLED=true
    fi
elif command -v dnf &> /dev/null; then
    echo "Fedora/RHEL detected. Checking for required packages..."
    if rpm -q portaudio-devel &> /dev/null; then
        echo "PortAudio development package already installed."
        PORTAUDIO_INSTALLED=true
    else
        echo "Installing PortAudio development package..."
        sudo dnf install -y portaudio-devel python3-virtualenv
        PORTAUDIO_INSTALLED=true
    fi
else
    echo "WARNING: Unsupported distribution. You may need to install the PortAudio development package manually."
    echo "Common package names: portaudio19-dev (Debian/Ubuntu), portaudio-devel (Fedora/RHEL)"
    read -p "Continue anyway? (y/n): " CONTINUE
    if [[ ! $CONTINUE =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment."
        echo "You may need to install the python3-venv package."
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
chmod +x run-linux.sh

echo
echo "Setup completed successfully!"

if [ "$PORTAUDIO_INSTALLED" = false ]; then
    echo
    echo "WARNING: PortAudio development package may not be installed."
    echo "If you encounter issues with audio recording, please install it manually."
fi

echo
echo "To run the application:"
echo "- Run ./run-linux.sh"
echo "- Or run 'python main.py' in the terminal after activating the environment"
echo
echo "Enjoy using Voice to Text with LLM Enhancement!" 