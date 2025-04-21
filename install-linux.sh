#!/usr/bin/env bash

# Voice to Text with LLM Enhancement - Linux Setup
echo "Voice to Text with LLM Enhancement - Linux Setup"
echo "==============================================="

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo: sudo ./install-linux.sh"
    exit 1
fi

# Store the original user who ran sudo
ORIG_USER=$SUDO_USER
if [ -z "$ORIG_USER" ]; then
    echo "Error: Could not determine the original user"
    exit 1
fi

# Function to check if a Python package is installed in the virtual environment
check_package() {
    local package=$1
    sudo -u $ORIG_USER bash -c "source .venv/bin/activate && python3 -c 'import $package' 2>/dev/null"
    return $?
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Installing Python..."
    if command -v apt-get &> /dev/null; then
        apt-get update
        apt-get install -y python3 python3-pip
    elif command -v dnf &> /dev/null; then
        dnf install -y python3 python3-pip
    elif command -v pacman &> /dev/null; then
        pacman -S --noconfirm python python-pip
    else
        echo "Could not install Python. Please install Python 3.8-3.11 manually."
        exit 1
    fi
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

# Function to check if system package is installed
check_system_package() {
    local package=$1
    if command -v apt-get &> /dev/null; then
        dpkg -l | grep -q "^ii.*$package"
    elif command -v dnf &> /dev/null; then
        dnf list installed "$package" &> /dev/null
    elif command -v pacman &> /dev/null; then
        pacman -Qi "$package" &> /dev/null
    else
        return 1
    fi
    return $?
}

# Install system dependencies based on distribution
echo "Checking system dependencies..."

MISSING_PACKAGES=()

if command -v apt-get &> /dev/null; then
    # Debian/Ubuntu
    PACKAGES=(
        "ffmpeg"
        "portaudio19-dev"
        "python3-pyaudio"
        "python3-venv"
        "python3-dev"
        "build-essential"
        "libasound2-dev"
        "pkg-config"
        "xclip"
        "wl-clipboard"
        "git"
        "curl"
    )
    
    for pkg in "${PACKAGES[@]}"; do
        if ! check_system_package "$pkg"; then
            MISSING_PACKAGES+=("$pkg")
        fi
    done
    
    if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
        echo "Installing missing packages: ${MISSING_PACKAGES[*]}"
        apt-get update
        apt-get install -y "${MISSING_PACKAGES[@]}"
    else
        echo "All system packages are already installed."
    fi

elif command -v dnf &> /dev/null; then
    # Fedora/RHEL
    PACKAGES=(
        "ffmpeg"
        "portaudio-devel"
        "python3-pyaudio"
        "python3-virtualenv"
        "python3-devel"
        "alsa-lib-devel"
        "pkg-config"
        "xclip"
        "wl-clipboard"
        "git"
        "curl"
    )
    
    for pkg in "${PACKAGES[@]}"; do
        if ! check_system_package "$pkg"; then
            MISSING_PACKAGES+=("$pkg")
        fi
    done
    
    if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
        echo "Installing missing packages: ${MISSING_PACKAGES[*]}"
        dnf install -y "${MISSING_PACKAGES[@]}"
    else
        echo "All system packages are already installed."
    fi

elif command -v pacman &> /dev/null; then
    # Arch Linux
    PACKAGES=(
        "ffmpeg"
        "portaudio"
        "python-pyaudio"
        "python-virtualenv"
        "python-pip"
        "base-devel"
        "alsa-lib"
        "pkg-config"
        "xclip"
        "wl-clipboard"
        "git"
        "curl"
    )
    
    for pkg in "${PACKAGES[@]}"; do
        if ! check_system_package "$pkg"; then
            MISSING_PACKAGES+=("$pkg")
        fi
    done
    
    if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
        echo "Installing missing packages: ${MISSING_PACKAGES[*]}"
        pacman -S --noconfirm "${MISSING_PACKAGES[@]}"
    else
        echo "All system packages are already installed."
    fi

else
    echo "WARNING: Unsupported distribution. Please install the following packages manually:"
    echo "- FFmpeg"
    echo "- PortAudio development package (portaudio19-dev or portaudio-devel)"
    echo "- Python development package (python3-dev or python3-devel)"
    echo "- Build tools (build-essential)"
    echo "- ALSA development package (libasound2-dev or alsa-lib-devel)"
    echo "- xclip"
    echo "- wl-clipboard"
    read -p "Continue anyway? (y/n): " CONTINUE
    if [[ ! $CONTINUE =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install uv if not present (as original user)
if ! command -v uv &> /dev/null; then
    echo "Installing uv package installer..."
    sudo -u $ORIG_USER curl -LsSf https://astral.sh/uv/install.sh | sudo -u $ORIG_USER sh
fi

# Check if virtual environment exists and is valid
VENV_NEEDS_SETUP=false
if [ ! -d ".venv" ]; then
    VENV_NEEDS_SETUP=true
elif ! sudo -u $ORIG_USER bash -c "source .venv/bin/activate 2>/dev/null && python3 --version" &>/dev/null; then
    echo "Virtual environment appears to be corrupted, recreating..."
    rm -rf .venv
    VENV_NEEDS_SETUP=true
fi

# Create virtual environment if needed (as original user)
if [ "$VENV_NEEDS_SETUP" = true ]; then
    echo "Creating virtual environment..."
    # Use --system-site-packages so that distribution provided Python modules (e.g. python3-pyaudio)
    # remain visible inside the virtual environment. This is important because wheels for some
    # audio libraries (like PyAudio) are often unavailable for the newest Python versions and the
    # distro‑packaged build is the most reliable way to obtain them.
    sudo -u $ORIG_USER python3 -m venv --system-site-packages .venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment."
        echo "You may need to install the python3-venv package."
        exit 1
    fi
fi

# Set ownership of the virtual environment to the original user
chown -R $ORIG_USER:$ORIG_USER .venv

# Function to run pip (or uv pip) commands as the original user
run_pip_as_user() {
    local pip_args="$1"
    sudo -u $ORIG_USER env PATH="/home/$ORIG_USER/.local/bin:$PATH" bash -c "source .venv/bin/activate && if command -v uv >/dev/null; then uv pip ${pip_args}; else python -m pip ${pip_args}; fi"
}

# Install/upgrade dependencies only if needed
echo "Checking Python dependencies..."

# Always upgrade pip to latest version
run_pip_as_user "install --upgrade pip"

# Check and install required packages
REQUIRED_PACKAGES=("pyaudio" "rich" "openai-whisper" "pyperclip")
MISSING_PY_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! check_package "${package//-/_}"; then
        MISSING_PY_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PY_PACKAGES[@]} -ne 0 ] || [ "$VENV_NEEDS_SETUP" = true ]; then
    echo "Installing/upgrading Python packages..."
    run_pip_as_user "install -r requirements.txt"
    
    # Install specific packages that might need special handling
    if [[ " ${MISSING_PY_PACKAGES[@]} " =~ " pyaudio " ]]; then
        run_pip_as_user "install pyaudio"
    fi
    if [[ " ${MISSING_PY_PACKAGES[@]} " =~ " openai-whisper " ]]; then
        run_pip_as_user "install git+https://github.com/openai/whisper.git"
    fi
    if [[ " ${MISSING_PY_PACKAGES[@]} " =~ " rich " ]]; then
        run_pip_as_user "install rich"
    fi
    if [[ " ${MISSING_PY_PACKAGES[@]} " =~ " pyperclip " ]]; then
        run_pip_as_user "install pyperclip"
    fi
else
    echo "All Python packages are already installed."
fi

# -----------------------------------------------------------------------------
# Optional: Ensure Ollama is installed and has at least one model
# -----------------------------------------------------------------------------

# Function to test if system is capable of running Ollama using Python helper
check_capable_for_ollama() {
    python3 - <<'PY'
import sys, importlib
try:
    from voicetotext.utils import system_checks as sc
    capable, _ = sc.check_system_for_ollama()
    sys.exit(0 if capable else 1)
except Exception:
    # If module missing, be conservative and allow install
    sys.exit(0)
PY
}

# Install Ollama binary if missing and system is capable
if ! command -v ollama &> /dev/null; then
    if check_capable_for_ollama; then
        echo "Ollama binary not found. Installing Ollama ..."
        curl -fsSL https://ollama.ai/install.sh | sh
        # Ensure ollama is on PATH for current shell session
        export PATH=$PATH:/usr/local/bin
    else
        echo "System may not meet requirements for Ollama. Skipping Ollama installation."
    fi
fi

# Start Ollama server if the binary exists but server not running
if command -v ollama &> /dev/null; then
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "Starting Ollama server in background ..."
        nohup ollama serve >/dev/null 2>&1 &
        sleep 3
    fi
fi

# Ensure at least one model installed (previous logic)
# -----------------------------------------------------------------------------
# Optional: Ensure Ollama has at least one model installed
# -----------------------------------------------------------------------------
if command -v ollama &> /dev/null; then
    echo "Checking Ollama models..."
    # Try quick query (2s timeout) to Ollama tags API
    OLLAMA_RESPONSE=$(curl -s --max-time 2 http://localhost:11434/api/tags || true)
    if [[ -z "$OLLAMA_RESPONSE" ]]; then
        echo "Ollama server not responding (port 11434). Skipping automatic model pull."
    else
        # Detect if models array is empty
        if echo "$OLLAMA_RESPONSE" | grep -q '"models"\s*:\s*\[\s*\]'; then
            DEFAULT_MODEL="llama3"
            echo "No Ollama models detected. Pulling default model: $DEFAULT_MODEL ..."
            sudo -u $ORIG_USER env PATH="/home/$ORIG_USER/.local/bin:$PATH" bash -c "ollama pull $DEFAULT_MODEL"
        else
            echo "Ollama already has models installed."
        fi
    fi
else
    echo "Ollama binary not found. Skipping Ollama model setup."
fi

# Make run script executable
chmod +x "$PROJECT_DIR/run-linux.sh"

echo
echo "Setup completed successfully!"
echo
echo "To run the application:"
echo "- Run ./run-linux.sh"
echo "- Or run 'python main.py' in the terminal after activating the environment"
echo
echo "Enjoy using Voice to Text with LLM Enhancement!"

# -----------------------------------------------------------------------------
# Create desktop shortcut (.desktop file) for the current user so they can
# launch the application directly from their desktop.
# -----------------------------------------------------------------------------

# Resolve the desktop directory for the original user (works with XDG + fallback)
DESKTOP_DIR=$(sudo -u $ORIG_USER bash -c 'xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop"')
mkdir -p "$DESKTOP_DIR"

# Use absolute paths so the icon shows correctly regardless of cwd
PROJECT_DIR="$(realpath .)"

# Copy icon (only if it exists) otherwise fall back to a generic system icon
ICON_FIELD="audio-input-microphone" # default fallback

# Locate icon – allow for repo layouts such as resources/icon.png
if [ -f "$PROJECT_DIR/icon.png" ]; then
    ICON_SOURCE="$PROJECT_DIR/icon.png"
elif [ -f "$PROJECT_DIR/resources/icon.png" ]; then
    ICON_SOURCE="$PROJECT_DIR/resources/icon.png"
else
    ICON_SOURCE=""
fi

if [ -n "$ICON_SOURCE" ]; then
    ICON_DEST_DIR="/home/$ORIG_USER/.local/share/icons"
    sudo -u $ORIG_USER mkdir -p "$ICON_DEST_DIR"
    if sudo -u $ORIG_USER cp -f "$ICON_SOURCE" "$ICON_DEST_DIR/keyless-icon.png"; then
        chown $ORIG_USER:$ORIG_USER "$ICON_DEST_DIR/keyless-icon.png"
        chmod 644 "$ICON_DEST_DIR/keyless-icon.png"
        ICON_FIELD="$ICON_DEST_DIR/keyless-icon.png"
        # Refresh GTK icon cache so the desktop shows new icon
        if command -v gtk-update-icon-cache >/dev/null 2>&1; then
            gtk-update-icon-cache -q "$(dirname "$ICON_DEST_DIR")"
        fi
    else
        echo "Warning: failed to copy icon, continuing without custom icon."
    fi
else
    echo "Warning: icon.png not found – using generic system icon."
fi

DESKTOP_FILE="$DESKTOP_DIR/Keyless.desktop"

cat <<EOF | sudo -u $ORIG_USER tee "$DESKTOP_FILE" >/dev/null
[Desktop Entry]
Type=Application
Version=1.0
Name=Keyless Voice‑to‑Text
Comment=Launch Keyless Voice‑to‑Text
Exec=gnome-terminal -- bash -c 'cd "$PROJECT_DIR" && ./run-linux.sh'
Icon=$ICON_FIELD
Terminal=true
Categories=Utility;
EOF

chmod +x "$DESKTOP_FILE"

# Try to pre‑authorise the launcher in GNOME / Nautilus so the user doesn't have to right‑click → Allow Launching
if command -v gio &>/dev/null; then
    sudo -u $ORIG_USER gio set "$DESKTOP_FILE" "metadata::trusted" yes || true
fi

echo "Created desktop shortcut at $DESKTOP_FILE. If your desktop shows an 'Untrusted Desktop File' warning, right‑click the file and choose 'Allow Launching'." 