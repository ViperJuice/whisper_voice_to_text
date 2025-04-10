# Dual-Mode Voice-to-Text

A powerful voice-to-text tool with dual operating modes and intelligent LLM refinement, compatible with Windows, macOS, and Linux.

## Features

- **Two Operation Modes:**
  - **Simple Mode (Super+Space):** Fast transcription with basic cleanup
  - **Enhanced Mode (Super+Alt+Space):** Polished transcription with LLM refinement
  
- **Flexible LLM Selection:**
  - **OpenAI API** (default, cloud-based)
  - **Anthropic API** (optional, cloud-based)
  - **Google Flash Light** (optional, cloud-based)
  - **Deepseek Lite** (optional, cloud-based)
  - **Local Ollama LLM** (optional, private)
  - Hotkey switching between available LLM modes
  - Intelligent system assessment for Ollama compatibility

- **Smart System Detection:**
  - Automatic GPU capability detection
  - Optimal Whisper model recommendation
  - Ollama compatibility check
  - Cross-platform compatibility

- **User-Friendly Interface:**
  - Push-to-talk operation (hold keys while speaking)
  - Direct text insertion at cursor position
  - Clear terminal feedback
  - Easy LLM mode switching

## Requirements

- Python 3.8+ with pip
- CUDA-compatible GPU recommended (but works on CPU)
- Microphone
- Operating System:
  - Windows 10/11
  - macOS 10.15+
  - Linux (tested on Pop!_OS, Ubuntu, Fedora)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/voice-to-text.git
cd voice-to-text
```

### 2. Create and Activate Virtual Environment

```bash
# Using venv
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# OR using uv (faster, recommended)
pip install uv
uv venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# OR using uv (faster)
uv pip install -r requirements.txt
```

### 4. Install System Dependencies

#### Windows:
1. Install [PortAudio](https://www.portaudio.com/download.html)
2. Install [FFmpeg](https://ffmpeg.org/download.html) and add it to your PATH

#### macOS:
```bash
brew install portaudio ffmpeg
```

#### Linux:
```bash
sudo apt install -y ffmpeg portaudio19-dev
```

### 5. Configure API Keys

Set environment variables for your preferred LLM services:

#### Windows:
```powershell
# Add to your PowerShell profile
$env:OPENAI_API_KEY="your-openai-api-key"
$env:ANTHROPIC_API_KEY="your-anthropic-api-key"  # Optional
$env:GOOGLE_API_KEY="your-google-api-key"        # Optional
$env:DEEPSEEK_API_KEY="your-deepseek-api-key"    # Optional
```

#### macOS/Linux:
```bash
# Add to your .bashrc or .zshrc
export OPENAI_API_KEY="your-api-key-here"
export ANTHROPIC_API_KEY="your-api-key-here"  # Optional
export GOOGLE_API_KEY="your-api-key-here"     # Optional
export DEEPSEEK_API_KEY="your-api-key-here"   # Optional
```

To obtain API keys:
- **OpenAI**: Visit [OpenAI Platform](https://platform.openai.com/api-keys)
- **Anthropic**: Visit [Anthropic Console](https://console.anthropic.com/account/keys)
- **Google**: Visit [Google MakerSuite](https://makersuite.google.com/app/apikey)
- **Deepseek**: Visit [Deepseek Platform](https://platform.deepseek.com/)

Note: You only need to configure the API keys for the services you plan to use. OpenAI is the default and recommended option.

### 6. Optional: Install Ollama (for Local LLM)

The application will automatically check if your system is capable of running Ollama and provide installation instructions if appropriate.

#### Windows:
1. Download and install from [Ollama website](https://ollama.com/download)
2. Add Ollama to your PATH

#### macOS:
```bash
brew install ollama
```

#### Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

After installation, pull the recommended model:
```bash
ollama pull llama3
```

## Usage

### Starting the Application

#### Windows:
```powershell
python voice_to_text.py
```

#### macOS/Linux:
```bash
# Simple start
python voice_to_text.py

# OR use the launcher script to filter ALSA warnings (Linux only)
./run_voice_to_text.sh
```

### Using Voice-to-Text

1. **Simple Mode:**
   - Press and hold **Super+Space** (Windows/Command key + Space)
   - Speak into your microphone
   - Release the keys to process and insert text

2. **Enhanced Mode:**
   - Press and hold **Super+Alt+Space**
   - Speak into your microphone
   - Release the keys to process with LLM refinement

3. **LLM Mode Switching:**
   - **Super+Alt+Space+O**: Switch to OpenAI mode (default)
   - **Super+Alt+Space+C**: Switch to Anthropic mode (if configured)
   - **Super+Alt+Space+G**: Switch to Google Flash Light mode (if configured)
   - **Super+Alt+Space+D**: Switch to Deepseek Lite mode (if configured)
   - **Super+Alt+Space+L**: Switch to Local LLM mode (if Ollama is running)

4. **Exit:**
   - Press **Esc** key

## Configuration

You can modify the following variables in `voice_to_text.py`:

- `CHUNK`, `FORMAT`, `CHANNELS`, `RATE`: Adjust audio recording parameters
- Customize the cleanup patterns in `algorithmic_cleanup()`

## Troubleshooting

### Audio Issues

- Check microphone connection and permissions
- Ensure proper audio drivers are installed
- Try adjusting the `RATE` and `CHUNK` variables

### LLM Connectivity

- For OpenAI: Verify API key is correctly set
- For Anthropic: Verify API key is correctly set (if using)
- For local LLM: Ensure Ollama is running (if using)

### Performance Issues

- Try a smaller Whisper model if transcription is slow
- Ensure your GPU drivers are up to date
- Close other GPU-intensive applications

## How It Works

1. **Recording:**
   - Audio is captured when hotkeys are pressed
   - Raw audio is saved as a temporary WAV file

2. **Transcription:**
   - OpenAI's Whisper model processes the audio
   - Raw text is extracted from the transcription

3. **Refinement:**
   - Simple Mode: Basic algorithmic cleanup
   - Enhanced Mode: LLM refinement using selected provider

4. **Insertion:**
   - Refined text is typed at the current cursor position

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition model
- [Ollama](https://ollama.com/) for the local LLM capabilities