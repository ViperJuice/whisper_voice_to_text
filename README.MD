# Dual-Mode Voice-to-Text for Pop!_OS

A powerful voice-to-text tool with dual operating modes and intelligent LLM refinement.

## Features

- **Two Operation Modes:**
  - **Simple Mode (Super+Space):** Fast transcription with basic cleanup
  - **Enhanced Mode (Super+Alt+Space):** Polished transcription with LLM refinement
  
- **Intelligent LLM Cascading Fallback:**
  - Local Ollama LLM (primary, private, and fast)
  - OpenAI API (first fallback)
  - Anthropic API (second fallback)
  - Algorithmic cleanup (final fallback)

- **Smart System Detection:**
  - Automatic GPU capability detection
  - Optimal Whisper model recommendation
  - Automatic Ollama model selection

- **User-Friendly Interface:**
  - Push-to-talk operation (hold keys while speaking)
  - Direct text insertion at cursor position
  - Clear terminal feedback

## Requirements

- Pop!_OS Linux (may work on other distributions with modification)
- Python 3.8+ with pip
- CUDA-compatible GPU recommended (but works on CPU)
- Microphone

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
source .venv/bin/activate

# OR using uv (faster)
pip install uv
uv venv
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

```bash
sudo apt install -y ffmpeg portaudio19-dev
```

For enhanced mode with local LLM:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the recommended model
ollama pull llama3
```

### 5. Configure API Keys (Optional)

For fallback options, set environment variables:

```bash
# Add to your .bashrc or .zshrc
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Usage

### Starting the Application

```bash
# Simple start
python voice_to_text.py

# OR use the launcher script to filter ALSA warnings
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

3. **Exit:**
   - Press **Esc** key

## Configuration

You can modify the following variables in `voice_to_text.py`:

- `LOCAL_LLM_MODEL`: Change the default Ollama model (default: "llama3")
- `CHUNK`, `FORMAT`, `CHANNELS`, `RATE`: Adjust audio recording parameters
- Customize the cleanup patterns in `algorithmic_cleanup()`

## Troubleshooting

### Audio Issues

- Check microphone connection and permissions
- Ensure you have the audio group permissions: `sudo usermod -a -G audio $USER`
- Try adjusting the `RATE` and `CHUNK` variables

### LLM Connectivity

- For local LLM: Ensure Ollama is running with `ollama serve`
- For API fallbacks: Verify API keys are correctly set
- Check internet connectivity for API fallbacks

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
   - Enhanced Mode: Multi-tier LLM refinement cascade

4. **Insertion:**
   - Refined text is typed at the current cursor position

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition model
- [Ollama](https://ollama.com/) for the local LLM capabilities