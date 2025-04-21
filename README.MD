# Voice to Text with LLM Enhancement

A powerful voice transcription tool that uses OpenAI's Whisper model for high-quality speech-to-text conversion and various LLM models for enhancing transcriptions.

## Features

- **Real-time transcription** using OpenAI's Whisper model
- **GPU acceleration** for faster transcription (when available)
- **LLM-enhanced transcriptions** with support for multiple models:
  - OpenAI (GPT)
  - Claude (Anthropic)
  - Google (Gemini)
  - DeepSeek
  - Ollama (local models)
- **Convenient hotkeys** for controlling recording and switching between models
- **Cross-platform support** for Windows, macOS, and Linux
- **Automatic cleanup** of temporary files
- **Graceful fallback** to local models when API keys are unavailable

## System Requirements

- Python 3.8-3.11
- FFmpeg (for audio processing)
- PortAudio (for audio recording)
- API keys for LLM services (optional, for enhanced transcription)
- NVIDIA GPU (optional, for acceleration)
- uv (optional, recommended for faster package installation)

## Installation

### Windows

1. Download or clone this repository
2. Choose one of the following installation methods:
   - **Option 1:** Double-click `install-windows.bat`
   - **Option 2:** Right-click `run-windows.ps1` and select "Run with PowerShell"
3. Run the application using one of the following methods:
   - **Option 1:** Double-click `run-windows-simple.bat` (simplest method)
   - **Option 2:** Right-click `run-windows.ps1` and select "Run with PowerShell" (recommended)
   - **Option 3:** From PowerShell, run `.\.venv\Scripts\activate; python main.py`

### macOS

1. Download or clone this repository
2. Open Terminal in the project directory
3. Make the installation script executable: `chmod +x install-mac.sh`
4. Run the installation script: `./install-mac.sh`
5. Run the application: `./run-mac.sh`

### Linux

1. Download or clone this repository
2. Open Terminal in the project directory
3. Make the installation script executable: `chmod +x install-linux.sh`
4. Run the installation script: `sudo ./install-linux.sh`
5. Run the application: `./run-linux.sh`

## Using uv Package Installer (Recommended)

This project supports [uv](https://github.com/astral-sh/uv), a faster Python package installer. The scripts will automatically detect and use uv if it's installed. To install uv:

```bash
# Install uv
pip install uv

# Or on Windows with PowerShell
pip install uv
```

Benefits of using uv:
- Faster dependency installation
- Better caching of packages
- Improved reliability for GPU-accelerated packages

## Usage

Once the application is running, you can use the following hotkeys:

### Windows Hotkeys
- `alt+win` = Simple mode (hold to start recording, release to transcribe with Whisper)
- `alt+win+ctrl` = Enhanced mode (hold to start recording, release to transcribe with Whisper and enhance with LLM)
- `alt+win+o` = Toggle mode (switch between local Whisper and OpenAI API transcription)
- `alt+win+ctrl+o` = OpenAI Enhanced (transcribe with Whisper and enhance with OpenAI)
- `alt+win+ctrl+g` = Gemini Enhanced (transcribe with Whisper and enhance with Google Gemini)
- `alt+win+ctrl+d` = DeepSeek Enhanced (transcribe with Whisper and enhance with DeepSeek)
- `alt+win+ctrl+c` = Claude Enhanced (transcribe with Whisper and enhance with Claude)
- `alt+win+ctrl+s` = Stop recording
- `esc` = Exit application
- `alt+win+ctrl+x` = Cleanup temporary files

### macOS Hotkeys
- `option+command` = Simple mode
- `option+command+control` = Enhanced mode
- `option+command+o` = Toggle mode
- `option+command+control+o` = OpenAI Enhanced
- `option+command+control+g` = Gemini Enhanced
- `option+command+control+d` = DeepSeek Enhanced
- `option+command+control+c` = Claude Enhanced
- `option+command+control+s` = Stop recording
- `esc` = Exit application
- `option+command+control+x` = Cleanup temporary files

### Linux Hotkeys
- `alt+super` = Simple mode
- `alt+super+ctrl` = Enhanced mode
- `alt+super+o` = Toggle mode
- `alt+super+ctrl+o` = OpenAI Enhanced
- `alt+super+ctrl+g` = Gemini Enhanced
- `alt+super+ctrl+d` = DeepSeek Enhanced
- `alt+super+ctrl+c` = Claude Enhanced
- `alt+super+ctrl+s` = Stop recording
- `esc` = Exit application
- `alt+super+ctrl+x` = Cleanup temporary files

Note: The default transcription mode is Whisper with the ability to toggle to OpenAI API transcription.

## Setting up API Keys

For enhanced transcription using LLM models, you'll need to set up API keys for the respective services:

1. Create a file named `.env` in the project root directory
2. Add your API keys in the following format:
```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
```

If no API keys are provided, the application will fall back to using Ollama with a local model.

### Setting up Ollama (Optional)

For local LLM processing without API keys:

1. Download and install Ollama from [ollama.ai](https://ollama.ai/download)
2. Pull a model: `ollama pull llama3`
3. The application will automatically detect and use Ollama if no API keys are set

## Improvements in This Version

- **Resilient installation scripts** that work with pre-existing dependencies
- **Enhanced error handling** with informative messages
- **PowerShell support** for better Windows integration
- **Automatic FFmpeg detection** with installation suggestions
- **GPU detection** for optimal Whisper model selection
- **Better compatibility** with Python 3.8-3.11
- **uv support** for faster package installation

## Troubleshooting

- **Keyboard hotkeys not working**: 
  - On macOS, grant accessibility permissions to Terminal or your preferred terminal app.
  - On Linux, ensure that you're running the application with appropriate permissions.
  - On Windows, try running as Administrator if hotkeys aren't responding.

- **Audio not recording**:
  - Verify that your microphone is properly connected and selected as the default recording device.
  - Ensure that PortAudio is properly installed.
  - Check that FFmpeg is installed and in your system PATH.

- **LLM enhancement not working**:
  - Verify that your API keys are correctly set up in the `.env` file.
  - Check your internet connection.
  - Ensure that the API services are available and your account is active.
  - For Ollama, ensure the service is running with `ollama serve` command.

- **Installation fails on Windows**:
  - Try running the PowerShell script instead of the batch file.
  - Ensure you have permission to create directories in the project folder.
  - If using uv fails, the scripts will automatically fall back to standard pip.

## License

[MIT License](LICENSE)

### Desktop Shortcuts (auto-generated)

After running the corresponding **install-*.sh / .bat** script a desktop launcher is created automatically so you can start Keyless with a double-click.

| OS | Shortcut created | Notes |
|----|------------------|-------|
| Linux | `~/Desktop/Keyless.desktop` | Uses the provided `icon.png`, runs `run-linux.sh` inside a terminal.  If your file manager asks you to *trust* the launcher, choose *Yes*. |
| macOS | `~/Desktop/Keyless.command` | A self-contained shell script that `cd`s into the project folder and executes `run-mac.sh`.  Double-click to run (macOS may prompt to allow execution the first time). |
| Windows | `Keyless Voice-to-Text.lnk` on your desktop | Points to PowerShell with arguments to execute `run-windows.ps1`, icon set to `icon.png`.  If SmartScreen warns you, choose *More info → Run anyway*. |

**Linux trust prompt**: On GNOME/Unity you may see "Untrusted Desktop File" the first time.  Right‑click the shortcut and select **Allow Launching**.  The installer tries to pre‑approve the file but some desktop environments still show the prompt.

If the icon does not appear, make sure `icon.png` remains inside the project folder; the `.desktop` file uses an absolute path so moving the folder will break the icon until you reinstall.