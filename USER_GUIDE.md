# User Guide: Dual-Mode Voice-to-Text

This guide provides comprehensive instructions for using the voice-to-text tool with both simple and enhanced modes.

## Quick Start

1. **Launch the application**:
   ```bash
   ./run_voice_to_text.sh
   ```

2. **Use Simple Mode**:
   - Press and hold **Super+Space** (Windows/Command key + Space)
   - Speak into your microphone
   - Release the keys to process and insert text

3. **Use Enhanced Mode**:
   - Press and hold **Super+Alt+Space**
   - Speak into your microphone
   - Release the keys to process with LLM refinement

4. **Switch LLM Modes** (Enhanced Mode only):
   - Press **Super+Alt+Space+O** for OpenAI
   - Press **Super+Alt+Space+C** for Anthropic
   - Press **Super+Alt+Space+G** for Google Flash Light
   - Press **Super+Alt+Space+D** for Deepseek Lite
   - Press **Super+Alt+Space+L** for Local LLM

5. **Exit the application**:
   - Press **Esc** key

## Detailed Features

### Simple Mode (Super+Space)

Simple Mode provides basic voice-to-text conversion with algorithmic cleanup:

- Removes common filler words (uh, um, like, etc.)
- Eliminates repetitions
- Fixes basic spacing and punctuation
- Capitalizes first words of sentences
- Maintains most of your original wording

This mode is:
- Fast (processes immediately)
- Works completely offline
- Ideal for quick notes and commands

### Enhanced Mode (Super+Alt+Space)

Enhanced Mode uses a flexible LLM system to significantly improve text quality:

- Intelligently removes filler words and unnecessary phrases
- Improves sentence structure and clarity
- Maintains your personal speaking style
- Makes text more concise and professional

You can choose between five LLM options:

1. **OpenAI API** (default)
   - Powerful cloud-based model
   - Requires API key
   - Best for complex language tasks

2. **Anthropic API**
   - High-quality cloud-based model
   - Requires API key
   - Excellent for natural language

3. **Google Flash Light**
   - Fast and efficient cloud-based model
   - Requires API key
   - Great for quick responses

4. **Deepseek Lite**
   - Lightweight cloud-based model
   - Requires API key
   - Good balance of speed and quality

5. **Local Ollama LLM**
   - Private and fast
   - Works completely offline
   - Requires Ollama installation

The system will:
- Use your selected LLM mode first
- Automatically fall back to other available options if the selected mode fails
- Finally use algorithmic cleanup if all LLM options fail

### LLM Mode Switching

You can switch between LLM modes at any time using these hotkeys:

- **Super+Alt+Space+O**: Switch to OpenAI mode
- **Super+Alt+Space+C**: Switch to Anthropic mode
- **Super+Alt+Space+G**: Switch to Google Flash Light mode
- **Super+Alt+Space+D**: Switch to Deepseek Lite mode
- **Super+Alt+Space+L**: Switch to Local LLM mode

The current mode is displayed in the terminal, and the system will remember your selection until you change it.

## Optimizing Performance

### Audio Quality Tips

For best transcription results:
- Speak clearly at a moderate pace
- Use a good quality microphone
- Minimize background noise
- Position the microphone properly (typically 6-12 inches from mouth)
- Wait for the "Recording started..." message before speaking

### Transcription Speed

Factors affecting transcription speed:
- **GPU performance**: Using a CUDA-compatible GPU dramatically improves speed
- **Whisper model size**: Smaller models (base, small) are faster but less accurate
- **Recording length**: Longer recordings take proportionally longer to process
- **LLM enhancement**: Enhanced mode takes longer due to LLM processing

### Memory Usage

- The Whisper model requires significant memory (especially for medium/large models)
- GPU VRAM requirements:
  - base: ~1GB VRAM
  - small: ~2GB VRAM
  - medium: ~5GB VRAM
  - large: ~10GB VRAM

## LLM Configuration Options

### Local Ollama Setup

For best performance and privacy, use the local Ollama option:

1. Install Ollama:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. Pull a model (one-time setup):
   ```bash
   ollama pull llama3
   ```

3. Start the Ollama server:
   ```bash
   ollama serve
   ```

The application automatically selects the best available model from your Ollama installation.

### Cloud API Options

To enable API fallback options, you'll need to obtain and configure API keys for each service you want to use. Here's how to get started with each provider:

#### OpenAI API
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in to your OpenAI account
3. Navigate to the API keys section
4. Click "Create new secret key"
5. Copy the generated key
6. Set the environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

#### Anthropic API
1. Visit [Anthropic Console](https://console.anthropic.com/account/keys)
2. Create an account or log in
3. Navigate to the API keys section
4. Generate a new API key
5. Copy the key
6. Set the environment variable:
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

#### Google Flash Light
1. Visit [Google MakerSuite](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new project or select an existing one
4. Enable the necessary APIs
5. Create credentials and generate an API key
6. Set the environment variable:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

#### Deepseek Lite
1. Visit [Deepseek Platform](https://platform.deepseek.com/)
2. Create an account or log in
3. Navigate to the API section
4. Generate a new API key
5. Copy the key
6. Set the environment variable:
   ```bash
   export DEEPSEEK_API_KEY="your-api-key-here"
   ```

#### Making API Keys Permanent
To make your API keys permanent (so they persist after system restarts):

##### Windows:
1. Open System Properties
2. Click "Environment Variables"
3. Under "User variables", click "New"
4. Add each API key as a new variable

##### macOS/Linux:
Add the export commands to your shell configuration file:
```bash
# For bash users
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.bashrc
echo 'export GOOGLE_API_KEY="your-api-key-here"' >> ~/.bashrc
echo 'export DEEPSEEK_API_KEY="your-api-key-here"' >> ~/.bashrc

# For zsh users
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.zshrc
echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.zshrc
echo 'export GOOGLE_API_KEY="your-api-key-here"' >> ~/.zshrc
echo 'export DEEPSEEK_API_KEY="your-api-key-here"' >> ~/.zshrc
```

After adding the keys, restart your terminal or run:
```bash
source ~/.bashrc  # for bash
# or
source ~/.zshrc   # for zsh
```

#### API Key Security Tips
- Never share your API keys or commit them to version control
- Use different keys for development and production
- Regularly rotate your API keys
- Monitor your API usage to prevent unexpected charges
- Consider using a secrets manager for production environments

#### Verifying API Keys
The application will automatically check your API keys when it starts. You'll see a status message for each configured key:
- ✓ Green checkmark: Key is configured and valid
- ✗ Red X: Key is missing or invalid

If you see any red X marks, double-check your API key configuration and try again.

## Troubleshooting

### Common Issues

**No text appearing after speaking:**
- Ensure your microphone is working and properly selected
- Check that you're speaking loud enough
- Try a shorter phrase first to test

**LLM enhancement not working:**
- Verify Ollama is running with `ollama serve`
- Check API keys if using cloud options
- Look for error messages in the terminal

**Application crashes or freezes:**
- Try using a smaller Whisper model
- Ensure you have sufficient RAM and VRAM
- Check the terminal for error messages

### Debugging

If you encounter issues:
1. Check terminal output for error messages
2. Verify system dependencies are installed
3. Confirm microphone permissions are set correctly
4. Try running with explicit Python path:
   ```bash
   python3 voice_to_text.py
   ```

## Advanced Configuration

You can modify these variables in `voice_to_text.py` to customize behavior:

- **Audio parameters**:
  ```python
  CHUNK = 1024 * 4  # Audio chunk size
  RATE = 16000      # Sample rate (keep at 16kHz for Whisper)
  ```

- **LLM model selection**:
  ```python
  LOCAL_LLM_MODEL = "llama3"  # Default Ollama model
  ```

- **Filler word patterns** (in the `algorithmic_cleanup` function):
  ```python
  filler_words = [
      r'\buh\b', r'\bum\b', r'\ber\b', r'\blike\b(?! to)', r'\byou know\b', 
      # Add your own patterns here
  ]
  ```