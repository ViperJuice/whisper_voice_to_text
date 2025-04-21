# Keyboard Handling Flow

## Recording Flow

1. Start Recording:
   - When Windows+Alt are pressed, begin recording immediately
   - Recording continues while Windows+Alt are held down

2. End Recording (on Windows+Alt release):
   - Check for additional keys pressed during recording:
     a. If Ctrl was pressed:
        - Enter Enhanced Mode
        - Check for model selection keys:
          * O: Use Ollama model
          * G: Use GPT model
          * D: Use DeepSeek model
          * C: Use Claude model
          * No key: Use default enhanced mode
     b. If T was pressed (without Ctrl):
        - Delete recorded audio
        - Toggle to other transcription endpoint
        - No transcription of current audio
     c. If no special keys:
        - Transcribe in Simple Mode

3. Other Hotkeys:
   - Stop: Alt+Windows+Ctrl+S
   - Exit: Esc
   - Cleanup: Alt+Windows+Ctrl+X 