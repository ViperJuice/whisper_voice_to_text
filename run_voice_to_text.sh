#!/bin/bash
# Run the voice-to-text script with ALSA warnings filtered out and UV environment activated

# Activate the UV environment
cd ~/voice_to_text
source .venv/bin/activate

# Run the script with ALSA warnings filtered
python ~/voice_to_text/voice_to_text.py 2> >(grep -v -E "ALSA lib|pcm_|snd_")