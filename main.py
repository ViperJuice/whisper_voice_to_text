#!/usr/bin/env python3
"""
Voice-to-text application main entry point.
Press and hold Windows+Alt for simple transcription.
Press and hold Windows+Alt+Ctrl for enhanced transcription with LLM.
"""

import os
import sys
import logging
from voicetotext.utils import logging_config

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voicetotext.core.app import VoiceToTextApp

logger = logging.getLogger(__name__)

# Initialise logging before anything else
debug_mode = bool(os.environ.get("VTT_DEBUG", "0") == "1")
logging_config.init_logging(debug=debug_mode, redirect_print=False)

def main():
    """Main function to run the voice-to-text application"""
    app = VoiceToTextApp()
    try:
        app.initialize()
        app.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user, exiting...")
        try:
            app.exit_application()
        except Exception:
            pass
    except Exception:
        logger.exception("Fatal error in top‑level application loop")
        # Ensure clean exit
        try:
            app.exit_application()
        except Exception:
            pass

if __name__ == "__main__":
    main() 