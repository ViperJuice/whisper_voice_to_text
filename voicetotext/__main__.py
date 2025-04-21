"""
Main entry point for voice to text application.
"""

from voicetotext.core.app import VoiceToTextApp

def main():
    """Main function to start the voice to text application"""
    app = VoiceToTextApp()
    app.initialize()
    app.run()

if __name__ == "__main__":
    main() 