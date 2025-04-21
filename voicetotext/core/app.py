"""
Main application logic for the voice to text application.
"""

import os
import sys
import time
import threading
import glob
import wave
import pyaudio
from datetime import datetime
import torch
import logging

# Import modules from the package
from voicetotext.core import audio
from voicetotext.core import transcription
from voicetotext.models import llm
from voicetotext.utils import pynput_keyboard
from voicetotext.utils import config as cfg
from voicetotext.controllers.recorder import RecorderController
from voicetotext.utils import system_checks
from voicetotext.core.transcription import transcribe_with_openai_api
from voicetotext.utils import logging_config

# ---------------------------------------------------------------------------
# Logger setup for this module.  The root logger is configured in
# logging_config.init_logging().
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Replace legacy print() calls in this module with logger.info so that we don't
# have to edit every occurrence manually.  This keeps output concise while
# allowing DEBUG gating via the global logger configuration.
# ---------------------------------------------------------------------------

def _info_print(*args, **kwargs):  # type: ignore[var-annotated]
    logger.info(" ".join(str(a) for a in args))

# Shadow built-in print within this module only.
print = _info_print  # type: ignore[assignment]

class VoiceToTextApp:
    """Main application class for Voice to Text"""
    
    def __init__(self):
        """Initialize the application"""
        self.frames = []
        self.is_recording = False
        self.recording_thread = None
        self.key_monitor_thread = None
        self.exit_event = threading.Event()
        self.enhanced_mode = False
        self.temp_llm_model = None
        self.config = None
        self.whisper_model = None
        self.device = None
        self.system_checks_available = False
        # Always default to OLLAMA as the default LLM
        self.llm_model = "OLLAMA"
        self.temp_base_dir = None
        self.recorder = None
        self.llm = None  # Will be initialized in initialize()
        self.use_local_whisper = True  # Single source of truth
        
        # Try to import system checks
        try:
            from voicetotext.utils import system_checks
            self.system_checks_available = True
            print("System checks module available")
        except ImportError:
            print("Warning: system_checks.py not found. Using default configuration.")
    
    def initialize(self):
        """Initialize application components"""
        # 1. Initialise logging early so everything afterwards
        #    uses the unified logger.  Debug mode can be toggled
        #    via config file or env var.
        debug_mode = bool(os.environ.get("VTT_DEBUG", "0") == "1")
        logging_config.init_logging(debug=debug_mode, redirect_print=False)

        # Expose debug flag to other methods
        self.debug = debug_mode

        logger.info("Voice-to-Text with LLM Enhancement")
        logger.info("=================================")

        try:
            from dotenv import load_dotenv
            
            # Get the application root directory (where .env should be located)
            app_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            env_file = os.path.join(app_root, '.env')
            
            if os.path.exists(env_file):
                print(f"Loading environment variables from: {env_file}")
                load_dotenv(env_file)
                print("✅ Environment variables loaded from .env file")
            else:
                print(f"⚠️ .env file not found at: {env_file}")
        except ImportError:
            print("⚠️ python-dotenv not installed, cannot load .env file")
            print("Install with: uv pip install python-dotenv")
        except Exception as e:
            print(f"⚠️ Error loading .env file: {e}")
        
        # Load configuration
        self.config = cfg.load_config()
        
        # Initialize audio
        audio.initialize_audio()
        
        # Get temporary directory
        self.temp_base_dir = audio.get_temp_dir()
        print(f"Using temp directory: {self.temp_base_dir}")
        
        # Determine which device to use
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Initialize FFmpeg
        self._configure_ffmpeg()
        
        # Initialize Whisper model
        self._load_whisper_model()
        
        # Check for API keys and report available models
        self._check_available_llm_models()
        
        # Initialize LLM module
        from voicetotext.models import llm as llm_module
        self.llm = llm_module
        
        # Initialize recorder controller
        self.recorder = RecorderController(app=self)
        
        # Register hotkeys
        self._register_hotkeys()
        
        # Show that initialization is complete
        print("\n✅ Voice to Text application initialized")
        
        # Overlay/notifications disabled
    
    def _configure_ffmpeg(self):
        """Configure FFmpeg path"""
        # If system_checks is available, let it handle FFmpeg config
        if self.system_checks_available:
            from voicetotext.utils import system_checks
            system_checks.check_ffmpeg()
        else:
            self._configure_ffmpeg_fallback()
    
    def _configure_ffmpeg_fallback(self):
        """Fallback FFmpeg configuration if system_checks is not available"""
        import platform
        import subprocess
        
        if os.name == 'nt':  # Windows
            print("Configuring ffmpeg path for Windows...")
            # Allow override via environment variable
            ffmpeg_path = os.environ.get("FFMPEG_PATH", r"C:\ffmpeg\bin")
            
            # Try different common ffmpeg locations
            possible_paths = [
                ffmpeg_path,
                os.path.join(ffmpeg_path, 'bin'),
                os.path.join(ffmpeg_path, 'ffmpeg'),
                os.path.join(ffmpeg_path, 'ffmpeg', 'bin'),
                os.path.join(os.environ.get('USERPROFILE', ''), 'ffmpeg', 'bin'),
                os.path.join(os.environ.get('USERPROFILE', ''), 'ffmpeg', 'ffmpeg', 'bin'),
            ]
            
            # Check if any of these paths exist
            for path in possible_paths:
                if os.path.exists(path):
                    ffmpeg_path = path
                    print(f"Found ffmpeg at: {ffmpeg_path}")
                    break
            
            # Add ffmpeg to PATH
            if ffmpeg_path not in os.environ['PATH']:
                os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ['PATH']
                print(f"Added ffmpeg to PATH: {ffmpeg_path}")
        else:
            # Basic check for Linux/Mac
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
                print("✓ FFmpeg is installed and available")
            except (subprocess.SubprocessError, FileNotFoundError):
                print("✗ FFmpeg is missing (needed for audio processing)")
                if platform.system() == 'Darwin':  # macOS
                    print("  Install with: brew install ffmpeg")
                else:  # Linux
                    print("  Install with: sudo apt install ffmpeg (Ubuntu/Debian)")
                    print("  or: sudo dnf install ffmpeg (Fedora/RHEL)")
                sys.exit(1)
    
    def _load_whisper_model(self):
        """Load the Whisper model"""
        whisper_size = "base"
        
        # If system_checks is available, use recommended model
        if self.system_checks_available:
            from voicetotext.utils import system_checks
            sys_config = system_checks.system_check()
            if "whisper_model" in sys_config:
                whisper_size = sys_config["whisper_model"]
                print(f"Using recommended Whisper model size: {whisper_size}")
            
            # Check if Ollama is running, but don't change the default model
            try:
                ollama_running, selected_ollama_model, _ = system_checks.check_ollama_status()
                if ollama_running and selected_ollama_model:
                    print(f"Ollama is running with model: {selected_ollama_model}")
                else:
                    print("Ollama not running or no model available")
            except:
                print("Could not check Ollama status")
            
            # Update config with system checks but preserve OLLAMA as default
            cfg.update_config_with_system_checks(self.config, sys_config, "OLLAMA")
        else:
            # Use OLLAMA as the default regardless of config
            self.config["default_llm"] = "OLLAMA"
        
        # Load the model
        self.whisper_model, self.device = transcription.load_whisper_model(whisper_size)
        print(f"Default LLM for enhancement: {self.llm_model}")
    
    def _check_available_llm_models(self):
        """Check which LLM models have API keys available"""
        available_models = []
        missing_keys = []
        
        # Check for OpenAI API key
        if os.environ.get("OPENAI_API_KEY"):
            available_models.append("OPENAI")
        else:
            missing_keys.append(("OPENAI", "OPENAI_API_KEY"))
            
        # Check for Claude API key
        if os.environ.get("ANTHROPIC_API_KEY"):
            available_models.append("CLAUDE")
        else:
            missing_keys.append(("CLAUDE", "ANTHROPIC_API_KEY"))
            
        # Check for Google API key
        if os.environ.get("GOOGLE_API_KEY"):
            available_models.append("GEMINI")
        else:
            missing_keys.append(("GEMINI", "GOOGLE_API_KEY"))
            
        # Check for DeepSeek API key
        if os.environ.get("DEEPSEEK_API_KEY"):
            available_models.append("DEEPSEEK")
        else:
            missing_keys.append(("DEEPSEEK", "DEEPSEEK_API_KEY"))
            
        # Ollama is always available (or at least we'll try to use it)
        available_models.append("OLLAMA")
        
        # Report available models
        if len(available_models) > 1:  # More than just OLLAMA
            print("\nAvailable LLM models for enhancement:")
            for model in available_models:
                print(f" ✓ {model}")
            
            # Keep OLLAMA as the default model
            print(f"Using OLLAMA as the default enhancement model")
        else:
            print("\nOnly OLLAMA is available for enhancement.")
        
        # Show missing API keys
        if missing_keys:
            print("\nMissing API keys:")
            for model, env_var in missing_keys:
                print(f" - {model}: requires {env_var} environment variable")
            print("\nTo use these models, set the corresponding environment variable:")
            print("PowerShell: $env:API_KEY_NAME = 'your-api-key'")
            print("CMD: set API_KEY_NAME=your-api-key")
    
    def _register_hotkeys(self):
        """Register keyboard hotkeys"""
        try:
            # Import the pynput-based keyboard handler
            from voicetotext.utils import pynput_keyboard
            
            # Define callbacks for the keyboard handler
            callbacks = {
                "start_recording": self.start_recording,
                "stop_recording": self.stop_recording,
                "exit": self.exit_application,
                "cleanup": self.cleanup_temp_files,
                "model_switch": self._report_model_switch_during_hold,
                "toggle_transcription": self._toggle_transcription_mode,
                "start_simple": self.start_recording,
                "start_enhanced": self.start_recording
            }
            
            # Create and start the keyboard handler
            self.keyboard_handler = pynput_keyboard.setup_pynput_keyboard(self.config, callbacks)
            print("Using pynput keyboard handler")
            
        except Exception as e:
            print(f"Error registering hotkeys: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)  # Exit if we can't set up keyboard handling
    
    def start_recording(self, enhanced=False):
        """Start the physical audio recording process."""
        try:
            print(f"\n[App Start] Received start_recording call with enhanced={enhanced}")
            
            current_time = time.time()
            last_start_time = getattr(self, "_last_recording_start", 0)
            if current_time - last_start_time < 0.5 or self.is_recording:
                print(f"[App Start] Ignoring rapid/duplicate start call. Elapsed={current_time-last_start_time:.2f}s")
                return # Prevent rapid/duplicate starts
                
            self._last_recording_start = current_time
            self.is_recording = True
            self.frames = []
            self.temp_llm_model = None # Reset any previous model selection
            
            # Set enhanced mode flag explicitly based on the parameter
            self.enhanced_mode = enhanced
            print(f"[App Start] Setting self.enhanced_mode = {self.enhanced_mode}")
            

            print("\n" + "="*60)
            print(f"RECORDING STARTED")
            if enhanced:
                print("Press model keys (o/c/g/d) while holding Ctrl for enhanced mode")
            print("="*60 + "\n")

            # Start the actual audio capture in a thread
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.is_recording = False
    
    def record_audio(self):
        """Thread function to record audio from microphone"""
        try:
            # Reset any previous model selection
            print("\n" + "="*60)
            print("STARTING NEW RECORDING")
            print("="*60 + "\n")
            
            # Get frames from recorder - this is where key monitoring happens
            self.frames = self.recorder.start_recording()
            
        except Exception as e:
            print(f"Error in recording thread: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Make sure we're marked as not recording when done
            self.is_recording = False
    
    def stop_recording(self, final_mode='simple', selected_model=None):
        """Stop recording, determine final mode, and process audio."""
        print(f"\n[App Stop] Received stop signal. final_mode='{final_mode}', selected_model='{selected_model}'")
        
        if not self.is_recording:
            print("[App Stop] Warning: Stop called but not currently recording.")
            self.cleanup_temp_files()
            return # Not recording, nothing to stop

        current_time = time.time()
        last_stop_time = getattr(self, "_last_recording_stop", 0)    
        self._last_recording_stop = current_time
        self.is_recording = False # Mark as stopped immediately

        print(f"[App Stop] Final Mode determined by keyboard handler: {final_mode.upper()}")
        
        # Special handling for toggle mode - don't process the recording
        if final_mode == 'toggle':
            print("[App Stop] Toggle mode detected - skipping audio processing")
            try:
                print("[App Stop] Stopping physical recorder...")
                self.recorder.stop_recording()
            except Exception as e:
                print(f"[App Stop] Error stopping recorder: {e}")
            
            # Wait for recording thread to finish
            if self.recording_thread and self.recording_thread.is_alive():
                try:
                    print("[App Stop] Waiting for recording thread to finish...")
                    self.recording_thread.join(timeout=2.0)
                except Exception as e:
                    print(f"[App Stop] Error joining recording thread: {e}")
            
            # Delete the recorded audio by cleaning up temp files
            print("[App Stop] Cleaning up recorded audio for toggle mode")
            self.cleanup_temp_files()
            return

        # Normal processing for non-toggle modes
        if final_mode == 'enhanced' and selected_model:
            print(f"[App Stop] Specific model selected for enhancement: {selected_model}")
        elif final_mode == 'enhanced':
            print(f"[App Stop] No specific model selected, will use default enhancement model: {self.llm_model}")
        else:
            print("[App Stop] Simple mode selected - no enhancement will be applied.")
            
        # Set the app's enhanced_mode flag based on the final decision from keyboard handler
        self.enhanced_mode = (final_mode == 'enhanced')
        print(f"[App Stop] self.enhanced_mode set to: {self.enhanced_mode}")
        
        # Set the temporary model to be used by process_recording
        self.temp_llm_model = selected_model # This will be None if simple or no key pressed
        print(f"[App Stop] self.temp_llm_model set to: {self.temp_llm_model}")

        # Stop the physical recorder
        try:
            print("[App Stop] Stopping physical recorder...")
            self.recorder.stop_recording()
        except Exception as e:
            print(f"[App Stop] Error stopping recorder: {e}")

        # Wait for the recording thread to finish capturing audio
        if self.recording_thread and self.recording_thread.is_alive():
            try:
                print("[App Stop] Waiting for recording thread to finish...")
                self.recording_thread.join(timeout=2.0)
                if self.recording_thread.is_alive():
                   print("[App Stop] Warning: Recording thread did not finish cleanly.")
                else:
                    print("[App Stop] Recording thread finished.")
            except Exception as e:
                print(f"[App Stop] Error joining recording thread: {e}")
        else:
             print("[App Stop] Recording thread was not active or already finished.")
        
        # Now process the captured audio with the determined mode/model
        print(f"\n[App Stop] Proceeding to process audio. Using temp_llm_model: {self.temp_llm_model}")
        transcription, enhanced_text = self.process_recording(current_model=self.temp_llm_model) # Pass the final decision
        
        # Handle the text output
        if transcription:
            # Determine the final text - use original transcription for simple mode
            final_text = enhanced_text if enhanced_text and self.enhanced_mode else transcription
            
            # Handle the text output (copy to clipboard and paste)
            try:
                self._type_at_cursor(final_text)
            except Exception as e:
                print(f"[App Stop] Error handling text output: {e}")
                print("    (Please copy the text from the terminal output manually)")
        else:
            print("[App Stop] No transcription available to process")

        # Notifications disabled

    def _report_model_switch_during_hold(self, model_name):
        """Handles the model_switch callback from keyboard handler (for potential immediate feedback)"""
        # This is primarily for *feedback* during the hold. The final decision uses temp_llm_model set in stop_recording.
        print("\n" + "="*60)
        print(f"[App Report] Model key '{model_name}' detected during hold.")
        # Optionally, check API key availability here for immediate user feedback
        # Example:
        # if model_name == "OPENAI" and not llm.is_openai_available():
        #     print("[App Report] Warning: OpenAI API key not found!")
        print("="*60)
        # We don't set self.temp_llm_model here anymore, that happens in stop_recording
        # self.temp_llm_model = model_name.upper() if model_name else "OLLAMA"
        # if hasattr(self, 'recorder') and self.recorder:
        #     self.recorder.selected_model = self.temp_llm_model
        # return self.temp_llm_model 
    
    def process_recording(self, current_model=None):
        """Process the recording to transcribe and optionally enhance it using LLM."""
        print(f"\n[App Process] Starting processing. Received current_model parameter: '{current_model}'")
        print(f"[App Process] Current app state: self.enhanced_mode = {self.enhanced_mode}")
        print(f"[App Process] Current app state: self.temp_llm_model = {self.temp_llm_model}")
        
        try:
            print("\n" + "="*60)
            if self.enhanced_mode:
                print(f"PROCESSING RECORDING WITH LLM ENHANCEMENT (Model: {current_model or 'ollama'})")
            else:
                print("PROCESSING RECORDING (SIMPLE MODE - NO ENHANCEMENT)")
            print("="*60 + "\n")
            
            # Ensure there is audio data to process
            if not self.recorder.audio_data:
                print("[App Process] No audio data found in recorder.")
                return None, None
                
            # Create a timestamp for filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
            filepath = os.path.join(self.temp_base_dir, filename)
            
            try:
                # Save the audio data
                print(f"[App Process] Saving recording to {filepath}")
                with wave.open(filepath, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self.recorder.sample_rate)
                    wf.writeframes(self.recorder.audio_data)
            except Exception as e:
                print(f"[App Process] Error saving audio: {e}")
                return None, None
            
            # Initialize transcription variable
            transcription = None
            
            # Transcribe the audio
            print("[App Process] Transcribing audio...")
            try:
                if not self.use_local_whisper:
                    # Use OpenAI API for transcription
                    api_key = self.llm.get_api_key("OPENAI")
                    if not api_key:
                        print("[App Process] OpenAI API key not found, falling back to local Whisper")
                        self.use_local_whisper = True
                    else:
                        # Get language from config if available
                        language = self.config.get("transcription", {}).get("language")
                        temperature = self.config.get("transcription", {}).get("temperature", 0.0)
                        
                        # Use OpenAI API for transcription
                        transcription = transcribe_with_openai_api(
                            filepath, 
                            api_key,
                            language=language,
                            temperature=temperature
                        )
                        if not transcription:
                            print("[App Process] OpenAI API transcription failed, falling back to local Whisper")
                            self.use_local_whisper = True
                
                if self.use_local_whisper:
                    # Use local Whisper model
                    transcription = self.whisper_model.transcribe(filepath)["text"].strip()
                
                if not transcription:
                    print("[App Process] Transcription empty")
                    return None, None
                
                print(f"[App Process] Transcription: {transcription}")
                self.cleanup_temp_files()
            except Exception as e:
                print(f"[App Process] Error during transcription: {e}")
                return None, None
            
            # Only proceed with enhancement if we have a valid transcription
            if not transcription:
                print("[App Process] No transcription available to process")
                return None, None
                
            enhanced_text = None
            
            # Print decision logic for enhanced mode
            print(f"\n[App Process] ENHANCEMENT DECISION:")
            print(f"[App Process]   - self.enhanced_mode = {self.enhanced_mode}")
            print(f"[App Process]   - current_model parameter = {current_model}")
            
            # Only enhance with LLM if in enhanced mode
            if self.enhanced_mode:
                # Determine the model to use for enhancement:
                # Use the 'current_model' parameter passed from stop_recording
                # This parameter now holds the definitive model choice
                model_to_use = current_model if current_model else self.llm_model
                
                # Ensure model name is uppercase
                model_to_use = model_to_use.upper() if model_to_use else "OLLAMA"
                
                print("\n" + "="*60)
                print(f"[App Process] Attempting Enhancement with final model: {model_to_use}")
                print("="*60)
                
                # Print debug information about model source (simplified)
                print("Processing Final Model Check:")
                print(f"  - Using model: '{model_to_use}' (from stop_recording decision)")
                print("-"*60)
                
                # Verify self.llm is properly set
                if not hasattr(self, 'llm') or self.llm is None:
                    print(f"[App Process] LLM module not initialized, initializing now")
                    from voicetotext.models import llm as llm_module
                    self.llm = llm_module
                
                # Check if the selected model has a valid API key
                api_key_valid = True
                original_model_request = model_to_use # Keep track for fallback message
                
                if model_to_use == "OPENAI" and not self.llm.is_openai_available():
                    print(f"[App Process] OpenAI API key not found - falling back to OLLAMA")
                    print(f"  (To use OpenAI, set the OPENAI_API_KEY environment variable)")
                    model_to_use = "OLLAMA"
                    api_key_valid = False
                elif model_to_use == "CLAUDE" and not self.llm.is_claude_available():
                    print(f"[App Process] Claude API key not found - falling back to OLLAMA")
                    print(f"  (To use Claude, set the ANTHROPIC_API_KEY environment variable)")
                    model_to_use = "OLLAMA"
                    api_key_valid = False
                elif model_to_use == "GEMINI" and not self.llm.is_google_available():
                    print(f"[App Process] Google API key not found - falling back to OLLAMA")
                    print(f"  (To use Gemini, set the GOOGLE_API_KEY environment variable)")
                    model_to_use = "OLLAMA"
                    api_key_valid = False
                elif model_to_use == "DEEPSEEK" and not self.llm.is_deepseek_available():
                    print(f"[App Process] DeepSeek API key not found - falling back to OLLAMA")
                    print(f"  (To use DeepSeek, set the DEEPSEEK_API_KEY environment variable)")
                    model_to_use = "OLLAMA"
                    api_key_valid = False
                
                # Final model selection message
                if not api_key_valid:
                     print(f"[App Process] Using FALLBACK model due to missing API key: {model_to_use}")
                else:
                    print(f"[App Process] Using selected model: {model_to_use}")
                
                try:
                    print(f"[App Process] Enhancing transcription with {model_to_use}...")
                    
                    # Call enhance_with_llm with specific model
                    enhanced_text = self.llm.enhance_with_llm(transcription, llm_model=model_to_use)
                    
                    if enhanced_text == transcription:
                        print("[App Process] LLM enhancement returned the original text")
                    else:
                        print(f"[App Process] Enhancement complete!")
                        print(f"  Enhanced text: {enhanced_text}")
                except Exception as e:
                    print(f"[App Process] Error during LLM enhancement: {e}")
                    import traceback
                    traceback.print_exc()
                    # If enhancement fails, fall back to original transcription
                    enhanced_text = transcription
            else:
                # In simple mode, skip enhancement entirely
                print("[App Process] Simple mode - skipping LLM enhancement")
                enhanced_text = None  # Ensure no enhancement is used
            
            # Determine the final text - use original transcription for simple mode
            final_text = enhanced_text if enhanced_text and self.enhanced_mode else transcription
            
            # Print final text selection decision
            print("\n[App Process] FINAL TEXT DECISION:")
            print(f"[App Process]   - self.enhanced_mode = {self.enhanced_mode}")
            print(f"[App Process]   - enhanced_text available = {enhanced_text is not None}")
            print(f"[App Process]   - Using {'enhanced' if enhanced_text and self.enhanced_mode else 'original'} text")
            
            # Return the text to be handled by the caller
            return transcription, enhanced_text
        except Exception as e:
            print(f"[App Process] Error in process_recording: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _type_at_cursor(self, text):
        """Copy text to clipboard and attempt to paste it at cursor using pynput"""
        if not text:
            print("No text to type")
            return
            
        # Clean and normalize the text
        text = text.strip()
        
        # Print the text to terminal first
        print("\n" + "="*50)
        print("FINAL TEXT OUTPUT:")
        print("="*50)
        print(text)
        print("="*50 + "\n")
            
        # Copy to clipboard and verify
        try:
            import pyperclip
            pyperclip.copy(text)
            # Verify clipboard content
            clipboard_text = pyperclip.paste()
            if clipboard_text == text:
                print("✅ Text successfully copied to clipboard")
            else:
                print("⚠️ Clipboard verification failed - text mismatch")
                print(f"Expected: '{text}'")
                print(f"Got: '{clipboard_text}'")
                return
        except Exception as e:
            print(f"❌ Error copying to clipboard: {e}")
            print("Could not copy text. Please copy the output from the console.")
            return

        # Attempt to paste at cursor
        try:
            from pynput.keyboard import Controller, Key
            import time
            
            # Create keyboard controller
            keyboard = Controller()
            
            # Stop the keyboard listener if it's running
            if hasattr(self, 'keyboard_handler') and self.keyboard_handler and self.keyboard_handler.listener.is_alive():
                print("[Typing] Stopping keyboard listener...")
                self.keyboard_handler.listener.stop()
                # Wait for listener to fully stop
                time.sleep(0.5)
            
            print("Attempting to paste text at cursor...")
            
            # Press Ctrl+V to paste
            with keyboard.pressed(Key.ctrl):
                keyboard.press('v')
                keyboard.release('v')
            
            # Add a space after pasting
            keyboard.press(' ')
            keyboard.release(' ')
            
            print("✅ Text pasting attempted")
            
        except Exception as e:
            print(f"❌ Error attempting to paste text: {e}")
            print("(This can sometimes fail due to focus issues or permissions)")
        finally:
            # Restart the keyboard listener
            try:
                if hasattr(self, 'keyboard_handler') and self.keyboard_handler:
                    print("[Typing] Restarting keyboard listener...")
                    self.keyboard_handler.restart()
                    print("✅ Keyboard listener restarted successfully")
            except Exception as restart_e:
                print(f"❌ Error restarting keyboard listener: {restart_e}")
                print("Please restart the application to restore hotkey functionality.")

    def _save_audio_data(self, audio_data, filename):
        """Save audio data to a WAV file."""
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)
                wf.writeframes(audio_data)
            print(f"✅ Audio saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving audio: {e}")
            raise

    def _transcribe_audio(self, audio_file):
        """Transcribe audio file using the loaded whisper model."""
        try:
            print("🔄 Transcribing audio...")
            result = self.whisper_model.transcribe(audio_file)
            return result["text"]
        except Exception as e:
            print(f"❌ Error transcribing audio: {e}")
            raise
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during recording"""
        try:
            # Stop any ongoing recording
            if self.is_recording:
                print("\nStopping ongoing recording...")
                self.is_recording = False
                if self.recording_thread and self.recording_thread.is_alive():
                    self.recording_thread.join(timeout=1)
            
            # Clean up audio resources
            audio.cleanup_audio_resources()
            
            # Delete temporary .wav files
            temp_files = glob.glob(os.path.join(self.temp_base_dir, "*.wav"))
            for file in temp_files:
                try:
                    os.remove(file)
                    print(f"Deleted temporary file: {file}")
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
            
            # Delete temporary .txt files
            temp_files = glob.glob(os.path.join(self.temp_base_dir, "*.txt"))
            for file in temp_files:
                try:
                    os.remove(file)
                    print(f"Deleted temporary file: {file}")
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
            
            print("Cleanup completed.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def exit_application(self):
        """Exit the application cleanly"""
        # Guard against multiple exit calls
        if hasattr(self, '_exiting') and self._exiting:
            print("Already exiting, forcing termination")
            import os
            os._exit(0)
            return
            
        # Mark that we're in the process of exiting
        self._exiting = True
        
        try:
            print("\n👋 Exiting voice-to-text application...")
            
            # Stop the keyboard handler if it exists
            if hasattr(self, 'keyboard_handler') and self.keyboard_handler:
                try:
                    self.keyboard_handler.stop()
                    # Wait for the listener to fully stop
                    if self.keyboard_handler.listener:
                        self.keyboard_handler.listener.join(timeout=1.0)
                    print("Keyboard handler stopped")
                except Exception as e:
                    print(f"Error stopping keyboard handler: {e}")
            
            # Set the main event to signal the application to stop
            if hasattr(self, 'main_event'):
                self.main_event.set()
            
            # Stop any active recording
            if self.is_recording:
                self.is_recording = False
                
                # Also signal the recorder to stop
                if self.recorder:
                    try:
                        self.recorder.stop_recording()
                    except:
                        pass
                    
                try:
                    # Close audio stream directly instead of calling stop_recording
                    if audio.stream:
                        try:
                            audio.stream.stop_stream()
                            audio.stream.close()
                            audio.stream = None
                        except Exception as e:
                            print(f"Error closing audio stream: {e}")
                except Exception as e:
                    print(f"Error during recording cleanup: {e}")
            
            # No need to unhook keyboard since we're using pynput
            
        except Exception as e:
            print(f"Error during exit: {e}")
        finally:
            # Clean up resources
            try:
                self.cleanup_temp_files()
            except:
                pass
            
            # Force exit to ensure all threads are terminated
            import os
            os._exit(0)
    
    def run(self):
        """Main method to run the application"""
        try:
            # Show application information
            if self.debug:
                banner_lines = [
                    "\n" + "="*60,
                    "  VOICE TO TEXT WITH LLM ENHANCEMENT",
                    "="*60 + "\n",
                    "HOTKEYS:",
                    f"  - Simple Mode: {self.config['hotkeys']['simple_mode']}",
                    f"  - Enhanced Mode: {self.config['hotkeys']['enhanced_mode']}",
                    f"  - Toggle Transcription: {self.config['hotkeys']['simple_mode']}+t",
                    f"  - Exit: {self.config['hotkeys']['exit']}",
                    f"  - Cleanup Temp Files: {self.config['hotkeys']['cleanup']}",
                    "\nWhen recording in Enhanced Mode, press one of these keys WHILE HOLDING THE HOTKEYS to select an LLM model:",
                    "  - o: OPENAI  (requires OPENAI_API_KEY)",
                    "  - c: CLAUDE  (requires ANTHROPIC_API_KEY)",
                    "  - g: GEMINI  (requires GOOGLE_API_KEY)",
                    "  - d: DEEPSEEK (requires DEEPSEEK_API_KEY)",
                    "    (If no key is pressed, the default OLLAMA model will be used)",
                    "\nRelease the hotkey combination to stop recording and process.",
                    "\n" + "="*60,
                    "  READY! Press hotkeys to start recording.",
                    "="*60 + "\n",
                ]
                for line in banner_lines:
                    logger.info(line)
            else:
                logger.info("Ready. Press the configured hotkeys to start recording (set VTT_DEBUG=1 for full help).")

            # Use a more resilient way to keep the application running
            import threading
            self.main_event = threading.Event()
            
            # Register a handler for clean exit
            import signal
            def signal_handler(sig, frame):
                logger.info("Signal handler activated - exiting application")
                self.exit_application()
                
            # Register signal handlers for common exit signals
            signal.signal(signal.SIGINT, signal_handler)
            if hasattr(signal, 'SIGBREAK'):  # Windows
                signal.signal(signal.SIGBREAK, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Keep the application running until the event is set
            try:
                while not self.main_event.is_set():
                    self.main_event.wait(1.0)  # Wait with timeout to allow handling signals
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, exiting...")
                self.exit_application()
        
        except Exception as e:
            print(f"Error in main function: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up resources when exiting
            try:
                self.cleanup_temp_files()
            except:
                pass

    def set_mode(self, mode="simple"):
        """Set the recording mode (simple or enhanced)"""
        if mode in ["simple", "enhanced"]:
            self.mode = mode
            logger.info("Recording mode set to: %s", mode)
        else:
            logger.warning("Invalid mode '%s', using 'simple'", mode)
            self.mode = "simple" 

    def switch_model(self, model_name):
        """Force switch the model regardless of current state"""
        logger.info("\n" + "="*60)
        logger.info("DIRECT MODEL SWITCH: %s", model_name)
        logger.info("="*60)
        
        self.temp_llm_model = model_name.upper() if model_name else "OLLAMA"
        
        # Also update in recorder if available
        if hasattr(self, 'recorder') and self.recorder:
            self.recorder.selected_model = self.temp_llm_model
            
        logger.info("Model switched to: %s", self.temp_llm_model)
        return self.temp_llm_model 

    def _toggle_transcription_mode(self):
        """Toggle between local Whisper and OpenAI API transcription."""
        # First check if local Whisper is available
        local_available = self.whisper_model is not None
        if not local_available:
            print("\n⚠️ Cannot toggle: Local Whisper model not loaded")
            print("This is a critical error - the application requires Whisper")
            return

        # Check OpenAI API key before allowing toggle
        openai_key = self.llm.get_api_key("OPENAI")
        openai_available = openai_key is not None and openai_key != 'your_openai_key_here'
        
        if not openai_available:
            print("\n⚠️ Cannot toggle: OpenAI API key not properly configured")
            print("To enable OpenAI API transcription, please:")
            print("1. Create a file named '.env' in the project root")
            print("2. Add your OpenAI API key: OPENAI_API_KEY=your_key_here")
            print("3. Restart the application")
            print("\nContinuing with Local Whisper only")
            # Force local Whisper mode
            self.use_local_whisper = True
            return
            
        # If we get here, both services are available
        self.use_local_whisper = not self.use_local_whisper
        current_service = "Local Whisper" if self.use_local_whisper else "OpenAI API"
        print(f"\n✅ Switched to {current_service} for transcription")
        
        # Print current status
        print("\nCurrent Transcription Service Status:")
        print(f"  - Local Whisper: {'✓' if self.use_local_whisper else '•'}")
        print(f"  - OpenAI API: {'•' if self.use_local_whisper else '✓'}")
        print("\nPress Windows+Alt+T again to toggle back.")

    def _update_overlay(self, text: str):
        pass  # overlay disabled