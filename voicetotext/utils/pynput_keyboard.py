"""
Keyboard handling using pynput for the voice-to-text application.
This replaces the previous keyboard module to improve hotkey detection reliability.
"""

import platform
import threading
import time
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
import sys
import logging

# ---------------------------------------------------------------------------
# Module‑level logger; all legacy print calls in this file are converted to
# DEBUG‑level logging so they are hidden unless VTT_DEBUG=1 (or root log level
# is DEBUG).  This avoids editing hundreds of individual print statements.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _debug_print(*args, **kwargs):  # type: ignore[var-annotated]
    """Internal helper that forwards *print()* messages to logger.debug()."""
    logger.debug(" ".join(str(a) for a in args))


# Shadow built‑in print in *this module only* so all existing calls now emit
# debug‑level log lines instead of writing directly to stdout.
print = _debug_print  # type: ignore[assignment]

class PynputKeyboardHandler:
    """Keyboard handler using pynput for improved hotkey detection reliability"""
    
    # Key code mappings for special keys
    KEY_MAP = {
        91: 'win',     # Windows key (Left)
        92: 'win',     # Windows key (Right)
        17: 'ctrl_l',  # Left Ctrl
        18: 'alt_l',   # Left Alt
        27: 'esc',     # Escape
        79: 'o',       # O key
        71: 'g',       # G key
        68: 'd',       # D key
        67: 'c',       # C key
        84: 't'        # T key
    }

    # Special key mapping for pynput Key enum
    SPECIAL_KEY_MAP = {
        Key.alt_l: (18, 'alt_l'),
        Key.ctrl_l: (17, 'ctrl_l'),
        Key.cmd: (91, 'win'),
        Key.esc: (27, 'esc'),
    }
    
    def __init__(self, config):
        """Initialize the keyboard handler"""
        self.config = config
        self.listener = None
        self.pressed_keys = set()
        self.recording_started = False
        self.selected_model = None
        self.callbacks = {}
        self.enhanced_mode = False  # Track enhanced mode state
        self.last_win_press = 0  # Track last Windows key press time
        self.recording_start_time = 0  # Track when recording started
        self.toggle_mode_detected = False
        self._setup_hotkeys()
        self._start_listener()
        
    def _start_listener(self):
        """Start the keyboard listener"""
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False
        )
        self.listener.start()
        print("\nKeyboard listener started")
        self._print_hotkeys()
        
    def _normalize_key(self, key):
        """Normalize a key to its string representation"""
        try:
            print(f"\nNormalizing key: {key}")
            print(f"Key type: {type(key)}")
            
            # First check for virtual key codes (highest priority)
            if hasattr(key, 'vk') and key.vk is not None:
                print(f"Found virtual key code: {key.vk}")
                if key.vk in self.KEY_MAP:
                    normalized = self.KEY_MAP[key.vk]
                    print(f"Virtual key code {key.vk} mapped to: {normalized}")
                    return normalized
                print(f"Virtual key code {key.vk} not in mapping")
            
            # Then check for special keys using Key enum
            if isinstance(key, Key):
                if key in self.SPECIAL_KEY_MAP:
                    vk, name = self.SPECIAL_KEY_MAP[key]
                    print(f"Special key {key} mapped to: {name} (vk: {vk})")
                    return name
                # Handle platform-specific cases
                if key == Key.cmd:
                    print("Found Command/Windows key")
                    return 'win' if sys.platform == 'win32' else 'cmd'
                print(f"Special key {key} not in mapping")
            
            # Finally check for character keys
            if hasattr(key, 'char') and key.char:
                normalized = key.char.lower()
                print(f"Character key normalized to: {normalized}")
                return normalized
            
            print("No valid key mapping found")
            return None
            
        except Exception:
            logger.exception("Unhandled error in _normalize_key")
            return None

    def _check_hotkey(self, hotkey_name):
        """Check if a specific hotkey is currently pressed."""
        required_keys = self.hotkeys[hotkey_name]["required_keys"] 
        current_keys_set = set(self.pressed_keys)
        
        # Debug output to help diagnose issues
        print(f"\nChecking hotkey {hotkey_name}:")
        print(f"Required keys: {sorted(required_keys)}")
        print(f"Current keys: {sorted(current_keys_set)}")
        
        # Check if all required keys are pressed (order independent)
        # Set comparison: required keys must be a subset of current keys
        if required_keys.issubset(current_keys_set):
            print("Hotkey match: True")
            print(f"Matched hotkey: {hotkey_name}")
            print("Required keys found in current keys")
            return True
        else:
            print("Hotkey match: False")
            # Show which keys are missing
            missing_keys = required_keys - current_keys_set
            print(f"Missing keys: {missing_keys}")
            return False

    def _check_hotkeys_in_priority(self, hotkey_names):
        """Check hotkeys in priority order, returning the first match"""
        for hotkey_name in hotkey_names:
            if self._check_hotkey(hotkey_name):
                return hotkey_name
        return None

    def _on_press(self, key):
        """Handle key press events"""
        try:
            key_name = self._normalize_key(key)
            if not key_name:
                return

            self.pressed_keys.add(key_name)
            print(f"\nKey pressed: {key_name}")
            print(f"Current pressed keys: {sorted(self.pressed_keys)}")
            
            # Check for exit hotkey first (highest priority)
            if key_name == "esc":  # Simplified check for exit
                print("Exit hotkey detected")
                if self.callbacks.get("exit"):
                    self.callbacks["exit"]()
                return False  # Stop listener
    
            # Check for enhanced mode first (highest priority)
            if self._check_hotkey("start_enhanced"):
                print("Enhanced mode hotkey detected")
                self.enhanced_mode = True
                self.selected_model = "ollama"  # Default for enhanced mode
                print(f"Enhanced mode active with model: {self.selected_model}")
                
                # Check for model selection immediately
                if self._check_hotkey("start_openai"):
                    self.selected_model = "openai"
                    print("Model selected: OpenAI")
                elif self._check_hotkey("start_gemini"):
                    self.selected_model = "gemini"
                    print("Model selected: Gemini")
                elif self._check_hotkey("start_deepseek"):
                    self.selected_model = "deepseek"
                    print("Model selected: Deepseek")
                elif self._check_hotkey("start_claude"):
                    self.selected_model = "claude"
                    print("Model selected: Claude")
                
                # Start recording if not already recording
                if not self.recording_started:
                    print("Starting recording in enhanced mode")
                    self.callbacks["start_recording"]()
                    self.recording_started = True
                    self.recording_start_time = time.time()
                return
        
            # Then check for simple mode (only if enhanced mode keys aren't pressed)
            if self._check_hotkey("start_simple") and not self._check_hotkey("start_enhanced"):
                if self._check_hotkey("toggle_mode"):
                    print("Toggle mode hotkey detected")
                    self.toggle_mode_detected = True
                    return
                # Start recording if not already recording
                if not self.recording_started:
                    print("Starting recording in simple mode")
                    self.callbacks["start_recording"]()
                    self.recording_started = True
                    self.recording_start_time = time.time()
                return
                    
        except Exception:
            logger.exception("Unhandled error in _on_press")

    def _on_release(self, key):
        """Handle key release events"""
        try:
            key_name = self._normalize_key(key)
            if not key_name:
                return

            print(f"\nKey released: {key_name}")
            print(f"Current keys before removal: {sorted(self.pressed_keys)}")
            
            # Keep a copy of the full hotkey combination before removing the key
            full_hotkey = set(self.pressed_keys)
            if key_name in self.pressed_keys:
                self.pressed_keys.remove(key_name)
            
            print(f"Keys after removal: {sorted(self.pressed_keys)}")
            print(f"Full hotkey that was active: {sorted(full_hotkey)}")
            print(f"Current state: recording={self.recording_started}, enhanced={self.enhanced_mode}, model={self.selected_model}")
            
            # Handle toggle mode (special case that stops recording if active)
            if not self.pressed_keys and self.toggle_mode_detected:
                print("Toggle mode activated on release")
                if self.recording_started:
                    print("Stopping recording for toggle mode")
                    self.callbacks["stop_recording"](final_mode="toggle", selected_model=None)
                    self.recording_started = False
                    self.toggle_mode_detected = False
                
                self.callbacks["toggle_transcription"]()
                self.enhanced_mode = False
                self.selected_model = None
                self.callbacks["cleanup"]()
                return

            # If all keys are released and we are recording, check if we should stop
            if self.recording_started and not self.pressed_keys:
                # Get current time
                current_time = time.time()
                # Calculate time since recording started
                elapsed_time = current_time - getattr(self, 'recording_start_time', 0)
                # Final mode determination based on enhanced_mode flag
                if self.enhanced_mode:
                    final_model = self.selected_model
                    print(f"Enhanced mode detected, using model: {final_model}")
                    self.callbacks["stop_recording"](final_mode="enhanced", selected_model=final_model)
                else:
                    print("Simple mode detected")
                    self.callbacks["stop_recording"](final_mode="simple", selected_model=None)
                
                # Reset state
                self.recording_started = False
                self.enhanced_mode = False
                self.selected_model = None

        except Exception:
            logger.exception("Unhandled error in _on_release")
            
    def _check_hotkey_against_set(self, hotkey_name, keys_set):
        """Check if a specific hotkey is a subset of the provided keys set"""
        if hotkey_name not in self.hotkeys:
            return False
        
        required_keys = self.hotkeys[hotkey_name]["required_keys"]
        result = required_keys.issubset(keys_set)
        print(f"Checking hotkey {hotkey_name}: Required={sorted(required_keys)}, Current={sorted(keys_set)}, Match={result}")
        
        if result:
            print(f"Matched hotkey: {hotkey_name}")
        else:
            # Show which keys are missing
            missing_keys = required_keys - keys_set
            print(f"Missing keys: {missing_keys}")
        
        return result

    def _setup_hotkeys(self):
        """Initialize the hotkey definitions and their associated callbacks."""
        # Platform-specific hotkey definitions based on rules.md
        if sys.platform == 'win32':
            self.hotkeys = {
                # Simple mode
                "start_simple": {
                    "required_keys": set(["alt_l", "win"])
                },
                # Enhanced mode (Local Ollama default)
                "start_enhanced": {
                    "required_keys": set(["alt_l", "win", "ctrl_l"])
                },
                # Toggle mode  
                "toggle_mode": {
                    "required_keys": set(["alt_l", "win", "t"])
                },
                # Model-specific enhanced modes
                "start_openai": {
                    "required_keys": set(["alt_l", "win", "ctrl_l", "o"])
                },
                "start_gemini": {
                    "required_keys": set(["alt_l", "win", "ctrl_l", "g"])
                },
                "start_deepseek": {
                    "required_keys": set(["alt_l", "win", "ctrl_l", "d"])
                },
                "start_claude": {
                    "required_keys": set(["alt_l", "win", "ctrl_l", "c"])
                },
                # Action hotkeys
                "exit": {
                    "required_keys": set(["esc"])
                }
            }
        elif sys.platform == 'darwin':
            self.hotkeys = {
                # Simple mode
                "start_simple": {
                    "required_keys": set(["alt_l", "cmd"])
                },
                # Enhanced mode (Local Ollama default)
                "start_enhanced": {
                    "required_keys": set(["alt_l", "cmd", "ctrl_l"])
                },
                # Toggle mode
                "toggle_mode": {
                    "required_keys": set(["alt_l", "cmd", "t"])
                },
                # Model-specific enhanced modes
                "start_openai": {
                    "required_keys": set(["alt_l", "cmd", "ctrl_l", "o"])
                },
                "start_gemini": {
                    "required_keys": set(["alt_l", "cmd", "ctrl_l", "g"])
                },
                "start_deepseek": {
                    "required_keys": set(["alt_l", "cmd", "ctrl_l", "d"])
                },
                "start_claude": {
                    "required_keys": set(["alt_l", "cmd", "ctrl_l", "c"])
                },
                # Action hotkeys
                "exit": {
                    "required_keys": set(["esc"])
                }
            }
        elif sys.platform.startswith('linux'):
            self.hotkeys = {
                # Simple mode
                "start_simple": {
                    "required_keys": set(["alt_l", "super"])
                },
                # Enhanced mode (Local Ollama default)
                "start_enhanced": {
                    "required_keys": set(["alt_l", "super", "ctrl_l"])
                },
                # Toggle mode
                "toggle_mode": {
                    "required_keys": set(["alt_l", "super", "t"])
                },
                # Model-specific enhanced modes
                "start_openai": {
                    "required_keys": set(["alt_l", "super", "ctrl_l", "o"])
                },
                "start_gemini": {
                },
                "start_deepseek": {
                    "required_keys": set(["alt_l", "super", "ctrl_l", "d"])
                },
                "start_claude": {
                    "required_keys": set(["alt_l", "super", "ctrl_l", "c"])
                },
                # Action hotkeys
                "stop": {
                    "required_keys": set(["alt_l", "super", "ctrl_l", "s"])
                },
                "exit": {
                    "required_keys": set(["esc"])
                }
            }
        
        print("\nHotkey Configuration:")
        print("=====================")
        for name, config in self.hotkeys.items():
            print(f"{name}: {sorted(config['required_keys'])}")
        print("=====================\n")

    def _print_hotkeys(self):
        """Print the current hotkey configuration"""
        print("\nHotkey Configuration:")
        print("=====================")
        for name, config in self.hotkeys.items():
            print(f"{name}: {sorted(config['required_keys'])}")
        print("=====================\n")

    def start(self):
        """Start the keyboard listener"""
        if not self.listener:
            self._start_listener()
            
    def stop(self):
        """Stop the keyboard listener"""
        if self.listener:
            self.listener.stop()
            self.listener = None
            
    def restart(self):
        """Restart the keyboard listener"""
        self.stop()
        self._start_listener()

    def reset_states(self):
        """Reset all states after transcription enhancement begins"""
        self.recording_started = False
        self.enhanced_mode = False
        self.selected_model = None


# Module-level function 
def setup_pynput_keyboard(config, callbacks):
    handler = PynputKeyboardHandler(config)
    handler.callbacks = callbacks
    handler.start()
    return handler 