"""
Configuration management for the voice to text application.
"""

import os
import json

# Default configuration
DEFAULT_CONFIG = {
    "hotkeys": {
        "simple_mode": "windows+alt",
        "enhanced_mode": "windows+alt+ctrl",
        "exit": "escape",
    },
    "llm_models": {
        "o": "OPENAI",
        "c": "CLAUDE",
        "d": "DEEPSEEK",
        "g": "GEMINI"
    },
    "default_llm": "OLLAMA",
    "llm_fallback_order": ["OLLAMA", "OPENAI", "DEEPSEEK", "CLAUDE", "GEMINI"]
}

def get_config_dir():
    """Get the configuration directory path"""
    config_dir = os.path.join(os.path.expanduser("~"), ".voice_to_text")
    os.makedirs(config_dir, exist_ok=True)
    return config_dir

def get_config_file_path():
    """Get the configuration file path"""
    config_dir = get_config_dir()
    return os.path.join(config_dir, "config.json")

def load_config():
    """Load or create configuration file"""
    config_file = get_config_file_path()
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
    
    # Save and return default config if no file exists or error occurs
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG

def save_config(config):
    """Save configuration to file"""
    config_file = get_config_file_path()
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {config_file}")
    except Exception as e:
        print(f"Error saving config: {e}")

def update_config_with_system_checks(config, sys_config, llm_model):
    """Update configuration based on system checks"""
    # Update hotkeys based on OS
    if "hotkeys" in sys_config:
        config["hotkeys"] = sys_config["hotkeys"]
    
    # Set default LLM
    config["default_llm"] = llm_model
    
    # Ensure Ollama is the highest priority in fallback order
    if "llm_fallback_order" not in config:
        config["llm_fallback_order"] = DEFAULT_CONFIG["llm_fallback_order"]
    elif "OLLAMA" not in config["llm_fallback_order"]:
        config["llm_fallback_order"].insert(0, "OLLAMA")
        
    # Update model keys to match requirements
    if "a" in config["llm_models"]:
        # Rename keys to match requirements
        config["llm_models"] = {
            "o": "OPENAI",
            "c": "CLAUDE",
            "d": "DEEPSEEK",
            "g": "GOOGLE"
        }
    
    # Save updated config
    save_config(config)
    return config 