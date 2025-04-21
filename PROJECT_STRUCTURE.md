# Project Structure

This document outlines the structure and organization of the Voice to Text with LLM Enhancement project.

## Directory Structure

```
whisper_voice_to_text/
├── voicetotext/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── app.py        # Main application logic
│   │   ├── audio.py      # Audio recording and processing
│   │   └── transcription.py  # Transcription handling
│   ├── models/
│   │   ├── __init__.py
│   │   └── llm.py        # LLM integration (OpenAI, Claude, etc.)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py     # Configuration handling
│   │   └── pynput_keyboard.py  # Keyboard hotkey handling with pynput
│   └── controllers/
│       ├── __init__.py
│       └── recorder.py    # Recording controller
├── tests/
│   └── test_*.py         # Test files
├── docs/
│   └── *.md              # Documentation files
├── .env                  # Environment variables
├── .gitignore           # Git ignore file
├── LICENSE              # License file
├── README.md            # Project readme
├── pyproject.toml       # Project configuration
├── requirements.txt     # Python dependencies
├── setup.py            # Package setup file
└── main.py             # Entry point
```

## Component Overview

### Core Components

- **app.py** - Main application class and logic
- **audio.py** - Audio recording and processing utilities
- **transcription.py** - Handles transcription using Whisper or OpenAI API

### Models

- **llm.py** - LLM integration for text enhancement

### Utils

- **config.py** - Configuration management
- **pynput_keyboard.py** - Keyboard hotkey handling using pynput

### Controllers

- **recorder.py** - Controls audio recording process

## Key Features

1. Real-time voice recording
2. Transcription using local Whisper or OpenAI API
3. LLM enhancement of transcribed text
4. Configurable hotkeys for control
5. Support for multiple LLM providers

## Core Application

- **main.py** - Entry point for the application
- **system_checks.py** - System capability detection and configuration
- **pyproject.toml** - Project configuration and metadata
- **requirements.txt** - Python dependencies
- **setup.py** - Installation script for pip

## Installation & Running

### Windows
- **install-windows.bat** - Windows installation script for command prompt
- **run-windows-simple.bat** - Simple Windows launcher for command prompt
- **run-windows.bat** - Enhanced Windows launcher with error handling
- **run-windows.ps1** - PowerShell launcher with advanced features (recommended)

### Linux
- **install-linux.sh** - Linux installation script
- **run-linux.sh** - Linux launcher

### macOS
- **install-mac.sh** - macOS installation script
- **run-mac.sh** - macOS launcher

## Source Code

### Package Structure
- **voicetotext/** - Main package
  - **__init__.py** - Package initialization
  - **__main__.py** - Package entry point
  - **core/** - Core functionality
    - **app.py** - Main application class
    - **audio.py** - Audio recording and processing
    - **transcription.py** - Whisper transcription
  - **models/** - LLM models
    - **llm.py** - LLM enhancement functions
  - **utils/** - Utility functions
    - **config.py** - Configuration management
    - **pynput_keyboard.py** - Keyboard handling with pynput

## Documentation

- **README.MD** - Main documentation and usage guide
- **CHANGELOG.md** - Version history and changes
- **LICENSE** - MIT license
- **CONTRIBUTING.md** - Guide for contributors
- **PROJECT_STRUCTURE.md** - This file

## Other Files

- **.gitignore** - Git ignore configuration
- **.editorconfig** - Editor configuration for consistent code style
- **ollama-installer.exe** - Installer for Ollama (Windows) 