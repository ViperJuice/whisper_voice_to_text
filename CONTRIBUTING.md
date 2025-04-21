# Contributing to Voice to Text with LLM Enhancement

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## Project Structure

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
└── main.py               # Entry point
```

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Write docstrings for functions and classes
- Use black for code formatting

## Testing

- Write tests for new features
- Run tests with pytest:
  ```bash
  pytest tests/
  ```

## Pull Request Process

1. Create a feature branch
2. Make your changes
3. Run tests and ensure they pass
4. Update documentation if needed
5. Submit a pull request

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License. 