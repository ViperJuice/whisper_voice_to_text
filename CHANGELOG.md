# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2023-09-15

### Added
- PowerShell integration for Windows users
- GPU acceleration detection and utilization
- Automatic FFmpeg detection with installation suggestions
- Multiple installation and execution methods
- Better handling of API key fallbacks

### Changed
- Updated installation scripts to be more resilient
- Improved error handling with informative messages
- Enhanced compatibility with Python 3.8-3.11
- Reorganized project structure for better maintainability
- Simplified Windows execution with multiple options

### Fixed
- Issues with dependency installation when already present
- Key detection problems on different platforms
- Error handling for missing dependencies
- Temporary file management

## [0.1.0] - 2023-08-19

### Added
- Initial release
- Basic voice-to-text functionality with Whisper
- Enhanced mode with LLM support (OpenAI, Claude, Google, DeepSeek, Ollama)
- Model switching during enhanced mode recording
- Automatic file cleanup
- Cross-platform hotkey support
- Automatic transcription typing
- Modular project structure 