"""
Voice to Text with LLM Enhancement
A versatile voice transcription tool with optional LLM enhancement
"""

__version__ = "0.1.0"

from importlib import import_module

# Ensure utilities can be used as a library without running app.py
try:
    from voicetotext.utils import logging_config as _lc  # type: ignore
    _lc.init_logging(debug=False, redirect_print=False)
except Exception:  # noqa: S110
    # Fallback silently – logging will be configured by the application entry‑point.
    pass 