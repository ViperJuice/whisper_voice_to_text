#!/usr/bin/env python3
"""
Setup script for voicetotext package.
"""

from setuptools import setup, find_packages

with open("README.MD", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="whisper_voice_to_text",
    version="0.1.0",
    author="Jennifer",
    author_email="jennifer@example.com",
    description="Voice to text application with LLM enhancement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/whisper_voice_to_text",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.3",
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "openai-whisper>=20231117",
        "sounddevice>=0.4.6",
        "scipy>=1.11.3",
        "pynput>=1.7.6",
        "pyperclip>=1.8.2",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "anthropic>=0.15.0",
        "google-generativeai>=0.3.2",
        "openai>=1.12.0",
        "deepseek-ai>=1.1.0",
    ],
    extras_require={
        "dev": [
            "black>=23.1.0",
            "pytest>=7.3.1",
            "mypy>=1.2.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "voicetotext=voicetotext.__main__:main",
        ],
    },
) 