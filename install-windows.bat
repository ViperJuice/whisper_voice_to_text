@echo off
echo Voice to Text with LLM Enhancement - Windows Installer
echo ==================================================

REM Set local installation paths
set "PYTHON_DIR=python"
set "VENV_DIR=.venv"

REM Check for local Python installation first
if exist "%PYTHON_DIR%\python.exe" (
    echo Local Python installation found.
    set "PATH=%CD%\%PYTHON_DIR%;%PATH%"
) else (
    REM Check for system Python installation
    where python >nul 2>&1
    if errorlevel 1 (
        echo Python not found. Installing Python 3.11 locally...
        
        REM Download Python embeddable package
        echo Downloading Python embeddable package...
        curl -L https://www.python.org/ftp/python/3.11.8/python-3.11.8-embed-amd64.zip -o python.zip
        
        REM Extract Python using PowerShell's built-in zip functionality
        echo Extracting Python...
        if not exist "%PYTHON_DIR%" mkdir "%PYTHON_DIR%"
        powershell -Command "$shell = New-Object -ComObject Shell.Application; $zip = $shell.NameSpace((Get-Item python.zip).FullName); $dest = $shell.NameSpace((Get-Item %PYTHON_DIR%).FullName); $dest.CopyHere($zip.Items())"
        
        REM Clean up
        del python.zip
        
        REM Add Python to local PATH
        set "PATH=%CD%\%PYTHON_DIR%;%PATH%"
        
        REM Verify Python installation
        "%PYTHON_DIR%\python.exe" --version >nul 2>&1
        if errorlevel 1 (
            echo Failed to install Python.
            echo Please install Python 3.11 manually from https://www.python.org/downloads/
            pause
            exit /b 1
        )
    ) else (
        echo System Python installation found.
    )
)

REM Check for pip and install if needed
echo Checking for pip...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo pip not found. Installing pip...
    curl -L https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py --no-warn-script-location
    if errorlevel 1 (
        echo Failed to install pip.
        pause
        exit /b 1
    )
    del get-pip.py
) else (
    echo pip is already installed.
)

REM Install UV if not already installed
echo Checking for UV...
uv --version >nul 2>&1
if errorlevel 1 (
    echo UV not found. Installing UV...
    python -m pip install --no-warn-script-location uv
    if errorlevel 1 (
        echo Failed to install UV.
        pause
        exit /b 1
    )
) else (
    echo UV is already installed.
)

REM Create virtual environment if it doesn't exist
if not exist "%VENV_DIR%" (
    echo Creating virtual environment with UV...
    uv venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

REM Install dependencies using UV
echo Installing dependencies...

REM Check if PyTorch is already installed
echo Checking PyTorch installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" >nul 2>&1
if errorlevel 1 (
    echo Installing PyTorch dependencies...
    uv pip install --no-deps torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo Failed to install PyTorch.
        pause
        exit /b 1
    )
) else (
    echo PyTorch is already installed.
)

REM Check core dependencies
echo Checking core dependencies...
python -c "import numpy, soundfile, pyaudio, pynput, requests, openai, dotenv, psutil, packaging" >nul 2>&1
if errorlevel 1 (
    echo Installing core dependencies...
    uv pip install numpy>=1.20.0 soundfile>=0.10.3 pyaudio>=0.2.13 pynput>=1.7.6 requests>=2.28.2 openai>=1.0.0 python-dotenv>=1.0.0 psutil>=5.9.0 packaging>=23.0
    if errorlevel 1 (
        echo Failed to install core dependencies.
        pause
        exit /b 1
    )
) else (
    echo Core dependencies are already installed.
)

REM Check LLM dependencies
echo Checking LLM dependencies...
python -c "import anthropic, google.generativeai, ollama, retry" >nul 2>&1
if errorlevel 1 (
    echo Installing LLM dependencies...
    uv pip install anthropic>=0.5.0 google-generativeai>=0.3.0 ollama>=0.1.0 retry>=0.9.2
    if errorlevel 1 (
        echo Failed to install LLM dependencies.
        pause
        exit /b 1
    )
) else (
    echo LLM dependencies are already installed.
)

REM Check Whisper
echo Checking Whisper installation...
python -c "import whisper" >nul 2>&1
if errorlevel 1 (
    echo Installing Whisper...
    uv pip install "openai-whisper @ git+https://github.com/openai/whisper.git"
    if errorlevel 1 (
        echo Failed to install Whisper.
        pause
        exit /b 1
    )
) else (
    echo Whisper is already installed.
)

REM Check development dependencies
echo Checking development dependencies...
python -c "import black, pytest, mypy" >nul 2>&1
if errorlevel 1 (
    echo Installing development dependencies...
    uv pip install black>=23.1.0 pytest>=7.3.1 mypy>=1.2.0
    if errorlevel 1 (
        echo Failed to install development dependencies.
        pause
        exit /b 1
    )
) else (
    echo Development dependencies are already installed.
)

REM Check for Ollama
echo Checking for Ollama...
curl -s -m 2 http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo Ollama service not detected.
    choice /C YN /M "Would you like to install Ollama"
    if errorlevel 2 goto SKIP_OLLAMA
    if errorlevel 1 (
        echo Installing Ollama...
        curl -L https://ollama.ai/download/ollama-installer.exe -o ollama-installer.exe
        if exist ollama-installer.exe (
            start /wait ollama-installer.exe /S
            if errorlevel 1 (
                echo Failed to install Ollama.
                echo Please install manually from https://ollama.ai/download
            ) else (
                echo Starting Ollama service...
                start /min ollama\ollama.exe serve
                timeout /t 5 >nul
                echo Checking Ollama service...
                curl -s -m 2 http://localhost:11434/api/tags >nul 2>&1
                if errorlevel 1 (
                    echo Warning: Ollama service not responding. You may need to start it manually.
                ) else (
                    echo Ollama service is running.
                )
            )
            del ollama-installer.exe
        ) else (
            echo Failed to download Ollama installer.
        )
    )
) else (
    echo Ollama service is running.
)

:SKIP_OLLAMA
echo.
echo Installation complete!
echo.
echo To use this installation:
echo 1. Always run the application using run-windows.ps1
echo 2. The script will automatically set up the environment
echo 3. Python and FFmpeg are installed locally in this directory
pause

REM -----------------------------------------------------------------------------
REM Create Desktop Shortcut (Keyless Voice-to-Text)
REM -----------------------------------------------------------------------------

echo Creating desktop shortcut...

powershell -NoLogo -NoProfile -Command ^
    "$WshShell = New-Object -ComObject WScript.Shell;" ^
    "$desktop = [Environment]::GetFolderPath('Desktop');" ^
    "$shortcut = $WshShell.CreateShortcut([IO.Path]::Combine($desktop, 'Keyless Voice-to-Text.lnk'));" ^
    "$shortcut.TargetPath = '%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\powershell.exe';" ^
    "$shortcut.Arguments = '-ExecutionPolicy Bypass -NoLogo -File \"%~dp0run-windows.ps1\"';" ^
    "$shortcut.IconLocation = '%~dp0resources\\icon.png';" ^
    "$shortcut.WorkingDirectory = '%~dp0';" ^
    "$shortcut.Save();"

echo Desktop shortcut created on %USERPROFILE%\Desktop. 