# Voice to Text with LLM Enhancement - Launcher
Write-Host "Voice to Text with LLM Enhancement - Launcher" -ForegroundColor Green
# Ensure console uses UTF-8 to avoid UnicodeEncodeError for emoji output
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null
Write-Host "=============================================" -ForegroundColor Green
Write-Host "(Set environment variable VTT_DEBUG=1 for verbose logging)" -ForegroundColor Yellow

# Set local installation paths
$PYTHON_DIR = "python"
$FFMPEG_DIR = "ffmpeg"
$VENV_DIR = ".venv"

# URL for latest static FFmpeg build
$FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"

# Check if main.py exists
if (-not (Test-Path main.py)) {
    Write-Host "ERROR: main.py not found. Please run this script from the project root directory." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if virtual environment exists
if (-not (Test-Path $VENV_DIR)) {
    Write-Host "Virtual environment not found. Running installation first..." -ForegroundColor Yellow
    & .\install-windows.bat
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installation failed. Please run install-windows.bat manually." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "$VENV_DIR\Scripts\activate"

# Check FFmpeg
if (-not (Test-Path "$FFMPEG_DIR\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe")) {
    Write-Host "FFmpeg not found. Installing FFmpeg..." -ForegroundColor Yellow
    
    # Create FFmpeg directory if it doesn't exist
    if (-not (Test-Path $FFMPEG_DIR)) {
        New-Item -ItemType Directory -Path $FFMPEG_DIR | Out-Null
    }
    
    # Download FFmpeg
    $ffmpegZip = "$FFMPEG_DIR\ffmpeg.zip"
    
    Write-Host "Downloading FFmpeg..."
    Invoke-WebRequest -Uri $FFMPEG_URL -OutFile $ffmpegZip
    
    # Extract FFmpeg
    Write-Host "Extracting FFmpeg..."
    Expand-Archive -Path $ffmpegZip -DestinationPath $FFMPEG_DIR -Force
    Remove-Item $ffmpegZip
    
    if (-not (Test-Path "$FFMPEG_DIR\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe")) {
        Write-Host "Failed to install FFmpeg. Please install it manually." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Add local FFmpeg to PATH
$env:PATH = "$PWD\$FFMPEG_DIR\ffmpeg-master-latest-win64-gpl\bin;$env:PATH"

# Check GPU capabilities and set up PyTorch
Write-Host "Checking GPU capabilities..." -ForegroundColor Yellow
$gpuCheckScript = @"
import torch
import sys

def check_gpu():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_info = []
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_info.append(f"GPU {i}: {gpu_name}")
        return True, gpu_info
    return False, []

has_gpu, gpu_info = check_gpu()
print(f"GPU_AVAILABLE:{has_gpu}")
if gpu_info:
    print("GPU_INFO:" + "|".join(gpu_info))
"@

$gpuCheckScript | Out-File -FilePath "check_gpu.py" -Encoding utf8
$gpuCheckResult = python check_gpu.py
Remove-Item "check_gpu.py"

$hasGPU = $false
$gpuInfo = @()

foreach ($line in $gpuCheckResult) {
    if ($line -match "GPU_AVAILABLE:(.+)") {
        $hasGPU = [bool]::Parse($matches[1])
    }
    elseif ($line -match "GPU_INFO:(.+)") {
        $gpuInfo = $matches[1] -split "\|"
    }
}

if ($hasGPU) {
    Write-Host "✓ GPU detected:" -ForegroundColor Green
    foreach ($info in $gpuInfo) {
        Write-Host "  $info" -ForegroundColor Green
    }
    
    # Check if PyTorch with CUDA is installed
    $torchCheckScript = @"
import torch
print(f"TORCH_VERSION:{torch.__version__}")
print(f"CUDA_VERSION:{torch.version.cuda if torch.cuda.is_available() else 'None'}")
"@
    
    $torchCheckScript | Out-File -FilePath "check_torch.py" -Encoding utf8
    $torchCheckResult = python check_torch.py
    Remove-Item "check_torch.py"
    
    $torchVersion = ""
    $cudaVersion = ""
    
    foreach ($line in $torchCheckResult) {
        if ($line -match "TORCH_VERSION:(.+)") {
            $torchVersion = $matches[1]
        }
        elseif ($line -match "CUDA_VERSION:(.+)") {
            $cudaVersion = $matches[1]
        }
    }
    
    if ($cudaVersion -eq "None") {
        Write-Host "PyTorch with CUDA support not found. Installing..." -ForegroundColor Yellow
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    }
    else {
        Write-Host "✓ PyTorch with CUDA support is installed:" -ForegroundColor Green
        Write-Host "  PyTorch version: $torchVersion" -ForegroundColor Green
        Write-Host "  CUDA version: $cudaVersion" -ForegroundColor Green
    }
}
else {
    Write-Host "No GPU detected. Using CPU-only mode." -ForegroundColor Yellow
    # Install CPU-only PyTorch if not already installed
    $torchCheckScript = @"
import torch
print(f"TORCH_VERSION:{torch.__version__}")
"@
    
    $torchCheckScript | Out-File -FilePath "check_torch.py" -Encoding utf8
    $torchCheckResult = python check_torch.py
    Remove-Item "check_torch.py"
    
    if (-not $torchCheckResult) {
        Write-Host "Installing CPU-only PyTorch..." -ForegroundColor Yellow
        pip install torch torchvision torchaudio
    }
}

# Check Ollama
$ollamaRunning = $false
$ollamaProcess = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if ($ollamaProcess) {
    Write-Host "Ollama process found (PID: $($ollamaProcess.Id))" -ForegroundColor Green
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/version" -TimeoutSec 5 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $ollamaRunning = $true
            $version = ($response.Content | ConvertFrom-Json).version
            Write-Host "✓ Ollama server running: version $version" -ForegroundColor Green
        }
    }
    catch {
        Write-Host "Ollama process exists but API is not responding" -ForegroundColor Yellow
    }
}

if (-not $ollamaRunning) {
    Write-Host "Ollama service not detected." -ForegroundColor Yellow
    $choice = Read-Host "Would you like to install Ollama? (Y/N)"
    if ($choice -eq "Y") {
        Write-Host "Installing Ollama..." -ForegroundColor Yellow
        # Download and run Ollama installer
        $ollamaInstaller = "ollama_installer.exe"
        Invoke-WebRequest -Uri "https://ollama.com/download/windows" -OutFile $ollamaInstaller
        Start-Process -FilePath $ollamaInstaller -Wait
        Remove-Item $ollamaInstaller
    }
}

# Run the application
Write-Host "Starting Voice to Text application..." -ForegroundColor Green
python main.py

# Keep the window open if there's an error
if ($LASTEXITCODE -ne 0) {
    Write-Host "Application exited with error code $LASTEXITCODE" -ForegroundColor Red
    Read-Host "Press Enter to exit"
} 