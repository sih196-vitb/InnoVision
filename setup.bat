@echo off
setlocal enabledelayedexpansion
title AI Video Analytics Platform - BOP Setup & Environment Config

echo ===============================================================================
echo   AI VIDEO ANALYTICS PLATFORM FOR BORDER OUT POSTS (BOPs) & CHECK POSTS
echo   Hardware Target: NVIDIA RTX 3050 Laptop GPU (4GB VRAM) / Intel i5 / 24GB RAM
echo ===============================================================================
echo.

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in system PATH.
    echo Please install Python 3.10 or 3.11 from python.org and re-run this setup.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] Detected Python %PY_VER%

:: 2. Check NVIDIA GPU & CUDA
echo.
echo [INFO] Checking NVIDIA GPU and driver capability...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] NVIDIA GPU detected via nvidia-smi.
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
) else (
    echo [WARNING] nvidia-smi not found. GPU acceleration might be disabled or unavailable.
)

:: 3. Setup Virtual Environment
echo.
if not exist "venv\" (
    echo [INFO] Creating Python virtual environment in .\venv...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully.
) else (
    echo [INFO] Existing virtual environment found in .\venv.
)

:: 4. Activate Virtual Environment
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated: %VIRTUAL_ENV%

:: 5. Upgrade pip and wheel
echo.
echo [INFO] Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel --quiet

:: 6. Install PyTorch with CUDA support (CUDA 12.1 or 11.8)
echo.
echo [INFO] Verifying / Installing PyTorch with CUDA acceleration...
python -c "import torch; print('PyTorch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available())" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing PyTorch with CUDA 12.1 support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    if %errorlevel% neq 0 (
        echo [WARNING] CUDA 12.1 wheel installation failed. Falling back to default PyTorch...
        pip install torch torchvision
    )
) else (
    echo [OK] PyTorch is already installed.
)

:: 7. Install Platform Requirements
echo.
echo [INFO] Installing platform dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARNING] Some dependencies had warnings or issues during install. Continuing verification...
)

:: 8. Create Essential Runtime Directories
echo.
echo [INFO] Initializing runtime directories...
if not exist "data\test_media" mkdir data\test_media
if not exist "data\watchlist" mkdir data\watchlist
if not exist "logs" mkdir logs
if not exist "models\weights" mkdir models\weights
echo [OK] Runtime directories prepared.

:: 9. Verify Redis Connectivity
echo.
echo ===============================================================================
echo   VERIFYING REDIS CONNECTIVITY (queue:faces, queue:plates, queue:alerts)
echo ===============================================================================
python -c "import redis; r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=2); r.ping(); print('[OK] Successfully connected to local Redis broker on port 6379.')" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Redis service is ONLINE and responding at localhost:6379.
) else (
    echo [NOTICE] Local Redis server not detected on port 6379.
    echo -----------------------------------------------------------------------------
    echo The platform includes an automatic, high-performance in-memory FIFO queue fallback,
    echo so the system will operate immediately without Redis installed.
    echo.
    echo For multi-process / distributed cluster deployments, you can start Redis via:
    echo   Option A (Docker):  docker run -d -p 6379:6379 --name bop-redis redis:alpine
    echo   Option B (Memurai): https://www.memurai.com/ (Native Redis for Windows)
    echo   Option C (WSL):     sudo service redis-server start
    echo -----------------------------------------------------------------------------
)

:: 10. Run System Hardware Diagnostic
echo.
echo [INFO] Running hardware & VRAM allocation diagnostic...
python -c "import torch; print('CUDA Devices:', torch.cuda.device_count()); [print(f'Device {i}: {torch.cuda.get_device_name(i)} | VRAM: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('CUDA not active, CPU execution enabled.')"

echo.
echo ===============================================================================
echo   SETUP COMPLETE!
echo   To launch the entire platform, run:
echo       python run_platform.py
echo ===============================================================================
echo.
pause
