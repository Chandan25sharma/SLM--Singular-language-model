@echo off
REM ═══════════════════════════════════════════════════════════
REM  SLM Build Script (Windows)
REM  Builds the C++ core engine as slm.dll
REM ═══════════════════════════════════════════════════════════

echo.
echo  ╔═══════════════════════════════════════╗
echo  ║  Building SLM - C++ Core Engine       ║
echo  ╚═══════════════════════════════════════╝
echo.

REM Create build directory
if not exist build mkdir build

REM Configure with CMake
echo [1/3] Configuring with CMake...
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  ERROR: CMake configuration failed.
    echo  Make sure CMake and a C++ compiler are installed.
    echo.
    echo  You can still use SLM with the pure Python engine:
    echo    python slm_cli.py
    echo.
    exit /b 1
)

REM Build
echo [2/3] Building...
cmake --build build --config Release
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  ERROR: Build failed.
    echo.
    exit /b 1
)

echo [3/3] Done!
echo.
echo  ✓ SLM C++ engine built successfully
echo  ✓ Library: python\slm\slm.dll
echo.
echo  Run SLM:
echo    python slm_cli.py
echo.
