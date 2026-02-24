@echo off
setlocal enabledelayedexpansion

REM ═══════════════════════════════════════════════════════════
REM  SLM Build Script (Windows)
REM  Builds the C++ core engine as slm.dll
REM ═══════════════════════════════════════════════════════════

echo.
echo  ╔═══════════════════════════════════════╗
echo  ║  Building SLM - C++ Core Engine       ║
echo  ╚═══════════════════════════════════════╝
echo.

REM Try CMake first
where cmake >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [1/3] Configuring with CMake...
    if not exist build mkdir build
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
    if !ERRORLEVEL! EQU 0 (
        echo [2/3] Building with CMake...
        cmake --build build --config Release
        if !ERRORLEVEL! EQU 0 (
            echo [3/3] Done!
            goto :success
        )
    )
    echo.
    echo  WARNING: CMake build failed. Trying manual MSVC compilation...
) else (
    echo [!] CMake not found. Trying manual MSVC compilation...
)

REM Manual MSVC Compilation Fallback
echo [1/3] Locating Visual Studio Build Tools...

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if not exist "!VCVARS!" (
    set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
)
if not exist "!VCVARS!" (
    for /f "usebackq tokens=*" %%i in (`where vcvarsall.bat 2^>nul`) do set "VCVARS=%%i"
)

if not exist "!VCVARS!" (
    echo.
    echo  ERROR: Could not find vcvarsall.bat.
    echo  Please install Visual Studio Build Tools with C++ workload.
    goto :failure
)

echo [2/3] Setting up environment and compiling...
call "!VCVARS!" x64 >nul

cl.exe /LD /O2 /EHsc /std:c++17 /Icore\include /D_USRDLL /D_WINDLL ^
    core\src\tokenizer.cpp ^
    core\src\frequency_store.cpp ^
    core\src\ngram_model.cpp ^
    core\src\generator.cpp ^
    core\src\slm_api.cpp ^
    /Fe:python\slm\slm.dll

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  ERROR: Manual compilation failed.
    goto :failure
)

echo [3/3] Done!

:success
echo.
echo  ✓ SLM C++ engine built successfully
echo  ✓ Library: python\slm\slm.dll
echo.
echo  Run SLM:
echo    python slm_cli.py
echo.
exit /b 0

:failure
echo.
echo  You can still use SLM with the pure Python engine:
echo    python slm_cli.py
echo.
exit /b 1

