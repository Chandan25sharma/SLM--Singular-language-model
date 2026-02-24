#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  SLM Build Script (Linux/Mac)
#  Builds the C++ core engine as libslm.so / libslm.dylib
# ═══════════════════════════════════════════════════════════

echo ""
echo "  ╔═══════════════════════════════════════╗"
echo "  ║  Building SLM - C++ Core Engine       ║"
echo "  ╚═══════════════════════════════════════╝"
echo ""

# Create build directory
mkdir -p build

# Configure
echo "[1/3] Configuring with CMake..."
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
if [ $? -ne 0 ]; then
    echo ""
    echo "  ERROR: CMake configuration failed."
    echo "  Make sure CMake and a C++ compiler (g++ / clang++) are installed."
    echo ""
    echo "  You can still use SLM with the pure Python engine:"
    echo "    python3 slm_cli.py"
    echo ""
    exit 1
fi

# Build
echo "[2/3] Building..."
cmake --build build --config Release
if [ $? -ne 0 ]; then
    echo ""
    echo "  ERROR: Build failed."
    echo ""
    exit 1
fi

echo "[3/3] Done!"
echo ""
echo "  ✓ SLM C++ engine built successfully"
echo ""
echo "  Run SLM:"
echo "    python3 slm_cli.py"
echo ""
