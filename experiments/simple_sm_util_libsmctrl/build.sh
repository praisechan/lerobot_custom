#!/bin/bash
# Quick build script for sm_bw_sweep

set -e  # Exit on error

# Create build directory
mkdir -p build
cd build

# Configure and build
echo "Configuring with CMake..."
cmake ..

echo "Building..."
cmake --build . -j$(nproc)

echo ""
echo "Build complete! Executable: ./build/sm_bw_sweep"
echo ""
echo "Quick test:"
echo "  cd build && ./sm_bw_sweep --min_sms 1 --max_sms 4 --repeats 2"
