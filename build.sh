#!/bin/bash

# Build script for loom-scene-gen
# Builds for all major platforms and architectures

set -e

echo "Building loom-scene-gen for all platforms..."

# Create build directory
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"

# Get version from git or use default
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_TIME=$(date -u '+%Y-%m-%d_%H:%M:%S')

# Linux builds
echo "Building for Linux..."
GOOS=linux GOARCH=amd64 go build -o "$BUILD_DIR/loom-scene-gen-linux-amd64" main.go
GOOS=linux GOARCH=arm64 go build -o "$BUILD_DIR/loom-scene-gen-linux-arm64" main.go
GOOS=linux GOARCH=386 go build -o "$BUILD_DIR/loom-scene-gen-linux-386" main.go

# Windows builds
echo "Building for Windows..."
GOOS=windows GOARCH=amd64 go build -o "$BUILD_DIR/loom-scene-gen-windows-amd64.exe" main.go
GOOS=windows GOARCH=arm64 go build -o "$BUILD_DIR/loom-scene-gen-windows-arm64.exe" main.go
GOOS=windows GOARCH=386 go build -o "$BUILD_DIR/loom-scene-gen-windows-386.exe" main.go

# macOS builds
echo "Building for macOS..."
GOOS=darwin GOARCH=amd64 go build -o "$BUILD_DIR/loom-scene-gen-darwin-amd64" main.go
GOOS=darwin GOARCH=arm64 go build -o "$BUILD_DIR/loom-scene-gen-darwin-arm64" main.go

# FreeBSD builds
echo "Building for FreeBSD..."
GOOS=freebsd GOARCH=amd64 go build -o "$BUILD_DIR/loom-scene-gen-freebsd-amd64" main.go
GOOS=freebsd GOARCH=arm64 go build -o "$BUILD_DIR/loom-scene-gen-freebsd-arm64" main.go

# Copy necessary files to build directory
echo "Copying necessary files..."
cp -r scenes "$BUILD_DIR/" 2>/dev/null || true
cp network_config.json "$BUILD_DIR/" 2>/dev/null || true
cp saved_model.json "$BUILD_DIR/" 2>/dev/null || true
cp README.md "$BUILD_DIR/" 2>/dev/null || true

echo ""
echo "Build complete! Binaries are located in: $BUILD_DIR/"
echo ""
echo "Linux:"
echo "  - loom-scene-gen-linux-amd64"
echo "  - loom-scene-gen-linux-arm64"
echo "  - loom-scene-gen-linux-386"
echo ""
echo "Windows:"
echo "  - loom-scene-gen-windows-amd64.exe"
echo "  - loom-scene-gen-windows-arm64.exe"
echo "  - loom-scene-gen-windows-386.exe"
echo ""
echo "macOS:"
echo "  - loom-scene-gen-darwin-amd64 (Intel)"
echo "  - loom-scene-gen-darwin-arm64 (Apple Silicon)"
echo ""
echo "FreeBSD:"
echo "  - loom-scene-gen-freebsd-amd64"
echo "  - loom-scene-gen-freebsd-arm64"
