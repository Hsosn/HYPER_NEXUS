#!/bin/bash
# =============================================================================
# Virtual Computer Setup Script
# Runs inside the Docker container to install and configure the desktop environment.
# =============================================================================
set -e

export DEBIAN_FRONTEND=noninteractive

echo "[setup-vm] Updating package lists..."
apt-get update -qq

echo "[setup-vm] Installing core packages..."
apt-get install -y --no-install-recommends \
    xvfb \
    x11vnc \
    xdotool \
    imagemagick \
    wget \
    curl \
    git \
    vim \
    nano \
    python3 \
    python3-pip \
    fonts-liberation \
    fonts-dejavu-core \
    dbus-x11 \
    at-spi2-core \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libxshmfence1 \
    fluxbox \
    2>/dev/null || true

echo "[setup-vm] Installing Firefox (optional)..."
apt-get install -y --no-install-recommends firefox 2>/dev/null || true

echo "[setup-vm] Cleaning up apt cache..."
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "[setup-vm] Starting Xvfb (virtual display :99, 1280x720x24)..."
Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset &>/dev/null &
XVFB_PID=$!
sleep 1

# Verify Xvfb started
if ! kill -0 "$XVFB_PID" 2>/dev/null; then
    echo "[setup-vm] ERROR: Xvfb failed to start!"
    exit 1
fi
echo "[setup-vm] Xvfb started (PID: $XVFB_PID)"

export DISPLAY=:99

echo "[setup-vm] Starting x11vnc (port 5900, no password)..."
x11vnc -display :99 -rfbport 5900 -forever -nopw -shared -bg -o /tmp/x11vnc.log 2>/dev/null
sleep 1

echo "[setup-vm] Starting Fluxbox window manager..."
fluxbox -display :99 &
sleep 2

echo "[setup-vm] Configuring locale and environment..."
export LANG=en_US.UTF-8
export LANGUAGE=en_US:en
echo "LANG=en_US.UTF-8" > /etc/default/locale
echo "LANGUAGE=en_US:en" >> /etc/default/locale
echo "DISPLAY=:99" >> /etc/environment

echo "[setup-vm] Creating workspace directory..."
mkdir -p /workspace
chmod 777 /workspace

echo ""
echo "============================================="
echo "  Virtual computer ready."
echo "  Display: :99 (1280x720x24)"
echo "  VNC: port 5900 (no password)"
echo "  Workspace: /workspace"
echo "============================================="
