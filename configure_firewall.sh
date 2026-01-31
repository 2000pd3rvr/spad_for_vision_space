#!/bin/bash
# macOS Firewall Configuration Script
# Run with: sudo ./configure_firewall.sh

echo "🛡️  Configuring macOS Firewall..."
echo "===================================="

# Enable firewall
echo "Enabling firewall..."
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on

# Block all incoming connections by default
echo "Setting block all incoming connections..."
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setblockall on

# Allow Python (Flask)
echo "Allowing Python for Flask..."
PYTHON_PATH=$(which python3)
if [ -n "$PYTHON_PATH" ]; then
    sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add "$PYTHON_PATH"
    sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp "$PYTHON_PATH"
    echo "✅ Python allowed: $PYTHON_PATH"
else
    echo "⚠️  Python3 not found in PATH"
fi

# Allow cloudflared
echo "Allowing cloudflared..."
CLOUDFLARED_PATH=$(which cloudflared)
if [ -n "$CLOUDFLARED_PATH" ]; then
    sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add "$CLOUDFLARED_PATH"
    sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp "$CLOUDFLARED_PATH"
    echo "✅ cloudflared allowed: $CLOUDFLARED_PATH"
else
    echo "⚠️  cloudflared not found (install with: brew install cloudflared)"
fi

# Verify firewall status
echo ""
echo "Firewall Status:"
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getblockall

echo ""
echo "✅ Firewall configuration complete!"
echo ""
echo "Note: You can also configure via System Preferences:"
echo "System Preferences > Security & Privacy > Firewall"

