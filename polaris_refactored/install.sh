#!/bin/bash
# POLARIS Framework - Installation Script
# This script sets up the POLARIS framework with all dependencies

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         POLARIS Framework Installation Script                 ║"
echo "║         Version 2.0.0                                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "📋 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $PYTHON_VERSION"

# Check if Python 3.8+
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "❌ Python 3.8+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi
echo "✅ Python version OK"
echo ""

# Create virtual environment
echo "🔧 Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "ℹ️  Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✅ pip upgraded"
echo ""

# Install core dependencies
echo "📥 Installing core dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Core dependencies installed"
else
    echo "❌ requirements.txt not found"
    exit 1
fi
echo ""

# Ask about development dependencies
echo "❓ Install development dependencies? (y/n)"
read -r INSTALL_DEV

if [ "$INSTALL_DEV" = "y" ] || [ "$INSTALL_DEV" = "Y" ]; then
    echo "📥 Installing development dependencies..."
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
        echo "✅ Development dependencies installed"
    else
        echo "❌ requirements-dev.txt not found"
    fi
    echo ""
fi

# Check for .env file
echo "🔐 Checking environment configuration..."
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found"
    echo "   Creating template .env file..."
    cat > .env << 'EOF'
# POLARIS Framework Environment Configuration
# Add your API keys and configuration here

# Google Gemini API Key (required for LLM features)
GEMINI_API_KEY="your-api-key-here"

# Optional: NATS Configuration
# NATS_SERVERS="nats://localhost:4222"
# NATS_USERNAME="user"
# NATS_PASSWORD="password"
EOF
    echo "✅ Template .env file created"
    echo "   ⚠️  Please update .env with your API keys"
else
    echo "✅ .env file found"
fi
echo ""

# Verify installation
echo "✔️  Verifying installation..."
echo ""

VERIFY_FAILED=0

# Check core packages
for package in nats yaml pydantic aiohttp; do
    if python3 -c "import $package" 2>/dev/null; then
        echo "   ✅ $package"
    else
        echo "   ❌ $package"
        VERIFY_FAILED=1
    fi
done

# Check google-generativeai
if python3 -c "import google.generativeai" 2>/dev/null; then
    echo "   ✅ google.generativeai"
else
    echo "   ⚠️  google.generativeai (optional)"
fi

# Check pytest if dev dependencies installed
if python3 -c "import pytest" 2>/dev/null; then
    echo "   ✅ pytest"
fi

echo ""

if [ $VERIFY_FAILED -eq 0 ]; then
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                  ✅ Installation Complete!                    ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📝 Next steps:"
    echo "   1. Update .env file with your GEMINI_API_KEY"
    echo "   2. Review configuration in config/gemini_agentic_config.yaml"
    echo "   3. Start the framework:"
    echo "      python start_polaris_framework.py start --config config/gemini_agentic_config.yaml"
    echo ""
    echo "📚 Documentation:"
    echo "   - README.md - Project overview"
    echo "   - TESTING_GUIDE.md - Testing instructions"
    echo ""
else
    echo "❌ Installation verification failed"
    echo "   Please check the error messages above and try again"
    exit 1
fi
