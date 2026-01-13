#!/usr/bin/env python3
"""
POLARIS SWIM System Setup Script

This script sets up the complete POLARIS framework for SWIM system adaptation
with agentic LLM reasoning, threshold-based control, and statistical world model.
"""

import os
import sys
import subprocess
import shutil
import urllib.request
import tarfile
import json
from pathlib import Path
from typing import Dict, Any

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_status(message: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")

def print_success(message: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")

def print_warning(message: str):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")

def print_error(message: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")

def command_exists(command: str) -> bool:
    """Check if a command exists in PATH."""
    return shutil.which(command) is not None

def check_port(port: int) -> bool:
    """Check if a port is available."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

def run_command(command: str, cwd: Path = None) -> bool:
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            capture_output=True, 
            text=True
        )
        if result.returncode != 0:
            print_error(f"Command failed: {command}")
            print_error(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print_error(f"Failed to run command '{command}': {e}")
        return False

def setup_virtual_environment() -> bool:
    """Set up Python virtual environment."""
    print_status("Setting up Python virtual environment...")
    
    venv_path = Path(".venv")
    if venv_path.exists():
        print_status("Virtual environment already exists")
        return True
    
    # Create virtual environment
    if not run_command("python3 -m venv .venv"):
        print_error("Failed to create virtual environment")
        return False
    
    print_success("Virtual environment created")
    return True

def install_dependencies() -> bool:
    """Install Python dependencies."""
    print_status("Installing Python dependencies...")
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        pip_cmd = ".venv\\Scripts\\pip"
    else:  # Unix-like
        pip_cmd = ".venv/bin/pip"
    
    # Upgrade pip first
    if not run_command(f"{pip_cmd} install --upgrade pip"):
        print_error("Failed to upgrade pip")
        return False
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt"):
        print_error("Failed to install dependencies")
        return False
    
    print_success("Dependencies installed successfully")
    return True

def setup_directories() -> bool:
    """Create necessary directories."""
    print_status("Creating directory structure...")
    
    directories = [
        "logs",
        "data/storage",
        "data/cache",
        "plugins/swim",
        "bin"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print_success("Directory structure created")
    return True

def setup_environment_file() -> bool:
    """Set up environment file."""
    print_status("Setting up environment file...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print_status(".env file already exists")
        return True
    
    if env_example.exists():
        shutil.copy(env_example, env_file)
        print_warning("Created .env file from .env.example")
        print_warning("Please update .env with your actual API keys!")
    else:
        # Create basic .env file
        env_content = """# POLARIS Framework Environment Variables

# LLM Configuration (Required for Agentic Reasoning)
GOOGLE_AI_API_KEY=your_gemini_api_key_here

# NATS Message Bus Configuration
NATS_URL=nats://localhost:4222

# Logging Configuration
LOGGER_LEVEL=DEBUG
LOGGER_FORMAT=json
LOGGER_OUTPUT=both

# Framework Configuration
POLARIS_SERVICE_NAME=polaris-swim-system
POLARIS_ENVIRONMENT=development
"""
        env_file.write_text(env_content)
        print_warning("Created basic .env file")
        print_warning("Please update with your actual API keys!")
    
    return True

def download_nats_server() -> bool:
    """Download and install NATS server."""
    print_status("Setting up NATS server...")
    
    nats_binary = Path("bin/nats-server")
    if nats_binary.exists():
        print_status("NATS server already installed")
        return True
    
    # Determine platform
    import platform
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "linux":
        if machine in ["x86_64", "amd64"]:
            arch = "linux-amd64"
        elif machine in ["aarch64", "arm64"]:
            arch = "linux-arm64"
        else:
            print_error(f"Unsupported architecture: {machine}")
            return False
    elif system == "darwin":  # macOS
        if machine in ["x86_64", "amd64"]:
            arch = "darwin-amd64"
        elif machine in ["arm64"]:
            arch = "darwin-arm64"
        else:
            print_error(f"Unsupported architecture: {machine}")
            return False
    else:
        print_error(f"Unsupported operating system: {system}")
        return False
    
    nats_version = "v2.10.7"
    download_url = f"https://github.com/nats-io/nats-server/releases/download/{nats_version}/nats-server-{nats_version}-{arch}.tar.gz"
    
    print_status(f"Downloading NATS server {nats_version} for {arch}...")
    
    try:
        # Download
        tar_file = Path(f"bin/nats-server-{nats_version}-{arch}.tar.gz")
        urllib.request.urlretrieve(download_url, tar_file)
        
        # Extract
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(path="bin")
        
        # Move binary
        extracted_dir = Path(f"bin/nats-server-{nats_version}-{arch}")
        shutil.move(extracted_dir / "nats-server", nats_binary)
        
        # Make executable
        nats_binary.chmod(0o755)
        
        # Clean up
        tar_file.unlink()
        shutil.rmtree(extracted_dir)
        
        print_success("NATS server installed successfully")
        return True
        
    except Exception as e:
        print_error(f"Failed to download NATS server: {e}")
        return False

def validate_configuration() -> bool:
    """Validate configuration files."""
    print_status("Validating configuration...")
    
    config_file = Path("config/swim_system_config_optimized.yaml")
    if not config_file.exists():
        print_error(f"Configuration file not found: {config_file}")
        return False
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Basic validation
        required_sections = ["framework", "llm", "control_reasoning", "digital_twin"]
        for section in required_sections:
            if section not in config:
                print_error(f"Missing configuration section: {section}")
                return False
        
        print_success("Configuration file validated")
        return True
        
    except Exception as e:
        print_error(f"Configuration validation failed: {e}")
        return False

def test_dependencies() -> bool:
    """Test that all dependencies are working."""
    print_status("Testing dependencies...")
    
    # Test Python imports
    test_script = """
import sys
sys.path.insert(0, 'src')

try:
    import nats
    import yaml
    import google.generativeai
    import aiohttp
    import pydantic
    print("✓ All Python dependencies imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
"""
    
    if os.name == 'nt':  # Windows
        python_cmd = ".venv\\Scripts\\python"
    else:  # Unix-like
        python_cmd = ".venv/bin/python"
    
    try:
        result = subprocess.run(
            [python_cmd, "-c", test_script],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print_success("All dependencies are working")
            return True
        else:
            print_error(f"Dependency test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print_error(f"Failed to test dependencies: {e}")
        return False

def create_startup_scripts() -> bool:
    """Create startup and utility scripts."""
    print_status("Creating startup scripts...")
    
    # Create start script
    start_script = Path("start_swim_system.sh")
    start_content = """#!/bin/bash

# POLARIS SWIM System Startup Script
set -e

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Load environment variables
if [ -f ".env" ]; then
    source .env
    print_status "Environment variables loaded"
else
    print_error ".env file not found. Please run: python3 setup_swim_system.py"
    exit 1
fi

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    print_status "Virtual environment activated"
else
    print_error "Virtual environment not found. Please run: python3 setup_swim_system.py"
    exit 1
fi

# Check if NATS is running
if ! lsof -Pi :4222 -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_status "Starting NATS server..."
    if [ -f "./bin/nats-server" ]; then
        ./bin/nats-server --port 4222 --jetstream &
        NATS_PID=$!
        echo $NATS_PID > .nats_pid
        sleep 2
        print_success "NATS server started (PID: $NATS_PID)"
    else
        print_error "NATS server not found. Please run: python3 setup_swim_system.py"
        exit 1
    fi
else
    print_status "NATS server already running"
fi

# Validate environment
if [ -z "$GOOGLE_AI_API_KEY" ] || [ "$GOOGLE_AI_API_KEY" = "your_gemini_api_key_here" ]; then
    print_warning "GOOGLE_AI_API_KEY not set properly!"
    print_warning "LLM-based reasoning will not work. Please update your .env file."
fi

# Start POLARIS framework
print_status "Starting POLARIS SWIM System..."
print_status "Configuration: config/swim_system_config_optimized.yaml"
print_status "Log Level: ${LOGGER_LEVEL:-DEBUG}"

python3 start_polaris_framework.py start \\
    --config config/swim_system_config_optimized.yaml \\
    --log-level ${LOGGER_LEVEL:-DEBUG} \\
    --shell
"""
    
    start_script.write_text(start_content)
    start_script.chmod(0o755)
    
    # Create stop script
    stop_script = Path("stop_swim_system.sh")
    stop_content = """#!/bin/bash

# POLARIS SWIM System Shutdown Script
set -e

echo "Stopping POLARIS SWIM System..."

# Stop NATS server if we started it
if [ -f ".nats_pid" ]; then
    NATS_PID=$(cat .nats_pid)
    if kill -0 $NATS_PID 2>/dev/null; then
        echo "Stopping NATS server (PID: $NATS_PID)..."
        kill $NATS_PID
        rm .nats_pid
        echo "✓ NATS server stopped"
    else
        echo "NATS server not running"
        rm .nats_pid
    fi
fi

# Kill any remaining POLARIS processes
pkill -f "start_polaris_framework.py" 2>/dev/null || true

echo "✓ POLARIS SWIM System stopped"
"""
    
    stop_script.write_text(stop_content)
    stop_script.chmod(0o755)
    
    print_success("Startup scripts created")
    return True

def main():
    """Main setup function."""
    print_status("Starting POLARIS SWIM System Setup...")
    
    # Check if we're in the correct directory
    if not Path("start_polaris_framework.py").exists():
        print_error("Please run this script from the polaris_refactored directory")
        return False
    
    # Check Python version
    if sys.version_info < (3, 8):
        print_error("Python 3.8+ is required")
        return False
    
    print_success(f"Python {sys.version.split()[0]} detected")
    
    # Setup steps
    steps = [
        ("Virtual Environment", setup_virtual_environment),
        ("Dependencies", install_dependencies),
        ("Directories", setup_directories),
        ("Environment File", setup_environment_file),
        ("NATS Server", download_nats_server),
        ("Configuration", validate_configuration),
        ("Dependencies Test", test_dependencies),
        ("Startup Scripts", create_startup_scripts),
    ]
    
    for step_name, step_func in steps:
        print_status(f"Step: {step_name}")
        if not step_func():
            print_error(f"Setup failed at step: {step_name}")
            return False
    
    # Final success message
    print_success("POLARIS SWIM System setup completed!")
    print("")
    print(f"{Colors.BLUE}Next Steps:{Colors.NC}")
    print("1. Update your .env file with the required API keys:")
    print("   - GOOGLE_AI_API_KEY (for LLM-based reasoning)")
    print("")
    print("2. Start the system:")
    print("   ./start_swim_system.sh")
    print("")
    print("3. Stop the system:")
    print("   ./stop_swim_system.sh")
    print("")
    print(f"{Colors.YELLOW}System Features Enabled:{Colors.NC}")
    print("✓ Agentic LLM Reasoning (Gemini-based)")
    print("✓ Threshold-based Reactive Control")
    print("✓ Statistical World Model")
    print("✓ Comprehensive Logging & Observability")
    print("✓ Meta-learner for Autonomous Optimization")
    print("")
    print(f"{Colors.GREEN}Setup complete! Ready to run POLARIS SWIM system.{Colors.NC}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_error("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)