#!/bin/bash

# POLARIS SWIM System Setup Script
# This script sets up the complete POLARIS framework for SWIM system adaptation
# with agentic LLM reasoning, threshold-based control, and statistical world model

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 1  # Port is in use
    else
        return 0  # Port is available
    fi
}

print_status "Starting POLARIS SWIM System Setup..."

# Check prerequisites
print_status "Checking prerequisites..."

# Check Python version
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python 3 found: $PYTHON_VERSION"
else
    print_error "Python 3 is required but not found. Please install Python 3.8+"
    exit 1
fi

# Check if we're in the correct directory
if [ ! -f "start_polaris_framework.py" ]; then
    print_error "Please run this script from the polaris_refactored directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv .venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Dependencies installed from requirements.txt"
else
    print_warning "requirements.txt not found, installing basic dependencies..."
    pip install pyyaml asyncio nats-py google-generativeai openai anthropic
fi

# Create necessary directories
print_status "Creating directory structure..."
mkdir -p logs
mkdir -p config
mkdir -p plugins/swim
mkdir -p data/storage
mkdir -p data/cache
print_success "Directory structure created"

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_warning "Created .env file from .env.example. Please update with your API keys!"
    else
        print_warning ".env.example not found. Please create .env file manually."
    fi
else
    print_status ".env file already exists"
fi

# Check for NATS server
print_status "Checking NATS server..."
if command_exists nats-server; then
    print_success "NATS server found"
    
    # Check if NATS is running
    if check_port 4222; then
        print_status "Starting NATS server..."
        nats-server --port 4222 --jetstream &
        NATS_PID=$!
        sleep 2
        
        if check_port 4222; then
            print_error "Failed to start NATS server on port 4222"
            exit 1
        else
            print_success "NATS server started (PID: $NATS_PID)"
            echo $NATS_PID > .nats_pid
        fi
    else
        print_status "NATS server already running on port 4222"
    fi
else
    print_warning "NATS server not found. Installing..."
    
    # Download and install NATS server
    NATS_VERSION="v2.10.7"
    NATS_ARCH="linux-amd64"
    
    if [ ! -d "bin" ]; then
        mkdir bin
    fi
    
    cd bin
    if [ ! -f "nats-server" ]; then
        print_status "Downloading NATS server $NATS_VERSION..."
        wget -q "https://github.com/nats-io/nats-server/releases/download/$NATS_VERSION/nats-server-$NATS_VERSION-$NATS_ARCH.tar.gz"
        tar -xzf "nats-server-$NATS_VERSION-$NATS_ARCH.tar.gz"
        mv "nats-server-$NATS_VERSION-$NATS_ARCH/nats-server" .
        rm -rf "nats-server-$NATS_VERSION-$NATS_ARCH"*
        chmod +x nats-server
        print_success "NATS server downloaded and installed"
    fi
    
    cd ..
    
    # Start NATS server
    print_status "Starting NATS server..."
    ./bin/nats-server --port 4222 --jetstream &
    NATS_PID=$!
    sleep 2
    
    if check_port 4222; then
        print_error "Failed to start NATS server on port 4222"
        exit 1
    else
        print_success "NATS server started (PID: $NATS_PID)"
        echo $NATS_PID > .nats_pid
    fi
fi

# Validate configuration file
print_status "Validating configuration..."
if [ -f "config/swim_system_config_optimized.yaml" ]; then
    python3 -c "
import yaml
try:
    with open('config/swim_system_config_optimized.yaml', 'r') as f:
        yaml.safe_load(f)
    print('Configuration file is valid')
except Exception as e:
    print(f'Configuration validation failed: {e}')
    exit(1)
"
    print_success "Configuration file validated"
else
    print_error "Configuration file not found: config/swim_system_config_optimized.yaml"
    exit 1
fi

# Check environment variables
print_status "Checking environment variables..."
source .env 2>/dev/null || true

if [ -z "$GOOGLE_AI_API_KEY" ] || [ "$GOOGLE_AI_API_KEY" = "your_gemini_api_key_here" ]; then
    print_warning "GOOGLE_AI_API_KEY not set. LLM-based reasoning will not work!"
    print_warning "Please set your Gemini API key in the .env file"
fi

# Create startup script
print_status "Creating startup script..."
cat > start_swim_system.sh << 'EOF'
#!/bin/bash

# POLARIS SWIM System Startup Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Load environment variables
if [ -f ".env" ]; then
    source .env
    print_status "Environment variables loaded"
else
    print_error ".env file not found. Please run setup_swim_system.sh first"
    exit 1
fi

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    print_status "Virtual environment activated"
else
    print_error "Virtual environment not found. Please run setup_swim_system.sh first"
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
        print_success "NATS server started"
    else
        print_error "NATS server not found. Please run setup_swim_system.sh first"
        exit 1
    fi
else
    print_status "NATS server already running"
fi

# Start POLARIS framework
print_status "Starting POLARIS SWIM System..."
print_status "Configuration: config/swim_system_config_optimized.yaml"
print_status "Log Level: ${LOGGER_LEVEL:-DEBUG}"
print_status "Environment: ${POLARIS_ENVIRONMENT:-development}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the framework
python3 start_polaris_framework.py start \
    --config config/swim_system_config_optimized.yaml \
    --log-level ${LOGGER_LEVEL:-DEBUG} \
    --shell

EOF

chmod +x start_swim_system.sh
print_success "Startup script created: start_swim_system.sh"

# Create shutdown script
print_status "Creating shutdown script..."
cat > stop_swim_system.sh << 'EOF'
#!/bin/bash

# POLARIS SWIM System Shutdown Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_status "Stopping POLARIS SWIM System..."

# Stop NATS server if we started it
if [ -f ".nats_pid" ]; then
    NATS_PID=$(cat .nats_pid)
    if kill -0 $NATS_PID 2>/dev/null; then
        print_status "Stopping NATS server (PID: $NATS_PID)..."
        kill $NATS_PID
        rm .nats_pid
        print_success "NATS server stopped"
    else
        print_status "NATS server not running"
        rm .nats_pid
    fi
fi

# Kill any remaining POLARIS processes
pkill -f "start_polaris_framework.py" 2>/dev/null || true
pkill -f "nats-server" 2>/dev/null || true

print_success "POLARIS SWIM System stopped"

EOF

chmod +x stop_swim_system.sh
print_success "Shutdown script created: stop_swim_system.sh"

# Create monitoring script
print_status "Creating monitoring script..."
cat > monitor_swim_system.sh << 'EOF'
#!/bin/bash

# POLARIS SWIM System Monitoring Script

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  POLARIS SWIM System Status${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_section() {
    echo -e "${YELLOW}$1${NC}"
    echo "--------------------------------"
}

print_header

# Check NATS server
print_section "NATS Server Status"
if lsof -Pi :4222 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} NATS server running on port 4222"
else
    echo -e "${RED}✗${NC} NATS server not running"
fi

# Check log files
print_section "Recent Logs"
if [ -f "logs/polaris-swim.log" ]; then
    echo "Last 10 log entries:"
    tail -n 10 logs/polaris-swim.log | while read line; do
        echo "  $line"
    done
else
    echo "No log file found"
fi

# Check processes
print_section "Running Processes"
ps aux | grep -E "(polaris|nats)" | grep -v grep || echo "No POLARIS processes found"

# Check disk space
print_section "Disk Usage"
df -h . | tail -1 | awk '{print "Available space: " $4 " (" $5 " used)"}'

# Check memory usage
print_section "Memory Usage"
free -h | grep "Mem:" | awk '{print "Memory: " $3 " used / " $2 " total"}'

echo ""
echo "To view live logs: tail -f logs/polaris-swim.log"
echo "To stop system: ./stop_swim_system.sh"

EOF

chmod +x monitor_swim_system.sh
print_success "Monitoring script created: monitor_swim_system.sh"

# Final setup summary
print_success "POLARIS SWIM System setup completed!"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Update your .env file with the required API keys:"
echo "   - GOOGLE_AI_API_KEY (for LLM-based reasoning)"
echo ""
echo "2. Start the system:"
echo "   ./start_swim_system.sh"
echo ""
echo "3. Monitor the system:"
echo "   ./monitor_swim_system.sh"
echo ""
echo "4. Stop the system:"
echo "   ./stop_swim_system.sh"
echo ""
echo -e "${YELLOW}Important Configuration Files:${NC}"
echo "- .env: Environment variables and API keys"
echo "- config/swim_system_config_optimized.yaml: Main system configuration"
echo "- logs/polaris-swim.log: System logs"
echo ""
echo -e "${YELLOW}System Features Enabled:${NC}"
echo "✓ Agentic LLM Reasoning (Gemini-based)"
echo "✓ Threshold-based Reactive Control"
echo "✓ Statistical World Model"
echo "✓ Comprehensive Logging & Observability"
echo "✓ Meta-learner for Autonomous Optimization"
echo ""
print_success "Setup complete! Ready to run POLARIS SWIM system."