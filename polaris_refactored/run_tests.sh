#!/bin/bash
# POLARIS Test Runner Script for Linux/WSL
# This script provides convenient shortcuts for running different test suites

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Change to the polaris_refactored directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "../.venv/bin" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source ../.venv/bin/activate
elif [ -d ".venv/bin" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

# Function to display usage
usage() {
    echo -e "${BLUE}POLARIS Test Runner${NC}"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  unit              Run unit tests only"
    echo "  integration       Run integration tests only"
    echo "  performance       Run performance tests only"
    echo "  all               Run all test suites"
    echo "  coverage          Run coverage analysis"
    echo "  quick             Run unit tests without coverage (fast)"
    echo "  watch             Run tests in watch mode"
    echo "  help              Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 unit                    # Run all unit tests"
    echo "  $0 integration             # Run all integration tests"
    echo "  $0 quick                   # Quick unit test run"
    echo ""
}

# Parse command line arguments
case "${1:-unit}" in
    unit)
        echo -e "${GREEN}Running Unit Tests...${NC}"
        python -m tests.run_tests --unit
        ;;
    integration)
        echo -e "${GREEN}Running Integration Tests...${NC}"
        python -m tests.run_tests --integration
        ;;
    performance)
        echo -e "${GREEN}Running Performance Tests...${NC}"
        python -m tests.run_tests --performance
        ;;
    all)
        echo -e "${GREEN}Running All Tests...${NC}"
        python -m tests.run_tests --all
        ;;
    coverage)
        echo -e "${GREEN}Running Coverage Analysis...${NC}"
        python -m tests.run_tests --coverage
        ;;
    quick)
        echo -e "${GREEN}Running Quick Unit Tests (no coverage)...${NC}"
        python -m tests.run_tests --unit --no-coverage
        ;;
    watch)
        echo -e "${GREEN}Running Tests in Watch Mode...${NC}"
        python -m tests.run_tests --watch
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        echo ""
        usage
        exit 1
        ;;
esac
