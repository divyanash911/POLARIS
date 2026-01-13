#!/usr/bin/env python3
"""
POLARIS SWIM System Validation Script

This script validates that the POLARIS framework is properly configured
and ready to run with all three adaptation strategies.
"""

import os
import sys
import subprocess
import socket
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

def print_header(title: str):
    print(f"\n{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}{title.center(60)}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}")

def print_check(description: str, status: bool, details: str = ""):
    status_symbol = f"{Colors.GREEN}✓{Colors.NC}" if status else f"{Colors.RED}✗{Colors.NC}"
    print(f"{status_symbol} {description}")
    if details:
        print(f"   {details}")

def check_python_version() -> Tuple[bool, str]:
    """Check Python version."""
    version = sys.version_info
    if version >= (3, 8):
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"

def check_virtual_environment() -> Tuple[bool, str]:
    """Check if virtual environment exists and is activated."""
    venv_path = Path(".venv")
    if not venv_path.exists():
        return False, "Virtual environment not found (.venv directory missing)"
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return True, "Virtual environment activated"
    else:
        return False, "Virtual environment exists but not activated"

def check_dependencies() -> Tuple[bool, str]:
    """Check if required Python packages are installed."""
    required_packages = [
        'nats',
        'yaml',
        'google.generativeai',
        'aiohttp',
        'pydantic'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        return False, f"Missing packages: {', '.join(missing_packages)}"
    else:
        return True, "All required packages installed"

def check_environment_variables() -> Tuple[bool, str]:
    """Check required environment variables."""
    required_vars = {
        'GOOGLE_AI_API_KEY': 'Google AI API key for LLM reasoning',
        'NATS_URL': 'NATS server URL',
        'LOGGER_LEVEL': 'Logging level'
    }
    
    missing_vars = []
    invalid_vars = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing_vars.append(f"{var} ({description})")
        elif var == 'GOOGLE_AI_API_KEY' and value == 'your_gemini_api_key_here':
            invalid_vars.append(f"{var} (still has default value)")
    
    if missing_vars or invalid_vars:
        issues = missing_vars + invalid_vars
        return False, f"Issues: {'; '.join(issues)}"
    else:
        return True, "All environment variables properly set"

def check_configuration_files() -> Tuple[bool, str]:
    """Check configuration files exist and are valid."""
    config_files = [
        'config/swim_system_config_optimized.yaml',
        '.env'
    ]
    
    missing_files = []
    for config_file in config_files:
        if not Path(config_file).exists():
            missing_files.append(config_file)
    
    if missing_files:
        return False, f"Missing files: {', '.join(missing_files)}"
    
    # Validate YAML syntax
    try:
        import yaml
        with open('config/swim_system_config_optimized.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for required sections
        required_sections = ['framework', 'llm', 'control_reasoning', 'digital_twin']
        missing_sections = [s for s in required_sections if s not in config]
        
        if missing_sections:
            return False, f"Missing config sections: {', '.join(missing_sections)}"
        
        return True, "Configuration files valid"
        
    except Exception as e:
        return False, f"Configuration validation error: {e}"

def check_directories() -> Tuple[bool, str]:
    """Check required directories exist."""
    required_dirs = [
        'logs',
        'data/storage',
        'data/cache',
        'plugins/swim',
        'bin'
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not Path(directory).exists():
            missing_dirs.append(directory)
    
    if missing_dirs:
        return False, f"Missing directories: {', '.join(missing_dirs)}"
    else:
        return True, "All required directories exist"

def check_nats_server() -> Tuple[bool, str]:
    """Check NATS server availability."""
    nats_binary = Path("bin/nats-server")
    if not nats_binary.exists():
        return False, "NATS server binary not found (bin/nats-server)"
    
    # Check if NATS is running
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('localhost', 4222))
            if result == 0:
                return True, "NATS server running on port 4222"
            else:
                return False, "NATS server not running (port 4222 not accessible)"
    except Exception as e:
        return False, f"Error checking NATS server: {e}"

def check_framework_imports() -> Tuple[bool, str]:
    """Check if framework components can be imported."""
    try:
        # Add src to path
        src_path = Path("src")
        if src_path.exists():
            sys.path.insert(0, str(src_path))
        
        # Test critical imports
        from framework.cli.manager import PolarisFrameworkManager
        from infrastructure.llm.client import LLMClient
        from control_reasoning.adaptive_controller import AdaptiveController
        from digital_twin.world_model import StatisticalWorldModel
        
        return True, "Framework components can be imported"
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"

def test_llm_connectivity() -> Tuple[bool, str]:
    """Test LLM API connectivity."""
    try:
        import google.generativeai as genai
        
        api_key = os.getenv('GOOGLE_AI_API_KEY')
        if not api_key or api_key == 'your_gemini_api_key_here':
            return False, "API key not set or still has default value"
        
        genai.configure(api_key=api_key)
        
        # Try to list models (lightweight test)
        try:
            models = list(genai.list_models())
            if models:
                return True, f"API connectivity verified ({len(models)} models available)"
            else:
                return False, "API accessible but no models found"
        except Exception as e:
            return False, f"API test failed: {e}"
            
    except Exception as e:
        return False, f"LLM connectivity test error: {e}"

def main():
    """Main validation function."""
    print_header("POLARIS SWIM System Validation")
    
    # Check if we're in the correct directory
    if not Path("start_polaris_framework.py").exists():
        print(f"{Colors.RED}✗ Please run this script from the polaris_refactored directory{Colors.NC}")
        return False
    
    # Define all checks
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Python Dependencies", check_dependencies),
        ("Environment Variables", check_environment_variables),
        ("Configuration Files", check_configuration_files),
        ("Directory Structure", check_directories),
        ("NATS Server", check_nats_server),
        ("Framework Imports", check_framework_imports),
        ("LLM Connectivity", test_llm_connectivity),
    ]
    
    # Run all checks
    results = []
    for check_name, check_func in checks:
        try:
            status, details = check_func()
            print_check(check_name, status, details)
            results.append((check_name, status, details))
        except Exception as e:
            print_check(check_name, False, f"Check failed: {e}")
            results.append((check_name, False, f"Check failed: {e}"))
    
    # Summary
    print_header("Validation Summary")
    
    passed = sum(1 for _, status, _ in results if status)
    total = len(results)
    
    print(f"Checks passed: {passed}/{total}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}✓ All checks passed! System is ready to run.{Colors.NC}")
        print(f"\nTo start the system:")
        print(f"  ./start_swim_system.sh")
        print(f"\nOr manually:")
        print(f"  python3 start_polaris_framework.py start --config config/swim_system_config_optimized.yaml --log-level DEBUG --shell")
        return True
    else:
        print(f"\n{Colors.RED}✗ {total - passed} checks failed. Please fix the issues above.{Colors.NC}")
        
        # Provide specific guidance for common issues
        failed_checks = [name for name, status, _ in results if not status]
        
        if "Virtual Environment" in failed_checks:
            print(f"\n{Colors.YELLOW}To fix virtual environment:{Colors.NC}")
            print(f"  python3 -m venv .venv")
            print(f"  source .venv/bin/activate")
        
        if "Python Dependencies" in failed_checks:
            print(f"\n{Colors.YELLOW}To fix dependencies:{Colors.NC}")
            print(f"  pip install -r requirements.txt")
        
        if "Environment Variables" in failed_checks:
            print(f"\n{Colors.YELLOW}To fix environment variables:{Colors.NC}")
            print(f"  cp .env.example .env")
            print(f"  nano .env  # Update with your API keys")
        
        if "NATS Server" in failed_checks:
            print(f"\n{Colors.YELLOW}To fix NATS server:{Colors.NC}")
            print(f"  python3 setup_swim_system.py  # Will download and install NATS")
        
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted by user{Colors.NC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Validation failed with error: {e}{Colors.NC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)