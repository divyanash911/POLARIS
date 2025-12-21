#!/usr/bin/env python3
"""
Test wrapper to run tests from the correct directory with proper PYTHONPATH
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Set up PYTHONPATH
    src_path = script_dir / "src"
    tests_path = script_dir
    
    env = os.environ.copy()
    pythonpath = f"{src_path}:{tests_path}"
    if "PYTHONPATH" in env:
        pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    
    # Change to the polaris_refactored directory
    os.chdir(script_dir)
    
    # Run the test runner with the provided arguments
    cmd = [sys.executable, "-m", "tests.run_tests"] + sys.argv[1:]
    
    print(f"Running from: {os.getcwd()}")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()