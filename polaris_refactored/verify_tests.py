#!/usr/bin/env python3
"""
Test Verification Script

This script verifies that the test infrastructure is properly set up
and can discover and run tests.
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} - PASSED")
        return True
    else:
        print(f"❌ {description} - FAILED")
        if result.stdout:
            print("STDOUT:", result.stdout[:500])
        if result.stderr:
            print("STDERR:", result.stderr[:500])
        return False


def main():
    """Main verification function."""
    print("POLARIS Test Infrastructure Verification")
    print("=" * 60)
    
    # Change to polaris_refactored directory
    script_dir = Path(__file__).parent
    
    results = []
    
    # 1. Verify pytest is installed
    results.append(run_command(
        [sys.executable, "-m", "pytest", "--version"],
        "pytest installation"
    ))
    
    # 2. Verify test discovery
    results.append(run_command(
        [sys.executable, "-m", "pytest", "--collect-only", "tests/unit", "-q"],
        "Unit test discovery"
    ))
    
    # 3. Verify pytest configuration
    pytest_ini = script_dir / "pytest.ini"
    if pytest_ini.exists():
        print(f"\n✅ pytest.ini found at {pytest_ini}")
        results.append(True)
    else:
        print(f"\n❌ pytest.ini not found at {pytest_ini}")
        results.append(False)
    
    # 4. Verify test directories exist
    test_dirs = ["tests/unit", "tests/integration", "tests/performance"]
    for test_dir in test_dirs:
        dir_path = script_dir / test_dir
        if dir_path.exists():
            print(f"✅ {test_dir} directory exists")
            results.append(True)
        else:
            print(f"❌ {test_dir} directory not found")
            results.append(False)
    
    # 5. Verify conftest.py exists
    conftest = script_dir / "tests" / "conftest.py"
    if conftest.exists():
        print(f"✅ conftest.py found at {conftest}")
        results.append(True)
    else:
        print(f"❌ conftest.py not found at {conftest}")
        results.append(False)
    
    # 6. Try running a simple test collection
    results.append(run_command(
        [sys.executable, "-m", "pytest", "tests/unit", "--collect-only", "-m", "not integration and not performance"],
        "Unit test collection with markers"
    ))
    
    # Summary
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All verification checks passed!")
        print("\nYou can now run tests using:")
        print("  - ./run_tests.sh unit (Linux/WSL)")
        print("  - run_tests.bat unit (Windows)")
        print("  - make unit")
        print("  - python -m tests.run_tests --unit")
        return 0
    else:
        print("❌ Some verification checks failed.")
        print("\nPlease ensure:")
        print("  1. Virtual environment is activated")
        print("  2. Dependencies are installed: pip install pytest pytest-cov pytest-asyncio")
        print("  3. You're in the polaris_refactored directory")
        return 1


if __name__ == "__main__":
    sys.exit(main())
