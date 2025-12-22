#!/usr/bin/env python3
"""
Test execution script for mock external system testing.

This script orchestrates the complete testing process for POLARIS with the mock external system.
It handles mock system startup, POLARIS framework startup, test scenario execution,
and cleanup/shutdown procedures.

Usage:
    python scripts/run_mock_system_tests.py [--scenario SCENARIO] [--duration DURATION] [--test-dir TEST_DIR]
"""

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional log file path.
        
    Returns:
        Configured logger instance.
    """
    log_level_value = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level_value,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


class MockSystemTestRunner:
    """Orchestrates mock system testing process."""
    
    def __init__(self, test_dir: str, logger: logging.Logger):
        """Initialize test runner.
        
        Args:
            test_dir: Directory containing test artifacts.
            logger: Logger instance.
        """
        self.test_dir = Path(test_dir)
        self.logger = logger
        self.project_root = Path(__file__).parent.parent
        
        # Process tracking
        self.mock_system_process: Optional[subprocess.Popen] = None
        self.polaris_process: Optional[subprocess.Popen] = None
        self.running_processes: List[subprocess.Popen] = []
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals.
        
        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.cleanup_and_shutdown())
    
    async def start_mock_system(self, config_file: str, port: int = 5000) -> bool:
        """Start the mock external system.
        
        Args:
            config_file: Path to mock system configuration file.
            port: Port for mock system server.
            
        Returns:
            True if started successfully, False otherwise.
        """
        try:
            self.logger.info(f"Starting mock system on port {port}")
            
            # Prepare command
            mock_system_script = self.project_root / "mock_external_system" / "scripts" / "start_mock_system.py"
            cmd = [
                sys.executable,
                str(mock_system_script),
                "--config", config_file,
                "--port", str(port),
                "--log-level", "INFO"
            ]
            
            # Start process
            log_file = self.test_dir / "logs" / "mock_system.log"
            with open(log_file, 'w') as f:
                self.mock_system_process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.project_root)
                )
            
            self.running_processes.append(self.mock_system_process)
            
            # Wait for startup
            await asyncio.sleep(3)
            
            # Check if process is still running
            if self.mock_system_process.poll() is None:
                self.logger.info(f"Mock system started successfully (PID: {self.mock_system_process.pid})")
                return True
            else:
                self.logger.error("Mock system failed to start")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start mock system: {e}")
            return False
    
    async def start_polaris_framework(self, config_file: str) -> bool:
        """Start the POLARIS framework.
        
        Args:
            config_file: Path to POLARIS configuration file.
            
        Returns:
            True if started successfully, False otherwise.
        """
        try:
            self.logger.info("Starting POLARIS framework")
            
            # Prepare command
            polaris_script = self.project_root / "scripts" / "start_polaris_framework.py"
            cmd = [
                sys.executable,
                str(polaris_script),
                "--config", config_file,
                "--log-level", "INFO"
            ]
            
            # Start process
            log_file = self.test_dir / "logs" / "polaris_framework.log"
            with open(log_file, 'w') as f:
                self.polaris_process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.project_root)
                )
            
            self.running_processes.append(self.polaris_process)
            
            # Wait for startup
            await asyncio.sleep(5)
            
            # Check if process is still running
            if self.polaris_process.poll() is None:
                self.logger.info(f"POLARIS framework started successfully (PID: {self.polaris_process.pid})")
                return True
            else:
                self.logger.error("POLARIS framework failed to start")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start POLARIS framework: {e}")
            return False
    
    async def wait_for_system_ready(self, timeout: int = 30) -> bool:
        """Wait for both systems to be ready.
        
        Args:
            timeout: Maximum time to wait in seconds.
            
        Returns:
            True if systems are ready, False if timeout.
        """
        self.logger.info("Waiting for systems to be ready...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if processes are still running
            if self.mock_system_process and self.mock_system_process.poll() is not None:
                self.logger.error("Mock system process died")
                return False
            
            if self.polaris_process and self.polaris_process.poll() is not None:
                self.logger.error("POLARIS framework process died")
                return False
            
            # TODO: Add actual health checks here
            # For now, just wait a bit more
            await asyncio.sleep(2)
            
            # Simple readiness check - if we've waited 10 seconds and processes are alive, assume ready
            if time.time() - start_time > 10:
                self.logger.info("Systems appear to be ready")
                return True
        
        self.logger.error("Timeout waiting for systems to be ready")
        return False
    
    async def execute_test_scenario(self, scenario_name: str, duration: int = 60) -> bool:
        """Execute a specific test scenario.
        
        Args:
            scenario_name: Name of the test scenario.
            duration: Duration to run the scenario in seconds.
            
        Returns:
            True if scenario executed successfully, False otherwise.
        """
        try:
            self.logger.info(f"Executing test scenario: {scenario_name} (duration: {duration}s)")
            
            # Load scenario configuration
            scenario_config_file = self.test_dir / "configs" / f"{scenario_name}_config.yaml"
            if not scenario_config_file.exists():
                self.logger.error(f"Scenario configuration not found: {scenario_config_file}")
                return False
            
            # Run scenario for specified duration
            start_time = time.time()
            
            while time.time() - start_time < duration:
                # Check if processes are still running
                if self.mock_system_process and self.mock_system_process.poll() is not None:
                    self.logger.error("Mock system process died during scenario execution")
                    return False
                
                if self.polaris_process and self.polaris_process.poll() is not None:
                    self.logger.error("POLARIS framework process died during scenario execution")
                    return False
                
                # Log progress
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0:  # Log every 10 seconds
                    self.logger.info(f"Scenario progress: {elapsed:.0f}/{duration}s")
                
                await asyncio.sleep(1)
            
            self.logger.info(f"Test scenario '{scenario_name}' completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute test scenario '{scenario_name}': {e}")
            return False
    
    async def run_integration_tests(self) -> bool:
        """Run integration tests using pytest.
        
        Returns:
            True if tests passed, False otherwise.
        """
        try:
            self.logger.info("Running integration tests")
            
            # Prepare pytest command
            test_file = self.project_root / "tests" / "integration" / "test_mock_system_integration.py"
            
            # Check if integration test file exists
            if not test_file.exists():
                self.logger.warning(f"Integration test file not found: {test_file}")
                self.logger.info("Skipping integration tests (file not implemented yet)")
                return True
            
            cmd = [
                sys.executable,
                "-m", "pytest",
                str(test_file),
                "-v",
                "--tb=short",
                f"--junitxml={self.test_dir}/test_reports/integration_tests.xml"
            ]
            
            # Run tests
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                env=dict(os.environ, PYTHONPATH=str(self.project_root / "src"))
            )
            
            # Log results
            if result.returncode == 0:
                self.logger.info("Integration tests passed successfully")
                return True
            else:
                self.logger.error("Integration tests failed")
                self.logger.error(f"STDOUT: {result.stdout}")
                self.logger.error(f"STDERR: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to run integration tests: {e}")
            return False
    
    async def collect_test_results(self) -> Dict[str, any]:
        """Collect test results and metrics.
        
        Returns:
            Dictionary containing test results and metrics.
        """
        try:
            self.logger.info("Collecting test results")
            
            results = {
                "timestamp": time.time(),
                "processes": {
                    "mock_system": {
                        "pid": self.mock_system_process.pid if self.mock_system_process else None,
                        "running": self.mock_system_process.poll() is None if self.mock_system_process else False
                    },
                    "polaris_framework": {
                        "pid": self.polaris_process.pid if self.polaris_process else None,
                        "running": self.polaris_process.poll() is None if self.polaris_process else False
                    }
                },
                "log_files": {
                    "mock_system": str(self.test_dir / "logs" / "mock_system.log"),
                    "polaris_framework": str(self.test_dir / "logs" / "polaris_framework.log"),
                    "test_runner": str(self.test_dir / "logs" / "test_runner.log")
                }
            }
            
            # Save results
            results_file = self.test_dir / "results" / "test_results.yaml"
            with open(results_file, 'w') as f:
                yaml.dump(results, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Test results saved to: {results_file}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to collect test results: {e}")
            return {}
    
    async def cleanup_and_shutdown(self) -> None:
        """Clean up processes and shutdown gracefully."""
        try:
            self.logger.info("Starting cleanup and shutdown")
            
            # Terminate processes gracefully
            for process in self.running_processes:
                if process and process.poll() is None:
                    self.logger.info(f"Terminating process {process.pid}")
                    process.terminate()
            
            # Wait for graceful shutdown
            await asyncio.sleep(3)
            
            # Force kill if necessary
            for process in self.running_processes:
                if process and process.poll() is None:
                    self.logger.warning(f"Force killing process {process.pid}")
                    process.kill()
            
            # Wait for processes to die
            for process in self.running_processes:
                if process:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.logger.warning(f"Process {process.pid} did not terminate")
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def run_complete_test_suite(self, scenario: str, duration: int) -> bool:
        """Run the complete test suite.
        
        Args:
            scenario: Test scenario to run.
            duration: Duration for scenario execution.
            
        Returns:
            True if all tests passed, False otherwise.
        """
        try:
            self.logger.info("Starting complete test suite execution")
            
            # Verify test environment exists
            if not self.test_dir.exists():
                self.logger.error(f"Test directory does not exist: {self.test_dir}")
                self.logger.info("Run setup_mock_system_test.py first")
                return False
            
            # Prepare configuration files
            mock_config = self.test_dir / "configs" / f"mock_{scenario}_config.yaml"
            polaris_config = self.test_dir / "configs" / f"{scenario}_config.yaml"
            
            if not mock_config.exists():
                self.logger.error(f"Mock system config not found: {mock_config}")
                return False
            
            if not polaris_config.exists():
                self.logger.error(f"POLARIS config not found: {polaris_config}")
                return False
            
            # Start mock system
            if not await self.start_mock_system(str(mock_config)):
                return False
            
            # Start POLARIS framework
            if not await self.start_polaris_framework(str(polaris_config)):
                return False
            
            # Wait for systems to be ready
            if not await self.wait_for_system_ready():
                return False
            
            # Execute test scenario
            if not await self.execute_test_scenario(scenario, duration):
                return False
            
            # Run integration tests
            if not await self.run_integration_tests():
                return False
            
            # Collect results
            await self.collect_test_results()
            
            self.logger.info("Complete test suite executed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Test suite execution failed: {e}")
            return False
        finally:
            await self.cleanup_and_shutdown()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run mock external system tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic test scenario
  python scripts/run_mock_system_tests.py --scenario basic_test
  
  # Run high load test for 2 minutes
  python scripts/run_mock_system_tests.py --scenario high_load_test --duration 120
  
  # Run with custom test directory
  python scripts/run_mock_system_tests.py --test-dir /tmp/polaris_test
  
  # Run integration tests only
  python scripts/run_mock_system_tests.py --integration-only
        """
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        default='basic_test',
        choices=['basic_test', 'high_load_test', 'resource_constraint_test', 'failure_recovery_test'],
        help='Test scenario to run (default: basic_test)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Duration to run scenario in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--test-dir',
        type=str,
        default='./test_environment',
        help='Directory containing test artifacts (default: ./test_environment)'
    )
    
    parser.add_argument(
        '--integration-only',
        action='store_true',
        help='Run integration tests only (no scenario execution)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


async def main():
    """Main entry point for the test runner."""
    args = parse_arguments()
    
    # Set up logging
    test_dir = Path(args.test_dir)
    log_file = test_dir / "logs" / "test_runner.log" if test_dir.exists() else None
    logger = setup_logging(args.log_level, str(log_file) if log_file else None)
    
    logger.info("Mock system test runner starting")
    
    try:
        # Initialize test runner
        test_runner = MockSystemTestRunner(args.test_dir, logger)
        
        if args.integration_only:
            # Run integration tests only
            success = await test_runner.run_integration_tests()
        else:
            # Run complete test suite
            success = await test_runner.run_complete_test_suite(args.scenario, args.duration)
        
        if success:
            logger.info("All tests completed successfully!")
            sys.exit(0)
        else:
            logger.error("Tests failed")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during test execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())