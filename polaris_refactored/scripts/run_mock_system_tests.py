#!/usr/bin/env python3
"""
Enhanced Test Execution Script for POLARIS Mock System Testing.

This script provides comprehensive testing with improved logging, observability,
real-time health checks, and detailed success/failure reporting.

Features:
- Real-time health monitoring of both systems
- Detailed structured logging with correlation IDs
- Metrics collection and reporting
- TCP connectivity verification
- Comprehensive test result summary

Usage:
    python scripts/run_mock_system_tests.py [--scenario SCENARIO] [--duration DURATION]
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Flag to track if shutdown was requested
_shutdown_requested = False


@dataclass
class TestMetrics:
    """Metrics collected during test execution."""
    start_time: float = 0.0
    end_time: float = 0.0
    mock_system_startup_time: float = 0.0
    polaris_startup_time: float = 0.0
    health_checks_passed: int = 0
    health_checks_failed: int = 0
    tcp_connections_successful: int = 0
    tcp_connections_failed: int = 0
    metrics_collected: int = 0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_seconds": self.end_time - self.start_time if self.end_time else 0,
            "mock_system_startup_time": self.mock_system_startup_time,
            "polaris_startup_time": self.polaris_startup_time,
            "health_checks_passed": self.health_checks_passed,
            "health_checks_failed": self.health_checks_failed,
            "tcp_connections_successful": self.tcp_connections_successful,
            "tcp_connections_failed": self.tcp_connections_failed,
            "metrics_collected": self.metrics_collected,
            "error_count": len(self.errors),
            "errors": self.errors[:10]  # Limit to first 10 errors
        }


class StructuredLogger:
    """Enhanced logger with structured output and correlation IDs."""
    
    def __init__(self, name: str, log_file: Optional[str] = None, log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(correlation_id)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler for detailed logs
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_format = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(correlation_id)s | %(component)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S.%f'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
        
        self.correlation_id = str(uuid.uuid4())[:8]
    
    def _log(self, level: int, msg: str, component: str = "test_runner", **kwargs):
        extra = {
            'correlation_id': self.correlation_id,
            'component': component,
            **kwargs
        }
        self.logger.log(level, msg, extra=extra)
    
    def info(self, msg: str, component: str = "test_runner"):
        self._log(logging.INFO, msg, component)
    
    def debug(self, msg: str, component: str = "test_runner"):
        self._log(logging.DEBUG, msg, component)
    
    def warning(self, msg: str, component: str = "test_runner"):
        self._log(logging.WARNING, msg, component)
    
    def error(self, msg: str, component: str = "test_runner"):
        self._log(logging.ERROR, msg, component)
    
    def success(self, msg: str, component: str = "test_runner"):
        self._log(logging.INFO, f"✓ {msg}", component)
    
    def failure(self, msg: str, component: str = "test_runner"):
        self._log(logging.ERROR, f"✗ {msg}", component)


class EnhancedMockSystemTestRunner:
    """Enhanced test runner with comprehensive observability."""
    
    def __init__(self, test_dir: str, logger: StructuredLogger):
        self.logger = logger
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent
        
        test_dir_path = Path(test_dir)
        if not test_dir_path.is_absolute():
            self.test_dir = self.script_dir / test_dir_path
        else:
            self.test_dir = test_dir_path
        
        self.mock_system_process: Optional[subprocess.Popen] = None
        self.polaris_process: Optional[subprocess.Popen] = None
        self.running_processes: List[subprocess.Popen] = []
        self._shutdown_in_progress = False
        
        self.metrics = TestMetrics()
        self.mock_system_port = 5000
    
    def _signal_handler(self, signum: int, frame) -> None:
        global _shutdown_requested
        if not _shutdown_requested:
            _shutdown_requested = True
            self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
    
    def check_tcp_port(self, host: str, port: int, timeout: float = 2.0) -> bool:
        """Check if a TCP port is accepting connections."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            self.logger.debug(f"TCP check failed for {host}:{port}: {e}")
            return False
    
    async def send_mock_system_command(self, command: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Send a command to the mock system and parse response."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection('localhost', self.mock_system_port),
                timeout=timeout
            )
            
            writer.write(f"{command}\n".encode())
            await writer.drain()
            
            response = await asyncio.wait_for(reader.readline(), timeout=timeout)
            writer.close()
            await writer.wait_closed()
            
            response_str = response.decode().strip()
            if response_str.startswith("OK|"):
                data = json.loads(response_str[3:])
                return {"status": "OK", "data": data.get("data", {}), "message": data.get("message", "")}
            elif response_str.startswith("ERROR|"):
                data = json.loads(response_str[6:])
                return {"status": "ERROR", "data": data.get("data", {}), "message": data.get("message", "")}
            
            return None
        except Exception as e:
            self.logger.debug(f"Command '{command}' failed: {e}")
            return None

    async def verify_mock_system_health(self) -> bool:
        """Verify mock system is healthy via TCP command."""
        response = await self.send_mock_system_command("health_check")
        if response and response.get("status") == "OK":
            self.metrics.health_checks_passed += 1
            data = response.get("data", {})
            self.logger.success(
                f"Mock system healthy - uptime: {data.get('uptime_seconds', 0):.1f}s, "
                f"connections: {data.get('connections', 0)}",
                component="mock_system"
            )
            return True
        self.metrics.health_checks_failed += 1
        return False
    
    async def collect_mock_system_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect metrics from mock system."""
        response = await self.send_mock_system_command("get_metrics")
        if response and response.get("status") == "OK":
            self.metrics.metrics_collected += 1
            return response.get("data", {})
        return None
    
    async def start_mock_system(self, config_file: str, port: int = 5000) -> bool:
        """Start the mock external system with enhanced monitoring."""
        self.mock_system_port = port
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting mock system on port {port}", component="mock_system")
            
            mock_system_script = self.project_root / "mock_external_system" / "scripts" / "start_mock_system.py"
            config_path = Path(config_file)
            if not config_path.is_absolute():
                config_path = self.project_root / config_path
            
            if not mock_system_script.exists():
                self.logger.failure(f"Mock system script not found: {mock_system_script}", component="mock_system")
                return False
            
            cmd = [
                sys.executable,
                str(mock_system_script),
                "--config", str(config_path),
                "--port", str(port),
                "--log-level", "DEBUG"
            ]
            
            log_dir = self.test_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / "mock_system.log"
            self.logger.info(f"Mock system logs: {log_file}", component="mock_system")
            
            with open(log_file, 'w') as f:
                # Write header
                f.write(f"=== Mock System Log Started at {datetime.now().isoformat()} ===\n")
                f.write(f"Config: {config_path}\n")
                f.write(f"Port: {port}\n")
                f.write("=" * 60 + "\n\n")
                f.flush()
                
                self.mock_system_process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.project_root),
                    start_new_session=True
                )
            
            self.running_processes.append(self.mock_system_process)
            
            # Wait for TCP port to be available with progressive checks
            max_wait = 15
            check_interval = 0.5
            elapsed = 0
            
            self.logger.info("Waiting for mock system TCP port...", component="mock_system")
            
            while elapsed < max_wait:
                if self.mock_system_process.poll() is not None:
                    self.logger.failure("Mock system process exited prematurely", component="mock_system")
                    self._log_process_output(log_file, "mock_system")
                    return False
                
                if self.check_tcp_port("localhost", port):
                    self.metrics.tcp_connections_successful += 1
                    self.logger.success(f"TCP port {port} is accepting connections", component="mock_system")
                    break
                
                await asyncio.sleep(check_interval)
                elapsed += check_interval
                
                if int(elapsed) % 3 == 0 and elapsed > 0:
                    self.logger.debug(f"Still waiting for port {port}... ({elapsed:.1f}s)", component="mock_system")
            else:
                self.metrics.tcp_connections_failed += 1
                self.logger.failure(f"Timeout waiting for TCP port {port}", component="mock_system")
                return False
            
            # Verify health via protocol
            await asyncio.sleep(1)  # Give server time to fully initialize
            if not await self.verify_mock_system_health():
                self.logger.warning("Health check failed, but TCP is up - continuing", component="mock_system")
            
            self.metrics.mock_system_startup_time = time.time() - start_time
            self.logger.success(
                f"Mock system started in {self.metrics.mock_system_startup_time:.2f}s (PID: {self.mock_system_process.pid})",
                component="mock_system"
            )
            return True
            
        except Exception as e:
            self.metrics.errors.append(f"Mock system start failed: {str(e)}")
            self.logger.failure(f"Failed to start mock system: {e}", component="mock_system")
            return False
    
    async def start_polaris_framework(self, config_file: str) -> bool:
        """Start the POLARIS framework with enhanced monitoring."""
        start_time = time.time()
        
        try:
            self.logger.info("Starting POLARIS framework", component="polaris")
            
            polaris_script = self.project_root / "start_polaris_framework.py"
            config_path = Path(config_file)
            if not config_path.is_absolute():
                config_path = self.project_root / config_path
            
            if not polaris_script.exists():
                self.logger.failure(f"POLARIS script not found: {polaris_script}", component="polaris")
                return False
            
            cmd = [
                sys.executable,
                str(polaris_script),
                "start",
                "--config", str(config_path),
                "--log-level", "DEBUG"
            ]
            
            log_dir = self.test_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / "polaris_framework.log"
            self.logger.info(f"POLARIS logs: {log_file}", component="polaris")
            
            with open(log_file, 'w') as f:
                f.write(f"=== POLARIS Framework Log Started at {datetime.now().isoformat()} ===\n")
                f.write(f"Config: {config_path}\n")
                f.write("=" * 60 + "\n\n")
                f.flush()
                
                self.polaris_process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.project_root),
                    start_new_session=True
                )
            
            self.running_processes.append(self.polaris_process)
            
            # Wait for startup
            await asyncio.sleep(5)
            
            if self.polaris_process.poll() is not None:
                self.logger.failure("POLARIS framework process exited prematurely", component="polaris")
                self._log_process_output(log_file, "polaris")
                return False
            
            self.metrics.polaris_startup_time = time.time() - start_time
            self.logger.success(
                f"POLARIS framework started in {self.metrics.polaris_startup_time:.2f}s (PID: {self.polaris_process.pid})",
                component="polaris"
            )
            return True
            
        except Exception as e:
            self.metrics.errors.append(f"POLARIS start failed: {str(e)}")
            self.logger.failure(f"Failed to start POLARIS framework: {e}", component="polaris")
            return False
    
    def _log_process_output(self, log_file: Path, component: str):
        """Log the last lines of a process output file."""
        try:
            if log_file.exists():
                with open(log_file, 'r') as f:
                    content = f.read()
                    lines = content.strip().split('\n')
                    last_lines = lines[-20:] if len(lines) > 20 else lines
                    self.logger.error(f"Last {len(last_lines)} lines from {component} log:", component=component)
                    for line in last_lines:
                        self.logger.error(f"  {line}", component=component)
        except Exception as e:
            self.logger.error(f"Could not read log file: {e}", component=component)

    async def wait_for_system_ready(self, timeout: int = 30) -> bool:
        """Wait for both systems to be ready with active health monitoring."""
        global _shutdown_requested
        self.logger.info("Performing system readiness checks...", component="test_runner")
        
        start_time = time.time()
        ready_checks = 0
        required_checks = 3  # Need 3 consecutive successful checks
        
        while time.time() - start_time < timeout:
            if _shutdown_requested:
                self.logger.warning("Shutdown requested during readiness check")
                return False
            
            # Check mock system process
            if self.mock_system_process and self.mock_system_process.poll() is not None:
                self.logger.failure("Mock system process died during readiness check", component="mock_system")
                return False
            
            # Check POLARIS process
            if self.polaris_process and self.polaris_process.poll() is not None:
                self.logger.failure("POLARIS process died during readiness check", component="polaris")
                return False
            
            # Verify mock system health
            if await self.verify_mock_system_health():
                ready_checks += 1
                self.logger.debug(f"Readiness check {ready_checks}/{required_checks} passed")
                
                if ready_checks >= required_checks:
                    self.logger.success("All systems ready and healthy!")
                    return True
            else:
                ready_checks = 0  # Reset on failure
            
            await asyncio.sleep(2)
        
        self.logger.failure(f"Timeout ({timeout}s) waiting for systems to be ready")
        return False
    
    async def execute_test_scenario(self, scenario_name: str, duration: int = 60) -> bool:
        """Execute test scenario with continuous monitoring."""
        global _shutdown_requested
        
        try:
            self.logger.info(f"Executing scenario: {scenario_name} (duration: {duration}s)", component="scenario")
            
            scenario_config_file = self.test_dir / "configs" / f"{scenario_name}_config.yaml"
            if not scenario_config_file.exists():
                self.logger.failure(f"Scenario config not found: {scenario_config_file}", component="scenario")
                return False
            
            start_time = time.time()
            last_metrics_time = 0
            metrics_interval = 10  # Collect metrics every 10 seconds
            
            collected_metrics = []
            
            while time.time() - start_time < duration:
                if _shutdown_requested:
                    self.logger.warning("Shutdown requested during scenario execution")
                    return False
                
                # Check processes
                if self.mock_system_process and self.mock_system_process.poll() is not None:
                    self.logger.failure("Mock system died during scenario", component="mock_system")
                    return False
                
                if self.polaris_process and self.polaris_process.poll() is not None:
                    self.logger.failure("POLARIS died during scenario", component="polaris")
                    return False
                
                elapsed = int(time.time() - start_time)
                
                # Collect metrics periodically
                if elapsed >= last_metrics_time + metrics_interval:
                    metrics = await self.collect_mock_system_metrics()
                    if metrics:
                        collected_metrics.append({
                            "timestamp": datetime.now().isoformat(),
                            "elapsed_seconds": elapsed,
                            "metrics": metrics
                        })
                        
                        # Log key metrics
                        cpu = metrics.get("cpu_usage", {}).get("value", "N/A")
                        mem = metrics.get("memory_usage", {}).get("value", "N/A")
                        resp = metrics.get("response_time", {}).get("value", "N/A")
                        
                        self.logger.info(
                            f"[{elapsed}/{duration}s] CPU: {cpu}%, Memory: {mem}MB, Response: {resp}ms",
                            component="metrics"
                        )
                    
                    last_metrics_time = elapsed
                
                # Progress update every 15 seconds
                if elapsed % 15 == 0 and elapsed > 0:
                    await self.verify_mock_system_health()
                
                await asyncio.sleep(1)
            
            # Save collected metrics
            metrics_file = self.test_dir / "results" / f"{scenario_name}_metrics.json"
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_file, 'w') as f:
                json.dump(collected_metrics, f, indent=2)
            
            self.logger.success(f"Scenario '{scenario_name}' completed - {len(collected_metrics)} metric samples collected")
            return True
            
        except Exception as e:
            self.metrics.errors.append(f"Scenario execution failed: {str(e)}")
            self.logger.failure(f"Scenario execution failed: {e}", component="scenario")
            return False
    
    async def collect_final_results(self) -> Dict[str, Any]:
        """Collect comprehensive test results."""
        self.logger.info("Collecting final test results...")
        
        self.metrics.end_time = time.time()
        
        # Get final mock system state
        mock_state = await self.send_mock_system_command("get_state")
        mock_actions = await self.send_mock_system_command("get_action_history")
        
        results = {
            "test_run_id": self.logger.correlation_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics.to_dict(),
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
            "mock_system_final_state": mock_state.get("data") if mock_state else None,
            "action_history": mock_actions.get("data") if mock_actions else None,
            "log_files": {
                "mock_system": str(self.test_dir / "logs" / "mock_system.log"),
                "polaris_framework": str(self.test_dir / "logs" / "polaris_framework.log"),
                "test_runner": str(self.test_dir / "logs" / "test_runner.log")
            }
        }
        
        # Save results
        results_file = self.test_dir / "results" / "test_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to: {results_file}")
        return results
    
    async def cleanup_and_shutdown(self) -> None:
        """Clean up processes with detailed logging."""
        if self._shutdown_in_progress:
            return
        self._shutdown_in_progress = True
        
        self.logger.info("Starting cleanup and shutdown...")
        
        for process in self.running_processes:
            if process and process.poll() is None:
                self.logger.info(f"Terminating process {process.pid}")
                try:
                    if sys.platform != "win32":
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    else:
                        process.terminate()
                except (ProcessLookupError, PermissionError, OSError) as e:
                    self.logger.debug(f"SIGTERM failed: {e}, trying direct terminate")
                    try:
                        process.terminate()
                    except Exception:
                        pass
        
        await asyncio.sleep(3)
        
        for process in self.running_processes:
            if process and process.poll() is None:
                self.logger.warning(f"Force killing process {process.pid}")
                try:
                    if sys.platform != "win32":
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:
                        process.kill()
                except Exception:
                    pass
        
        for process in self.running_processes:
            if process:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Process {process.pid} did not terminate")
        
        self.logger.success("Cleanup completed")

    async def run_complete_test_suite(self, scenario: str, duration: int) -> bool:
        """Run complete test suite with comprehensive reporting."""
        global _shutdown_requested
        
        original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.metrics.start_time = time.time()
        
        try:
            self.logger.info("=" * 60)
            self.logger.info("POLARIS Mock System Test Suite")
            self.logger.info(f"Test Run ID: {self.logger.correlation_id}")
            self.logger.info(f"Scenario: {scenario}")
            self.logger.info(f"Duration: {duration}s")
            self.logger.info("=" * 60)
            
            # Verify test environment
            if not self.test_dir.exists():
                self.logger.failure(f"Test directory does not exist: {self.test_dir}")
                self.logger.info("Run: python scripts/setup_mock_system_test.py first")
                return False
            
            # Prepare configs
            mock_config = self.test_dir / "configs" / f"mock_{scenario}_config.yaml"
            polaris_config = self.test_dir / "configs" / f"{scenario}_config.yaml"
            
            if not mock_config.exists():
                self.logger.failure(f"Mock config not found: {mock_config}")
                return False
            
            if not polaris_config.exists():
                self.logger.failure(f"POLARIS config not found: {polaris_config}")
                return False
            
            self.logger.info("-" * 40)
            self.logger.info("Phase 1: Starting Mock System")
            self.logger.info("-" * 40)
            
            if not await self.start_mock_system(str(mock_config)):
                return False
            
            if _shutdown_requested:
                return False
            
            self.logger.info("-" * 40)
            self.logger.info("Phase 2: Starting POLARIS Framework")
            self.logger.info("-" * 40)
            
            if not await self.start_polaris_framework(str(polaris_config)):
                return False
            
            if _shutdown_requested:
                return False
            
            self.logger.info("-" * 40)
            self.logger.info("Phase 3: System Readiness Verification")
            self.logger.info("-" * 40)
            
            if not await self.wait_for_system_ready():
                return False
            
            if _shutdown_requested:
                return False
            
            self.logger.info("-" * 40)
            self.logger.info("Phase 4: Test Scenario Execution")
            self.logger.info("-" * 40)
            
            if not await self.execute_test_scenario(scenario, duration):
                return False
            
            self.logger.info("-" * 40)
            self.logger.info("Phase 5: Results Collection")
            self.logger.info("-" * 40)
            
            results = await self.collect_final_results()
            
            # Print summary
            self.logger.info("=" * 60)
            self.logger.info("TEST SUMMARY")
            self.logger.info("=" * 60)
            self.logger.success(f"Test Run ID: {self.logger.correlation_id}")
            self.logger.success(f"Duration: {results['metrics']['duration_seconds']:.2f}s")
            self.logger.success(f"Mock System Startup: {results['metrics']['mock_system_startup_time']:.2f}s")
            self.logger.success(f"POLARIS Startup: {results['metrics']['polaris_startup_time']:.2f}s")
            self.logger.success(f"Health Checks Passed: {results['metrics']['health_checks_passed']}")
            self.logger.success(f"Metrics Collected: {results['metrics']['metrics_collected']}")
            
            if results['metrics']['error_count'] > 0:
                self.logger.warning(f"Errors: {results['metrics']['error_count']}")
            
            self.logger.info("=" * 60)
            self.logger.success("TEST SUITE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.metrics.errors.append(f"Test suite failed: {str(e)}")
            self.logger.failure(f"Test suite execution failed: {e}")
            return False
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)
            await self.cleanup_and_shutdown()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhanced POLARIS Mock System Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_mock_system_tests.py --scenario basic_test
  python scripts/run_mock_system_tests.py --scenario high_load_test --duration 120
  python scripts/run_mock_system_tests.py --setup-only
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
        '--setup-only',
        action='store_true',
        help='Only set up test environment, do not run tests'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


async def setup_test_environment(test_dir: Path, logger: StructuredLogger) -> bool:
    """Set up the test environment if it doesn't exist."""
    if test_dir.exists():
        logger.info(f"Test environment already exists: {test_dir}")
        return True
    
    logger.info("Setting up test environment...")
    
    script_dir = Path(__file__).parent
    setup_script = script_dir / "setup_mock_system_test.py"
    
    if not setup_script.exists():
        logger.failure(f"Setup script not found: {setup_script}")
        return False
    
    result = subprocess.run(
        [sys.executable, str(setup_script), "--test-dir", str(test_dir)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.failure(f"Setup failed: {result.stderr}")
        return False
    
    logger.success("Test environment set up successfully")
    return True


async def main():
    global _shutdown_requested
    _shutdown_requested = False
    
    args = parse_arguments()
    
    test_dir = Path(args.test_dir)
    if not test_dir.is_absolute():
        test_dir = Path(__file__).parent / test_dir
    
    log_dir = test_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "test_runner.log"
    
    logger = StructuredLogger("polaris_test", str(log_file), args.log_level)
    
    logger.info("Enhanced Mock System Test Runner Starting")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working Directory: {Path.cwd()}")
    
    try:
        # Setup environment if needed
        if not await setup_test_environment(test_dir, logger):
            sys.exit(1)
        
        if args.setup_only:
            logger.success("Setup complete (--setup-only specified)")
            sys.exit(0)
        
        # Run tests
        test_runner = EnhancedMockSystemTestRunner(str(test_dir), logger)
        success = await test_runner.run_complete_test_suite(args.scenario, args.duration)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.warning("Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.failure(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
