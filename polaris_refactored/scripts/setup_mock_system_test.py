#!/usr/bin/env python3
"""
Setup script for mock external system testing.

This script sets up the environment for testing POLARIS with the mock external system.
It creates necessary directory structures, verifies dependencies, and generates
configuration files for comprehensive testing.

Usage:
    python scripts/setup_mock_system_test.py [--test-dir TEST_DIR] [--clean] [--verify-only]
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        
    Returns:
        Configured logger instance.
    """
    log_level_value = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level_value,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    return logging.getLogger(__name__)


class MockSystemTestSetup:
    """Handles setup of mock system testing environment."""
    
    def __init__(self, test_dir: str, logger: logging.Logger):
        """Initialize setup handler.
        
        Args:
            test_dir: Directory for test artifacts.
            logger: Logger instance.
        """
        self.test_dir = Path(test_dir)
        self.logger = logger
        self.project_root = Path(__file__).parent.parent
        
    def create_directory_structure(self) -> bool:
        """Create necessary directory structure for testing.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            self.logger.info("Creating directory structure")
            
            # Create main test directory
            self.test_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            directories = [
                "logs",
                "results",
                "configs",
                "temp",
                "performance_results",
                "test_reports"
            ]
            
            for directory in directories:
                dir_path = self.test_dir / directory
                dir_path.mkdir(exist_ok=True)
                self.logger.debug(f"Created directory: {dir_path}")
            
            self.logger.info(f"Directory structure created in {self.test_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create directory structure: {e}")
            return False
    
    def verify_dependencies(self) -> Tuple[bool, List[str]]:
        """Verify all required dependencies are available.
        
        Returns:
            Tuple of (success, list of missing dependencies).
        """
        self.logger.info("Verifying dependencies")
        
        missing_deps = []
        
        # Check Python packages
        python_packages = [
            "asyncio",
            "yaml", 
            "pytest",
            "hypothesis",
            "aiofiles",
            "psutil"
        ]
        
        for package in python_packages:
            try:
                __import__(package)
                self.logger.debug(f"✓ Python package: {package}")
            except ImportError:
                missing_deps.append(f"Python package: {package}")
                self.logger.warning(f"✗ Missing Python package: {package}")
        
        # Check system commands
        system_commands = [
            "python3",
            "pip",
            "pytest"
        ]
        
        for command in system_commands:
            if shutil.which(command) is None:
                missing_deps.append(f"System command: {command}")
                self.logger.warning(f"✗ Missing system command: {command}")
            else:
                self.logger.debug(f"✓ System command: {command}")
        
        # Check POLARIS components
        polaris_components = [
            self.project_root / "src" / "framework",
            self.project_root / "src" / "adapters",
            self.project_root / "src" / "control_reasoning",
            self.project_root / "mock_external_system" / "src",
            self.project_root / "plugins" / "mock_system"
        ]
        
        for component in polaris_components:
            if not component.exists():
                missing_deps.append(f"POLARIS component: {component}")
                self.logger.warning(f"✗ Missing POLARIS component: {component}")
            else:
                self.logger.debug(f"✓ POLARIS component: {component}")
        
        success = len(missing_deps) == 0
        if success:
            self.logger.info("All dependencies verified successfully")
        else:
            self.logger.error(f"Missing {len(missing_deps)} dependencies")
        
        return success, missing_deps
    
    def generate_test_configurations(self) -> bool:
        """Generate configuration files for different test scenarios.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            self.logger.info("Generating test configurations")
            
            configs_dir = self.test_dir / "configs"
            
            # Base configuration template
            base_config = {
                "framework": {
                    "service_name": "polaris-mock-system-test",
                    "environment": "testing",
                    "logging_config": {
                        "level": "INFO",
                        "format": "text",
                        "output": "console",
                        "file_path": str(self.test_dir / "logs" / "polaris_test.log")
                    },
                    "plugin_search_paths": [str(self.project_root / "plugins")]
                },
                "managed_systems": {
                    "mock_system": {
                        "system_id": "mock_system",
                        "connector_type": "mock_system",
                        "enabled": True,
                        "connection": {
                            "host": "localhost",
                            "port": 5000
                        },
                        "implementation": {
                            "timeout": 10.0,
                            "max_retries": 3,
                            "retry_delay": 1.0
                        },
                        "monitoring_config": {
                            "collection_interval": 5,
                            "collection_strategy": "polling_direct_connector",
                            "metrics_to_collect": [
                                "cpu_usage", "memory_usage", "response_time",
                                "throughput", "error_rate", "capacity"
                            ]
                        }
                    }
                },
                "control_reasoning": {
                    "adaptive_controller": {
                        "enabled": True,
                        "control_strategies": ["threshold_reactive"]
                    },
                    "threshold_reactive": {
                        "enabled": True,
                        "default_cooldown_seconds": 30.0,
                        "rules": []
                    }
                }
            }
            
            # Test scenario configurations
            scenarios = {
                "basic_test": {
                    "description": "Basic functionality test",
                    "rules": [
                        {
                            "rule_id": "basic_scale_up",
                            "name": "Basic Scale Up Test",
                            "enabled": True,
                            "priority": 1,
                            "cooldown_seconds": 30.0,
                            "action_type": "SCALE_UP",
                            "conditions": [
                                {
                                    "metric_name": "cpu_usage",
                                    "operator": "gt",
                                    "value": 70.0,
                                    "weight": 1.0
                                }
                            ]
                        }
                    ]
                },
                "high_load_test": {
                    "description": "High load scenario test",
                    "rules": [
                        {
                            "rule_id": "high_load_scale_up",
                            "name": "High Load Scale Up",
                            "enabled": True,
                            "priority": 3,
                            "cooldown_seconds": 60.0,
                            "action_type": "SCALE_UP",
                            "conditions": [
                                {
                                    "metric_name": "cpu_usage",
                                    "operator": "gt",
                                    "value": 85.0,
                                    "weight": 1.0
                                }
                            ]
                        },
                        {
                            "rule_id": "high_response_time_optimize",
                            "name": "High Response Time Optimize",
                            "enabled": True,
                            "priority": 2,
                            "cooldown_seconds": 90.0,
                            "action_type": "OPTIMIZE_CONFIG",
                            "conditions": [
                                {
                                    "metric_name": "response_time",
                                    "operator": "gt",
                                    "value": 500.0,
                                    "weight": 1.0
                                }
                            ]
                        }
                    ]
                },
                "resource_constraint_test": {
                    "description": "Resource constraint scenario test",
                    "rules": [
                        {
                            "rule_id": "memory_constraint_qos",
                            "name": "Memory Constraint QoS Adjustment",
                            "enabled": True,
                            "priority": 2,
                            "cooldown_seconds": 120.0,
                            "action_type": "ADJUST_QOS",
                            "conditions": [
                                {
                                    "metric_name": "memory_usage",
                                    "operator": "gt",
                                    "value": 3072.0,
                                    "weight": 1.0
                                }
                            ]
                        }
                    ]
                },
                "failure_recovery_test": {
                    "description": "Failure recovery scenario test",
                    "rules": [
                        {
                            "rule_id": "high_error_rate_restart",
                            "name": "High Error Rate Service Restart",
                            "enabled": True,
                            "priority": 4,
                            "cooldown_seconds": 300.0,
                            "action_type": "RESTART_SERVICE",
                            "conditions": [
                                {
                                    "metric_name": "error_rate",
                                    "operator": "gt",
                                    "value": 15.0,
                                    "weight": 2.0
                                }
                            ]
                        }
                    ]
                }
            }
            
            # Generate configuration files for each scenario
            for scenario_name, scenario_config in scenarios.items():
                config = base_config.copy()
                config["control_reasoning"]["threshold_reactive"]["rules"] = scenario_config["rules"]
                
                config_file = configs_dir / f"{scenario_name}_config.yaml"
                with open(config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                
                self.logger.debug(f"Generated config: {config_file}")
            
            # Generate mock system configurations
            mock_scenarios = {
                "basic_test": {
                    "server": {"host": "localhost", "port": 5000},
                    "baseline_metrics": {
                        "cpu_usage": 25.0,
                        "memory_usage": 1024.0,
                        "response_time": 80.0,
                        "throughput": 60.0,
                        "error_rate": 0.2,
                        "active_connections": 8,
                        "capacity": 5
                    },
                    "simulation": {
                        "noise_factor": 0.05,
                        "update_interval": 1.0,
                        "load_response_time": 2.0
                    }
                },
                "high_load_test": {
                    "server": {"host": "localhost", "port": 5000},
                    "baseline_metrics": {
                        "cpu_usage": 90.0,
                        "memory_usage": 3584.0,
                        "response_time": 600.0,
                        "throughput": 25.0,
                        "error_rate": 2.5,
                        "active_connections": 75,
                        "capacity": 3
                    },
                    "simulation": {
                        "noise_factor": 0.15,
                        "update_interval": 1.0,
                        "load_response_time": 1.5
                    }
                },
                "resource_constraint_test": {
                    "server": {"host": "localhost", "port": 5000},
                    "baseline_metrics": {
                        "cpu_usage": 45.0,
                        "memory_usage": 3200.0,
                        "response_time": 200.0,
                        "throughput": 40.0,
                        "error_rate": 1.0,
                        "active_connections": 30,
                        "capacity": 4
                    },
                    "simulation": {
                        "noise_factor": 0.1,
                        "update_interval": 1.0,
                        "load_response_time": 2.0
                    }
                },
                "failure_recovery_test": {
                    "server": {"host": "localhost", "port": 5000},
                    "baseline_metrics": {
                        "cpu_usage": 60.0,
                        "memory_usage": 2048.0,
                        "response_time": 300.0,
                        "throughput": 30.0,
                        "error_rate": 20.0,
                        "active_connections": 50,
                        "capacity": 5
                    },
                    "simulation": {
                        "noise_factor": 0.2,
                        "update_interval": 1.0,
                        "load_response_time": 1.0
                    }
                }
            }
            
            for scenario_name, mock_config in mock_scenarios.items():
                mock_config_file = configs_dir / f"mock_{scenario_name}_config.yaml"
                with open(mock_config_file, 'w') as f:
                    yaml.dump(mock_config, f, default_flow_style=False, indent=2)
                
                self.logger.debug(f"Generated mock config: {mock_config_file}")
            
            self.logger.info(f"Generated {len(scenarios) + len(mock_scenarios)} configuration files")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate configurations: {e}")
            return False
    
    def create_test_scripts(self) -> bool:
        """Create helper test scripts.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            self.logger.info("Creating test scripts")
            
            # Create test runner script
            test_runner_script = f"""#!/bin/bash
# Test runner script for mock system testing
# Generated by setup_mock_system_test.py

set -e

TEST_DIR="{self.test_dir}"
PROJECT_ROOT="{self.project_root}"

echo "Starting mock system tests..."
echo "Test directory: $TEST_DIR"
echo "Project root: $PROJECT_ROOT"

# Function to cleanup on exit
cleanup() {{
    echo "Cleaning up test processes..."
    pkill -f "start_mock_system.py" || true
    pkill -f "polaris_framework.py" || true
    sleep 2
}}

trap cleanup EXIT

# Run tests
cd "$PROJECT_ROOT"
python -m pytest tests/integration/test_mock_system_integration.py -v --tb=short

echo "Mock system tests completed successfully!"
"""
            
            script_file = self.test_dir / "run_tests.sh"
            with open(script_file, 'w') as f:
                f.write(test_runner_script)
            
            # Make script executable
            script_file.chmod(0o755)
            
            self.logger.debug(f"Created test runner script: {script_file}")
            
            # Create cleanup script
            cleanup_script = f"""#!/bin/bash
# Cleanup script for mock system testing
# Generated by setup_mock_system_test.py

echo "Cleaning up mock system test environment..."

# Kill any running processes
pkill -f "start_mock_system.py" || true
pkill -f "polaris_framework.py" || true

# Remove temporary files
rm -rf "{self.test_dir}/temp/*"
rm -rf "{self.test_dir}/logs/*"

echo "Cleanup completed!"
"""
            
            cleanup_file = self.test_dir / "cleanup.sh"
            with open(cleanup_file, 'w') as f:
                f.write(cleanup_script)
            
            cleanup_file.chmod(0o755)
            
            self.logger.debug(f"Created cleanup script: {cleanup_file}")
            
            self.logger.info("Test scripts created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create test scripts: {e}")
            return False
    
    def setup_environment(self) -> bool:
        """Set up the complete test environment.
        
        Returns:
            True if successful, False otherwise.
        """
        self.logger.info("Setting up mock system test environment")
        
        # Create directory structure
        if not self.create_directory_structure():
            return False
        
        # Generate configurations
        if not self.generate_test_configurations():
            return False
        
        # Create test scripts
        if not self.create_test_scripts():
            return False
        
        self.logger.info("Mock system test environment setup completed successfully")
        return True
    
    def clean_environment(self) -> bool:
        """Clean up the test environment.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            self.logger.info(f"Cleaning test environment: {self.test_dir}")
            
            if self.test_dir.exists():
                shutil.rmtree(self.test_dir)
                self.logger.info("Test environment cleaned successfully")
            else:
                self.logger.info("Test environment directory does not exist")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clean test environment: {e}")
            return False


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Set up mock external system testing environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set up test environment in default directory
  python scripts/setup_mock_system_test.py
  
  # Set up test environment in custom directory
  python scripts/setup_mock_system_test.py --test-dir /tmp/polaris_test
  
  # Clean existing test environment
  python scripts/setup_mock_system_test.py --clean
  
  # Verify dependencies only
  python scripts/setup_mock_system_test.py --verify-only
        """
    )
    
    parser.add_argument(
        '--test-dir',
        type=str,
        default='./test_environment',
        help='Directory for test artifacts (default: ./test_environment)'
    )
    
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean existing test environment'
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Verify dependencies and exit'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the setup script."""
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    logger.info("Mock system test setup starting")
    
    try:
        # Initialize setup handler
        setup_handler = MockSystemTestSetup(args.test_dir, logger)
        
        # Handle clean option
        if args.clean:
            success = setup_handler.clean_environment()
            sys.exit(0 if success else 1)
        
        # Verify dependencies
        deps_ok, missing_deps = setup_handler.verify_dependencies()
        
        if not deps_ok:
            logger.error("Dependency verification failed:")
            for dep in missing_deps:
                logger.error(f"  - {dep}")
            
            logger.info("To install missing Python packages, run:")
            logger.info("  pip install asyncio pyyaml pytest hypothesis aiofiles psutil")
            
            sys.exit(1)
        
        if args.verify_only:
            logger.info("Dependency verification successful (--verify-only)")
            sys.exit(0)
        
        # Set up environment
        success = setup_handler.setup_environment()
        
        if success:
            logger.info(f"Setup completed successfully!")
            logger.info(f"Test directory: {args.test_dir}")
            logger.info("Next steps:")
            logger.info("  1. Run: python scripts/run_mock_system_tests.py")
            logger.info("  2. Or use the generated test scripts in the test directory")
        else:
            logger.error("Setup failed")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()