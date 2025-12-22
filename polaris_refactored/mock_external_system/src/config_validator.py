"""Configuration validation module for mock external system.

This module provides configuration validation to ensure all configuration
files are properly formatted and contain required fields.
"""

from typing import Dict, Any, List, Tuple
import yaml
from pathlib import Path


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigValidator:
    """Validates mock system configuration files."""
    
    # Required top-level sections
    REQUIRED_SECTIONS = ["server", "baseline_metrics", "simulation", "capacity"]
    
    # Required server configuration fields
    REQUIRED_SERVER_FIELDS = ["host", "port", "max_connections"]
    
    # Required baseline metrics
    REQUIRED_METRICS = [
        "cpu_usage",
        "memory_usage",
        "response_time",
        "throughput",
        "error_rate",
        "active_connections",
        "capacity"
    ]
    
    # Required simulation fields
    REQUIRED_SIMULATION_FIELDS = ["noise_factor", "update_interval", "load_response_time"]
    
    # Required capacity fields
    REQUIRED_CAPACITY_FIELDS = ["min_capacity", "max_capacity", "scale_up_increment", "scale_down_increment"]
    
    @staticmethod
    def validate_config_file(config_path: str) -> Tuple[bool, List[str]]:
        """Validate a configuration file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            Tuple of (is_valid, error_messages).
        """
        errors = []
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            return False, [f"Configuration file not found: {config_path}"]
        except yaml.YAMLError as e:
            return False, [f"Invalid YAML format: {str(e)}"]
        except Exception as e:
            return False, [f"Error reading configuration: {str(e)}"]
        
        if config is None:
            return False, ["Configuration file is empty"]
        
        # Validate required sections
        for section in ConfigValidator.REQUIRED_SECTIONS:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Validate server section
        if "server" in config:
            server_errors = ConfigValidator._validate_server(config["server"])
            errors.extend(server_errors)
        
        # Validate baseline_metrics section
        if "baseline_metrics" in config:
            metrics_errors = ConfigValidator._validate_baseline_metrics(config["baseline_metrics"])
            errors.extend(metrics_errors)
        
        # Validate simulation section
        if "simulation" in config:
            sim_errors = ConfigValidator._validate_simulation(config["simulation"])
            errors.extend(sim_errors)
        
        # Validate capacity section
        if "capacity" in config:
            cap_errors = ConfigValidator._validate_capacity(config["capacity"])
            errors.extend(cap_errors)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_server(server_config: Dict[str, Any]) -> List[str]:
        """Validate server configuration section."""
        errors = []
        
        for field in ConfigValidator.REQUIRED_SERVER_FIELDS:
            if field not in server_config:
                errors.append(f"Missing required server field: {field}")
        
        if "port" in server_config:
            port = server_config["port"]
            if not isinstance(port, int) or port < 1 or port > 65535:
                errors.append(f"Invalid port number: {port} (must be 1-65535)")
        
        if "max_connections" in server_config:
            max_conn = server_config["max_connections"]
            if not isinstance(max_conn, int) or max_conn < 1:
                errors.append(f"Invalid max_connections: {max_conn} (must be >= 1)")
        
        return errors
    
    @staticmethod
    def _validate_baseline_metrics(metrics_config: Dict[str, Any]) -> List[str]:
        """Validate baseline metrics configuration."""
        errors = []
        
        for metric in ConfigValidator.REQUIRED_METRICS:
            if metric not in metrics_config:
                errors.append(f"Missing required metric: {metric}")
        
        # Validate metric value types and ranges
        percentage_metrics = ["cpu_usage", "error_rate"]
        for metric in percentage_metrics:
            if metric in metrics_config:
                value = metrics_config[metric]
                if not isinstance(value, (int, float)):
                    errors.append(f"Metric {metric} must be numeric, got {type(value)}")
                elif value < 0 or value > 100:
                    errors.append(f"Metric {metric} must be 0-100, got {value}")
        
        non_negative_metrics = ["memory_usage", "response_time", "throughput", "active_connections", "capacity"]
        for metric in non_negative_metrics:
            if metric in metrics_config:
                value = metrics_config[metric]
                if not isinstance(value, (int, float)):
                    errors.append(f"Metric {metric} must be numeric, got {type(value)}")
                elif value < 0:
                    errors.append(f"Metric {metric} cannot be negative, got {value}")
        
        return errors
    
    @staticmethod
    def _validate_simulation(sim_config: Dict[str, Any]) -> List[str]:
        """Validate simulation configuration."""
        errors = []
        
        for field in ConfigValidator.REQUIRED_SIMULATION_FIELDS:
            if field not in sim_config:
                errors.append(f"Missing required simulation field: {field}")
        
        if "noise_factor" in sim_config:
            noise = sim_config["noise_factor"]
            if not isinstance(noise, (int, float)):
                errors.append(f"noise_factor must be numeric, got {type(noise)}")
            elif noise < 0 or noise > 1:
                errors.append(f"noise_factor must be 0-1, got {noise}")
        
        if "update_interval" in sim_config:
            interval = sim_config["update_interval"]
            if not isinstance(interval, (int, float)):
                errors.append(f"update_interval must be numeric, got {type(interval)}")
            elif interval <= 0:
                errors.append(f"update_interval must be positive, got {interval}")
        
        if "load_response_time" in sim_config:
            response = sim_config["load_response_time"]
            if not isinstance(response, (int, float)):
                errors.append(f"load_response_time must be numeric, got {type(response)}")
            elif response < 0:
                errors.append(f"load_response_time cannot be negative, got {response}")
        
        return errors
    
    @staticmethod
    def _validate_capacity(capacity_config: Dict[str, Any]) -> List[str]:
        """Validate capacity configuration."""
        errors = []
        
        for field in ConfigValidator.REQUIRED_CAPACITY_FIELDS:
            if field not in capacity_config:
                errors.append(f"Missing required capacity field: {field}")
        
        # Validate numeric types
        for field in ConfigValidator.REQUIRED_CAPACITY_FIELDS:
            if field in capacity_config:
                value = capacity_config[field]
                if not isinstance(value, int):
                    errors.append(f"Capacity field {field} must be integer, got {type(value)}")
                elif value < 1:
                    errors.append(f"Capacity field {field} must be >= 1, got {value}")
        
        # Validate min < max
        if "min_capacity" in capacity_config and "max_capacity" in capacity_config:
            min_cap = capacity_config["min_capacity"]
            max_cap = capacity_config["max_capacity"]
            if min_cap > max_cap:
                errors.append(f"min_capacity ({min_cap}) cannot exceed max_capacity ({max_cap})")
        
        return errors
    
    @staticmethod
    def validate_all_configs(config_dir: str) -> Tuple[bool, Dict[str, List[str]]]:
        """Validate all configuration files in a directory.
        
        Args:
            config_dir: Directory containing configuration files.
            
        Returns:
            Tuple of (all_valid, validation_results_dict).
        """
        config_path = Path(config_dir)
        results = {}
        all_valid = True
        
        # Validate default config
        default_config = config_path / "default_config.yaml"
        if default_config.exists():
            is_valid, errors = ConfigValidator.validate_config_file(str(default_config))
            results["default_config.yaml"] = errors
            all_valid = all_valid and is_valid
        
        # Validate scenario configs
        scenarios_dir = config_path / "scenarios"
        if scenarios_dir.exists():
            for scenario_file in scenarios_dir.glob("*.yaml"):
                is_valid, errors = ConfigValidator.validate_config_file(str(scenario_file))
                results[f"scenarios/{scenario_file.name}"] = errors
                all_valid = all_valid and is_valid
        
        return all_valid, results


def validate_and_report(config_dir: str) -> bool:
    """Validate all configurations and print report.
    
    Args:
        config_dir: Directory containing configuration files.
        
    Returns:
        True if all configurations are valid, False otherwise.
    """
    all_valid, results = ConfigValidator.validate_all_configs(config_dir)
    
    print("Configuration Validation Report")
    print("=" * 50)
    
    for config_file, errors in results.items():
        if errors:
            print(f"\n❌ {config_file}")
            for error in errors:
                print(f"   - {error}")
        else:
            print(f"\n✓ {config_file}")
    
    print("\n" + "=" * 50)
    if all_valid:
        print("✓ All configurations are valid!")
    else:
        print("❌ Some configurations have errors.")
    
    return all_valid
