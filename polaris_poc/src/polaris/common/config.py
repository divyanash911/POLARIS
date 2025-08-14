"""
Enhanced configuration management for POLARIS framework.

Provides configuration loading, validation, and management for both
the core framework and managed system plugins.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
import yaml
try:
    from jsonschema import validate, ValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    ValidationError = Exception

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATHS = [
    Path(__file__).parent / ".env",
    Path(__file__).parent / "config.yaml",
    Path(__file__).parent / "config.json",
]



def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, str]:
    """
    Recursively flattens a nested dictionary into environment-style
    keys (uppercase, underscore separated) and string values.
    Example:
        {"swim": {"host": "localhost", "port": 4242}}
        -> {"SWIM_HOST": "localhost", "SWIM_PORT": "4242"}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key.upper()] = str(v)
    return items


def _set_env_vars(config: Dict[str, Any], overwrite: bool = False):
    """Set environment variables from dictionary."""
    for key, value in config.items():
        if not overwrite and key in os.environ:
            continue
        os.environ[key] = str(value)
        logger.debug(f"Set environment variable: {key}={value}")


def _load_from_env_file(env_path: Path) -> bool:
    """Load variables from a .env file."""
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        logger.info(f"Loaded environment variables from {env_path}")
        return True
    return False

def _load_from_yaml(yaml_path: Path, overwrite: bool = True) -> bool:
    """
    Load environment variables from a YAML file.
    Supports both flat and grouped YAML structures.
    Example grouped YAML:
        swim:
          host: localhost
          port: 4242
    Will produce env vars:
        SWIM_HOST=localhost
        SWIM_PORT=4242
    """
    if not yaml_path.exists():
        logger.warning(f"YAML config file not found: {yaml_path}")
        return False

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            logger.error(f"YAML file {yaml_path} is not a mapping at root level.")
            return False

        flat_data = _flatten_dict(data)
        _set_env_vars(flat_data, overwrite=overwrite)
        logger.info(f"Loaded environment variables from {yaml_path}")
        return True

    except yaml.YAMLError as e:
        logger.exception(f"Failed to parse YAML config {yaml_path}: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error loading YAML config {yaml_path}: {e}")

    return False

def _load_from_json(json_path: Path) -> bool:
    """Load variables from a JSON config file."""
    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f) or {}
        _set_env_vars(data, overwrite=True)
        logger.info(f"Loaded environment variables from {json_path}")
        return True
    return False

def load_config(
    search_paths: Optional[list] = None,
    overwrite: bool = True,
    required_keys: Optional[list] = None
):
    """
    Load configuration variables into os.environ.

    Args:
        search_paths (list): Optional list of file paths to check.
        overwrite (bool): Whether to overwrite existing env vars.
        required_keys (list): List of keys that must be present after load.

    Raises:
        ValueError: If required keys are missing.
    """
    search_paths = search_paths or DEFAULT_CONFIG_PATHS

    loaded = False
    for path in search_paths:
        if path.suffix == ".env":
            loaded = _load_from_env_file(path) or loaded
        elif path.suffix in [".yaml", ".yml"]:
            loaded = _load_from_yaml(path) or loaded
        elif path.suffix == ".json":
            loaded = _load_from_json(path) or loaded

    if not loaded:
        logger.warning("No configuration file found, relying on system env vars only.")

    if required_keys:
        missing = [key for key in required_keys if key not in os.environ]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

    return True


def get_config(key: str, default: Any = None, cast_type: type = str):
    """
    Get a configuration value from the environment.

    Args:
        key (str): Environment variable name.
        default (Any): Default value if not found.
        cast_type (type): Type to cast the value into.

    Returns:
        Any: The configuration value.
    """
    value = os.environ.get(key, default)
    try:
        return cast_type(value) if value is not None else None
    except Exception:
        logger.warning(f"Failed to cast config value for key '{key}' to {cast_type.__name__}")
        return default


class ConfigurationManager:
    """Enhanced configuration manager with schema validation support.
    
    This class manages both POLARIS framework configuration and
    managed system plugin configurations with JSON schema validation.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize configuration manager.
        
        Args:
            logger: Logger instance for structured logging
        """
        self.logger = logger or logging.getLogger(__name__)
        self.framework_config: Dict[str, Any] = {}
        self.plugin_config: Dict[str, Any] = {}
        self.schema: Optional[Dict[str, Any]] = None
        
    def load_framework_config(
        self,
        config_path: Union[str, Path],
        required_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Load POLARIS framework configuration.
        
        Args:
            config_path: Path to framework configuration file
            required_keys: List of required configuration keys
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid or required keys missing
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ValueError(f"Configuration file not found: {config_path}")
        
        # Load configuration based on file type
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                self.framework_config = yaml.safe_load(f) or {}
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                self.framework_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file type: {config_path.suffix}")
        
        # Check required keys
        if required_keys:
            missing = [k for k in required_keys if k not in self.framework_config]
            if missing:
                raise ValueError(f"Missing required configuration keys: {missing}")
        
        # Flatten and set environment variables
        flat_config = _flatten_dict(self.framework_config)
        _set_env_vars(flat_config, overwrite=True)
        
        self.logger.info(
            "Framework configuration loaded",
            extra={"config_path": str(config_path)}
        )
        
        return self.framework_config
    
    def load_schema(self, schema_path: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON schema for plugin validation.
        
        Args:
            schema_path: Path to JSON schema file
            
        Returns:
            Loaded schema dictionary
            
        Raises:
            ValueError: If schema file not found or invalid
        """
        schema_path = Path(schema_path)
        
        if not schema_path.exists():
            raise ValueError(f"Schema file not found: {schema_path}")
        
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
        
        self.logger.info(
            "Schema loaded",
            extra={"schema_path": str(schema_path)}
        )
        
        return self.schema
    
    def load_plugin_config(
        self,
        plugin_dir: Union[str, Path],
        config_filename: str = "config.yaml",
        validate: bool = True
    ) -> Dict[str, Any]:
        """Load and validate managed system plugin configuration.
        
        Args:
            plugin_dir: Directory containing the plugin
            config_filename: Name of the configuration file
            validate: Whether to validate against schema
            
        Returns:
            Loaded and validated plugin configuration
            
        Raises:
            ValueError: If configuration is invalid or validation fails
        """
        plugin_dir = Path(plugin_dir)
        config_path = plugin_dir / config_filename
        
        if not config_path.exists():
            raise ValueError(f"Plugin configuration not found: {config_path}")
        
        # Load configuration
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                self.plugin_config = yaml.safe_load(f) or {}
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                self.plugin_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file type: {config_path.suffix}")
        
        # Validate against schema if requested
        if validate and self.schema:
            self.validate_config(self.plugin_config, self.schema)
        
        self.logger.info(
            "Plugin configuration loaded",
            extra={
                "plugin_dir": str(plugin_dir),
                "system_name": self.plugin_config.get("system_name", "unknown")
            }
        )
        
        return self.plugin_config
    
    def validate_config(
        self,
        config: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None
    ) -> None:
        """Validate configuration against JSON schema.
        
        Args:
            config: Configuration dictionary to validate
            schema: JSON schema (uses self.schema if not provided)
            
        Raises:
            ValidationError: If configuration doesn't match schema
            ValueError: If jsonschema is not available
        """
        if not JSONSCHEMA_AVAILABLE:
            self.logger.warning(
                "jsonschema not available, skipping validation. "
                "Install with: pip install jsonschema"
            )
            return
        
        schema = schema or self.schema
        if not schema:
            raise ValueError("No schema available for validation")
        
        try:
            validate(instance=config, schema=schema)
            self.logger.info("Configuration validation successful")
        except ValidationError as e:
            self.logger.error(
                "Configuration validation failed",
                extra={
                    "error": str(e),
                    "path": list(e.path) if e.path else None,
                    "message": e.message
                }
            )
            raise
    
    def get_plugin_connector_class(self) -> str:
        """Get the connector class path from plugin configuration.
        
        Returns:
            Connector class import path
            
        Raises:
            ValueError: If connector class not specified
        """
        if not self.plugin_config:
            raise ValueError("Plugin configuration not loaded")
        
        connector_class = self.plugin_config.get("implementation", {}).get("connector_class")
        if not connector_class:
            raise ValueError("Connector class not specified in plugin configuration")
        
        return connector_class
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration from plugin.
        
        Returns:
            Monitoring configuration dictionary
        """
        return self.plugin_config.get("monitoring", {})
    
    def get_execution_config(self) -> Dict[str, Any]:
        """Get execution configuration from plugin.
        
        Returns:
            Execution configuration dictionary
        """
        return self.plugin_config.get("execution", {})
    
    def get_connection_config(self) -> Dict[str, Any]:
        """Get connection configuration from plugin.
        
        Returns:
            Connection configuration dictionary
        """
        return self.plugin_config.get("connection", {})
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries.
        
        Later configurations override earlier ones.
        
        Args:
            *configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration dictionary
        """
        result = {}
        for config in configs:
            result.update(config)
        return result
