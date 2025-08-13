"""
Common configuration loader for POLARIS adapters.
Loads environment variables from a .env file or other supported formats
and sets them into os.environ for global access.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import yaml

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
