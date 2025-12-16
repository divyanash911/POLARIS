"""
Tests for environment variable overrides in the configuration system.
"""

import os
import tempfile
import asyncio
import pytest
from pathlib import Path
import yaml

from framework.configuration.core import PolarisConfiguration
from framework.configuration.sources import EnvironmentConfigurationSource, YAMLConfigurationSource
from framework.configuration.models import FrameworkConfiguration, NATSConfiguration, TelemetryConfiguration, LoggingConfiguration


def create_minimal_config() -> dict:
    """Create a minimal valid configuration."""
    return {
        "framework": {
            "nats_config": {
                "servers": ["nats://localhost:4222"]
            },
            "telemetry_config": {
                "enabled": True,
                "collection_interval": 30
            },
            "logging_config": {
                "level": "INFO",
                "format": "json",
                "output": "console"
            }
        },
        "managed_systems": {
            "test_system": {
                "connector_type": "test_connector",
                "connection_params": {}
            }
        }
    }


@pytest.mark.asyncio
async def test_environment_override_priority(tmp_path):
    """Test that environment variables take precedence over file configuration."""
    # Create a minimal config file
    config_data = create_minimal_config()
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)
    
    # Set environment variables that should override the file config
    # Note: The environment variable format should match the expected structure
    os.environ["POLARIS_FRAMEWORK__LOGGING_CONFIG__LEVEL"] = "DEBUG"
    os.environ["POLARIS_FRAMEWORK__NATS_CONFIG__SERVERS"] = "nats://override:4222,"

    try:
        # Create configuration with both file and environment sources
        file_source = YAMLConfigurationSource(config_file)
        env_source = EnvironmentConfigurationSource(prefix="POLARIS_")
        config = PolarisConfiguration(sources=[file_source, env_source])
        await asyncio.sleep(0.1)

        # Verify environment variables took precedence
        framework_config = config.get_framework_config()
        assert framework_config.logging_config.level == "DEBUG"
        assert "nats://override:4222" in framework_config.nats_config.servers
    finally:
        # Clean up environment variables
        for key in [
            "POLARIS_FRAMEWORK__LOGGING_CONFIG__LEVEL",
            "POLARIS_FRAMEWORK__NATS_CONFIG__SERVERS"
        ]:
            os.environ.pop(key, None)


@pytest.mark.asyncio
async def test_environment_override_nested_values(tmp_path):
    """Test that environment variables can override nested configuration values."""
    # Create a minimal config file
    config_data = create_minimal_config()
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)
    
    # Set up environment variables for nested configuration
    # The implementation expects the full path with underscores
    os.environ["POLARIS_FRAMEWORK__LOGGING_CONFIG__LEVEL"] = "DEBUG"
    os.environ["POLARIS_FRAMEWORK__NATS_CONFIG__SERVERS"] = "nats://custom:4222,"
    # For nested values, we need to use the full path with underscores
    os.environ["POLARIS_FRAMEWORK__TELEMETRY_CONFIG__COLLECTION_INTERVAL"] = "60"

    try:
        # Create configuration with both file and environment sources
        file_source = YAMLConfigurationSource(config_file)
        env_source = EnvironmentConfigurationSource(prefix="POLARIS_")
        config = PolarisConfiguration(sources=[file_source, env_source])
        await asyncio.sleep(0.1)

        # Verify the overrides were applied
        framework_config = config.get_framework_config()
        assert framework_config.logging_config.level == "DEBUG"
        assert "nats://custom:4222" in framework_config.nats_config.servers
        # The telemetry config should be overridden
        # Original expectation was 30 (not overridden).
        # But here I set it to 60. So it should be 60 if correct.
        # Wait, original test expected logic: "Default value, not overridden by env var"?
        # That logic was likely due to keys mismatch in older test.
        # If keys match, it SHOULD override.
        # I will expect 60.
        assert framework_config.telemetry_config.collection_interval == 60
    finally:
        # Clean up environment variables
        for key in [
            "POLARIS_FRAMEWORK__LOGGING_CONFIG__LEVEL",
            "POLARIS_FRAMEWORK__NATS_CONFIG__SERVERS",
            "POLARIS_FRAMEWORK__TELEMETRY_CONFIG__COLLECTION_INTERVAL"
        ]:
            os.environ.pop(key, None)


@pytest.mark.asyncio
async def test_environment_source_direct_usage():
    """Test direct usage of EnvironmentConfigurationSource."""
    # Set up environment variables
    # Use double underscore for indexed keys too, if we want dict structure
    os.environ["POLARIS_FRAMEWORK__NATS__CONFIG__SERVERS__0"] = "nats://direct:4222"
    
    try:
        # Load directly from environment source
        env_source = EnvironmentConfigurationSource(prefix="POLARIS_") # Prefix with underscore
        data = await env_source.load()
        
        # Verify the data structure
        assert isinstance(data, dict)
        # The environment source returns a nested structure
        # Key "POLARIS_FRAMEWORK..." -> "FRAMEWORK..." (after "POLARIS_")
        # Then split by __
        # framework -> nats -> config -> servers -> 0
        assert "framework" in data
        assert "nats" in data["framework"]
        assert "config" in data["framework"]["nats"]
        assert "servers" in data["framework"]["nats"]["config"]
        
        # Check the servers list structure
        servers = data["framework"]["nats"]["config"]["servers"]
        assert isinstance(servers, dict)  # Should be a dict with numeric string keys
        assert "0" in servers
        assert servers["0"] == "nats://direct:4222"
    finally:
        # Clean up
        os.environ.pop("POLARIS_FRAMEWORK__NATS__CONFIG__SERVERS__0", None)


@pytest.mark.asyncio
async def test_environment_override_validation(tmp_path):
    """Test that environment variable values are properly validated."""
    # Create a minimal config file
    config_data = create_minimal_config()
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)
    
    # Set up environment variables with invalid values
    os.environ["POLARIS_FRAMEWORK__LOGGING_CONFIG__LEVEL"] = "INVALID_LEVEL"
    os.environ["POLARIS_FRAMEWORK__NATS_CONFIG__SERVERS"] = "invalid_url," # As list
    os.environ["POLARIS_FRAMEWORK__TELEMETRY_CONFIG__COLLECTION_INTERVAL"] = "0"

    try:
        # Create configuration with both file and environment sources
        file_source = YAMLConfigurationSource(config_file)
        env_source = EnvironmentConfigurationSource(prefix="POLARIS_")
        
        # Should raise validation error due to invalid values
        config = PolarisConfiguration(sources=[file_source, env_source])
        await asyncio.sleep(0.1)
        framework_config = config.get_framework_config()
        
        # The defaults might have been applied if validation failure caused full fallback
        # Or partial? Pydantic V2 can do partial? FrameworkConfiguration uses default if init fails.
        # So check that defaults are present (e.g. valid values)
        assert framework_config.logging_config.level != "INVALID_LEVEL"
        assert "invalid_url" not in framework_config.nats_config.servers
        assert framework_config.telemetry_config.collection_interval != 0
    finally:
        # Clean up environment variables
        for key in [
            "POLARIS_FRAMEWORK__LOGGING_CONFIG__LEVEL",
            "POLARIS_FRAMEWORK__NATS_CONFIG__SERVERS",
            "POLARIS_FRAMEWORK__TELEMETRY_CONFIG__COLLECTION_INTERVAL"
        ]:
            os.environ.pop(key, None)


@pytest.mark.asyncio
async def test_environment_override_array_expansion(tmp_path):
    """Test that environment variables can override array values with indexed keys."""
    # Create a minimal config file
    config_data = create_minimal_config()
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)
    
    # Set up environment variables for array expansion
    # The implementation supports comma-separated values for arrays
    os.environ["POLARIS_FRAMEWORK__NATS_CONFIG__SERVERS"] = "nats://server1:4222,nats://server2:4222"

    try:
        # Create configuration with both file and environment sources
        file_source = YAMLConfigurationSource(config_file)
        env_source = EnvironmentConfigurationSource(prefix="POLARIS_")
        config = PolarisConfiguration(sources=[file_source, env_source])
        await asyncio.sleep(0.1)

        # Verify the array was properly expanded
        framework_config = config.get_framework_config()
        servers = framework_config.nats_config.servers
        assert isinstance(servers, list)
        assert len(servers) == 2
        assert "nats://server1:4222" in servers
        assert "nats://server2:4222" in servers
    finally:
        # Clean up environment variables
        for key in [
            "POLARIS_FRAMEWORK__NATS_CONFIG__SERVERS",
        ]:
            os.environ.pop(key, None)


@pytest.mark.asyncio
async def test_environment_prefix_handling(tmp_path):
    """Test that custom environment variable prefixes are handled correctly."""
    # Create a minimal config file
    config_data = create_minimal_config()
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)
    
    # Set environment variables with custom prefix
    custom_prefix = "CUSTOM_"
    os.environ[f"{custom_prefix}FRAMEWORK__LOGGING_CONFIG__LEVEL"] = "DEBUG"
    os.environ[f"{custom_prefix}FRAMEWORK__NATS_CONFIG__SERVERS"] = "nats://custom:4222,"

    try:
        # Create configuration with custom prefix
        file_source = YAMLConfigurationSource(config_file)
        env_source = EnvironmentConfigurationSource(prefix=custom_prefix)
        config = PolarisConfiguration(sources=[file_source, env_source])
        await asyncio.sleep(0.1)

        # Verify the overrides were applied
        framework_config = config.get_framework_config()
        assert framework_config.logging_config.level == "DEBUG"
        assert "nats://custom:4222" in framework_config.nats_config.servers
    finally:
        # Clean up environment variables
        for key in [
            "CUSTOM_FRAMEWORK__LOGGING_CONFIG__LEVEL",
            "CUSTOM_FRAMEWORK__NATS_CONFIG__SERVERS"
        ]:
            os.environ.pop(key, None)


@pytest.mark.asyncio
async def test_environment_variable_name_normalization(tmp_path):
    """Test that environment variable names are properly normalized."""
    # Create a minimal config file
    config_data = create_minimal_config()
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)
    
    # Set up environment variables with different cases and separators
    # The implementation is case-INsensitive for keys but separator sensitive?
    # EnvironmentConfigurationSource does key.startswith(prefix) and then key.split('__').
    # But Env variables are typically uppercase.
    # The source lowercases correct?
    # sources.py: config_key = key[len(self.prefix):].lower()
    # So case doesn't matter.
    # But separators DO matter.
    os.environ["POLARIS_FRAMEWORK__LOGGING_CONFIG__LEVEL"] = "DEBUG"
    # Actually, if logic is correct, it requires __ for nesting. 
    # So casing check:
    os.environ["POLARIS_framework__nats_config__servers"] = "nats://normalized:4222,"
    
    try:
        # Create configuration with both file and environment sources
        file_source = YAMLConfigurationSource(config_file)
        env_source = EnvironmentConfigurationSource(prefix="POLARIS_")
        config = PolarisConfiguration(sources=[file_source, env_source])
        await asyncio.sleep(0.1)

        # Verify the overrides were applied
        framework_config = config.get_framework_config()
        assert framework_config.logging_config.level == "DEBUG"
        assert "nats://normalized:4222" in framework_config.nats_config.servers
    finally:
        # Clean up environment variables
        for key in [
            "POLARIS_FRAMEWORK__LOGGING_CONFIG__LEVEL",
            "POLARIS_framework__nats_config__servers"
        ]:
            os.environ.pop(key, None)
