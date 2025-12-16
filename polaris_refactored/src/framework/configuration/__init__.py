"""
POLARIS Configuration System

Provides configuration management with multiple sources and hot reload capabilities.
"""

from .core import PolarisConfiguration
from .builder import ConfigurationBuilder
from .sources import ConfigurationSource, YAMLConfigurationSource, EnvironmentConfigurationSource
from .models import (
    FrameworkConfiguration, 
    ManagedSystemConfiguration,
    NATSConfiguration,
    TelemetryConfiguration,
    LoggingConfiguration
)
from .validation import ConfigurationValidator, ConfigurationValidationError
from .utils import load_configuration_from_file, load_default_configuration, load_hot_reload_configuration

__all__ = [
    'PolarisConfiguration',
    'ConfigurationBuilder', 
    'ConfigurationSource',
    'YAMLConfigurationSource',
    'EnvironmentConfigurationSource',
    'FrameworkConfiguration',
    'ManagedSystemConfiguration',
    'NATSConfiguration',
    'TelemetryConfiguration',
    'LoggingConfiguration',
    'ConfigurationValidator',
    'ConfigurationValidationError',
    'load_configuration_from_file',
    'load_default_configuration',
    'load_hot_reload_configuration'
]
