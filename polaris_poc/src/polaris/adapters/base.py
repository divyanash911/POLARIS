"""
Base classes and contracts for POLARIS adapters.

This module defines the abstract base classes and interfaces that
all managed system connectors and adapters must implement.
"""

import abc
import asyncio
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type

from polaris.common.config import ConfigurationManager
from polaris.common.nats_client import NATSClient


class ManagedSystemConnector(abc.ABC):
    """Abstract base class for all managed system connectors.
    
    This interface defines the contract that all managed system
    connectors must implement to integrate with POLARIS.
    """
    
    def __init__(self, system_config: Dict[str, Any], logger: logging.Logger):
        """Initialize the connector.
        
        Args:
            system_config: Complete configuration for the managed system
            logger: Logger instance for structured logging
        """
        self.config = system_config
        self.logger = logger
        self.connection_config = system_config.get("connection", {})
        self.implementation_config = system_config.get("implementation", {})
        
    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish connection to the managed system.
        
        This method should handle all necessary setup to communicate
        with the managed system, including authentication.
        
        Raises:
            ConnectionError: If connection cannot be established
        """
        pass
    
    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the managed system.
        
        This method should cleanly close all connections and
        release any resources.
        """
        pass
    
    @abc.abstractmethod
    async def execute_command(
        self,
        command_template: str,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a command on the managed system.
        
        Args:
            command_template: Command template (may include placeholders)
            params: Parameters to substitute in the template
            
        Returns:
            Raw response from the managed system
            
        Raises:
            Exception: If command execution fails
        """
        pass
    
    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Check if the managed system is healthy and responsive.
        
        Returns:
            True if system is healthy, False otherwise
        """
        pass
    
    async def validate_connection(self) -> bool:
        """Validate that the connection is still active.
        
        Default implementation performs a health check.
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            return await self.health_check()
        except Exception as e:
            self.logger.warning(
                "Connection validation failed",
                extra={"error": str(e)}
            )
            return False
    
    def get_timeout(self) -> float:
        """Get the configured timeout for operations.
        
        Returns:
            Timeout in seconds
        """
        return self.implementation_config.get("timeout", 30.0)
    
    def get_max_retries(self) -> int:
        """Get the configured maximum retries for operations.
        
        Returns:
            Maximum number of retries
        """
        return self.implementation_config.get("max_retries", 3)


class BaseAdapter(abc.ABC):
    """Base class for all POLARIS adapters.
    
    This class provides common functionality for monitor and execution
    adapters, including configuration loading, connector management,
    and NATS communication.
    """
    
    def __init__(
        self,
        polaris_config_path: str,
        plugin_dir: str,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the base adapter.
        
        Args:
            polaris_config_path: Path to POLARIS framework configuration
            plugin_dir: Directory containing the managed system plugin
            logger: Logger instance (created if not provided)
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.plugin_dir = Path(plugin_dir)
        
        # Initialize configuration manager
        self.config_manager = ConfigurationManager(self.logger)
        
        # Load framework configuration
        self.framework_config = self.config_manager.load_framework_config(
            polaris_config_path
        )
        
        # Load schema for validation
        schema_path = Path(__file__).parent.parent.parent / "config" / "managed_system.schema.json"
        if schema_path.exists():
            self.config_manager.load_schema(schema_path)
        
        # Load plugin configuration
        self.plugin_config = self.config_manager.load_plugin_config(
            self.plugin_dir,
            validate=True
        )
        
        # Initialize NATS client
        nats_config = self.framework_config.get("nats", {})
        self.nats_client = NATSClient(
            nats_url=nats_config.get("url", "nats://localhost:4222"),
            logger=self.logger,
            name=f"{self.__class__.__name__}-{self.plugin_config.get('system_name', 'unknown')}"
        )
        
        # Load and initialize the managed system connector
        self.connector = self._load_connector()
        
        # Runtime state
        self.running = False
        self._tasks = []
        
    def _load_connector(self) -> ManagedSystemConnector:
        """Dynamically load and instantiate the managed system connector.
        
        Returns:
            Instantiated connector object
            
        Raises:
            ImportError: If connector module cannot be imported
            ValueError: If connector class not found or invalid
        """
        # Add plugin directory to Python path
        if str(self.plugin_dir) not in sys.path:
            sys.path.insert(0, str(self.plugin_dir))
        
        # Get connector class path
        connector_path = self.config_manager.get_plugin_connector_class()
        
        # Split module and class name
        if '.' in connector_path:
            module_path, class_name = connector_path.rsplit('.', 1)
        else:
            # Assume it's just a class name in the connector module
            module_path = "connector"
            class_name = connector_path
        
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the connector class
            connector_class = getattr(module, class_name, None)
            if connector_class is None:
                raise ValueError(f"Class {class_name} not found in module {module_path}")
            
            # Verify it's a subclass of ManagedSystemConnector
            if not issubclass(connector_class, ManagedSystemConnector):
                raise ValueError(
                    f"{class_name} must be a subclass of ManagedSystemConnector"
                )
            
            # Instantiate the connector
            connector = connector_class(
                system_config=self.plugin_config,
                logger=self.logger
            )
            
            self.logger.info(
                "Connector loaded successfully",
                extra={
                    "connector_class": class_name,
                    "system_name": self.plugin_config.get("system_name")
                }
            )
            
            return connector
            
        except ImportError as e:
            self.logger.error(
                "Failed to import connector module",
                extra={
                    "module": module_path,
                    "error": str(e)
                }
            )
            raise
        except Exception as e:
            self.logger.error(
                "Failed to load connector",
                extra={
                    "connector_path": connector_path,
                    "error": str(e)
                }
            )
            raise
    
    async def start(self) -> None:
        """Start the adapter.
        
        This method establishes connections and starts processing.
        """
        self.logger.info("Starting adapter")
        
        # Connect to NATS
        await self.nats_client.connect()
        
        # Connect to managed system
        await self.connector.connect()
        
        # Set running flag
        self.running = True
        
        # Start adapter-specific processing
        await self._start_processing()
        
        self.logger.info(
            "Adapter started",
            extra={
                "system_name": self.plugin_config.get("system_name"),
                "status": "running"
            }
        )
    
    async def stop(self) -> None:
        """Stop the adapter gracefully.
        
        This method stops processing and closes connections.
        """
        self.logger.info("Stopping adapter")
        
        # Clear running flag
        self.running = False
        
        # Stop adapter-specific processing
        await self._stop_processing()
        
        # Cancel any running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Disconnect from managed system
        try:
            await self.connector.disconnect()
        except Exception as e:
            self.logger.error(
                "Error disconnecting from managed system",
                extra={"error": str(e)}
            )
        
        # Close NATS connection
        await self.nats_client.close()
        
        self.logger.info("Adapter stopped")
    
    @abc.abstractmethod
    async def _start_processing(self) -> None:
        """Start adapter-specific processing.
        
        This method should be implemented by subclasses to start
        their specific processing logic (e.g., monitoring loop,
        action subscription).
        """
        pass
    
    @abc.abstractmethod
    async def _stop_processing(self) -> None:
        """Stop adapter-specific processing.
        
        This method should be implemented by subclasses to stop
        their specific processing logic gracefully.
        """
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
