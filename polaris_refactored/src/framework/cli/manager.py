
import asyncio
import signal
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from .utils import RICH_AVAILABLE, rprint

# Types
class FrameworkState(Enum):
    """Framework operational states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class MetricsSnapshot:
    """Snapshot of system metrics at a point in time."""
    timestamp: datetime
    system_id: str
    metrics: Dict[str, Any]
    health_status: str

@dataclass
class AdaptationEvent:
    """Record of an adaptation event."""
    timestamp: datetime
    action_type: str
    system_id: str
    parameters: Dict[str, Any]
    status: str
    reason: str

@dataclass
class FrameworkStatus:
    """Complete framework status."""
    state: FrameworkState
    uptime_seconds: float
    components: List[str]
    managed_systems: Dict[str, Dict[str, Any]]
    recent_adaptations: List[AdaptationEvent]
    metrics_summary: Dict[str, MetricsSnapshot]
    meta_learner_enabled: bool = False


class PolarisFrameworkManager:
    """
    High-level manager for the POLARIS framework.
    Separated from the monolithic start script.
    """
    
    def __init__(self):
        self.framework = None
        self.state = FrameworkState.STOPPED
        self.start_time: Optional[datetime] = None
        self.config_path: Optional[str] = None
        self.logger = None
        self._shutdown_event = asyncio.Event()
        self._metrics_history: List[MetricsSnapshot] = []
        self._adaptation_history: List[AdaptationEvent] = []
        
    async def initialize(self, config_path: str, log_level: str = "INFO") -> bool:
        """Initialize the framework with configuration."""
        try:
            self.state = FrameworkState.STARTING
            self.config_path = config_path
            
            # Ensure logs directory exists - assuming relative to project root or cwd
            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Load configuration first to get logging settings
            from framework.configuration.utils import load_configuration_from_file
            configuration = load_configuration_from_file(config_path)
            
            # Get framework config with logging settings
            framework_config = configuration.get_framework_config()
            
            # Configure logging
            from infrastructure.observability import configure_logging, get_framework_logger, LogLevel
            from framework.configuration.models import LoggingConfiguration
            
            # Use logging config from YAML, but allow CLI override for log level
            log_config = framework_config.logging_config
            if log_level.upper() != "INFO":  # CLI override takes precedence
                log_config = LoggingConfiguration(
                    level=log_level.upper(),
                    format=log_config.format,
                    output=log_config.output,
                    file_path=log_config.file_path,
                    max_file_size=log_config.max_file_size,
                    backup_count=log_config.backup_count
                )
            
            configure_logging(log_config)
            self.logger = get_framework_logger("manager")
            
            self.logger.info(f"Initializing POLARIS framework with config: {config_path}")
            
            # Initialize components
            from infrastructure.di import DIContainer
            from infrastructure.message_bus import PolarisMessageBus
            from infrastructure.data_storage import DataStoreFactory
            from framework.plugin_management import PolarisPluginRegistry
            from framework.events import PolarisEventBus
            from framework.polaris_framework import PolarisFramework
            from infrastructure.observability import ObservabilityConfig
            
            log_level_enum = LogLevel[log_config.level.upper()]
            
            container = DIContainer()
            message_bus = PolarisMessageBus()
            data_store = DataStoreFactory.create_in_memory_store()
            plugin_registry = PolarisPluginRegistry()
            event_bus = PolarisEventBus()
            
            obs_config = ObservabilityConfig(
                service_name=getattr(framework_config, 'service_name', 'polaris'),
                log_level=log_level_enum
            )
            
            self.framework = PolarisFramework(
                container=container,
                configuration=configuration,
                message_bus=message_bus,
                data_store=data_store,
                plugin_registry=plugin_registry,
                event_bus=event_bus,
                observability_config=obs_config
            )
            
            # Initialize metrics exporter
            from infrastructure.observability.metrics import get_metrics_collector, JSONFileMetricsExporter
            self._metrics_exporter = JSONFileMetricsExporter(
                file_path=str(logs_dir / "metrics.jsonl")
            )
            self._metrics_collector = get_metrics_collector()
            
            self.logger.info("Framework initialized successfully")
            return True
            
        except Exception as e:
            self.state = FrameworkState.ERROR
            if self.logger:
                self.logger.error(f"Failed to initialize framework: {e}", exc_info=True)
            else:
                print(f"ERROR: Failed to initialize framework: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the framework."""
        if not self.framework:
            print("ERROR: Framework not initialized. Call initialize() first.")
            return False
        
        try:
            self.logger.info("Starting POLARIS framework...")
            await self.framework.start()
            self.state = FrameworkState.RUNNING
            self.start_time = datetime.now(timezone.utc)
            self.logger.info("POLARIS framework started successfully")
            return True
        except Exception as e:
            self.state = FrameworkState.ERROR
            self.logger.error(f"Failed to start framework: {e}", exc_info=True)
            return False
    
    async def stop(self) -> bool:
        """Stop the framework."""
        if not self.framework:
            return True
        
        try:
            self.state = FrameworkState.STOPPING
            self.logger.info("Stopping POLARIS framework...")
            await self.framework.stop()
            self.state = FrameworkState.STOPPED
            self.logger.info("POLARIS framework stopped")
            return True
        except Exception as e:
            self.state = FrameworkState.ERROR
            self.logger.error(f"Error stopping framework: {e}", exc_info=True)
            return False
    
    def get_status(self) -> FrameworkStatus:
        """Get comprehensive framework status."""
        uptime = 0.0
        if self.start_time:
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        components = []
        managed_systems = {}
        meta_learner_enabled = False
        
        if self.framework:
            status = self.framework.get_status()
            components = status.get("components", [])
            meta_learner_enabled = status.get("meta_learner_enabled", False)
            
            if hasattr(self.framework, 'configuration'):
                try:
                    systems = self.framework.configuration.get_all_managed_systems()
                    for sys_id, sys_config in systems.items():
                        managed_systems[sys_id] = {
                            "connector_type": sys_config.connector_type,
                            "enabled": sys_config.enabled,
                            "connection": sys_config.connection_params
                        }
                except Exception:
                    pass
        
        return FrameworkStatus(
            state=self.state,
            uptime_seconds=uptime,
            components=components,
            managed_systems=managed_systems,
            recent_adaptations=self._adaptation_history[-10:],
            metrics_summary={s.system_id: s for s in self._metrics_history[-5:]},
            meta_learner_enabled=meta_learner_enabled
        )
    
    async def run_until_shutdown(self):
        """Run the framework until shutdown signal received."""
        self._shutdown_event.clear()
        
        # Set up signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._signal_handler)
            except NotImplementedError:
                # Windows fallback
                signal.signal(sig, lambda s, f: self._signal_handler())
        
        self.logger.info("Framework running. Press Ctrl+C to stop.")
        
        metrics_export_task = asyncio.create_task(self._periodic_metrics_export())
        
        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            metrics_export_task.cancel()
            try:
                await metrics_export_task
            except asyncio.CancelledError:
                pass
            
            if hasattr(self, '_metrics_exporter') and hasattr(self, '_metrics_collector'):
                self._metrics_exporter.export_to_file(self._metrics_collector._metrics)
                self.logger.info("Final metrics exported before shutdown")
        
        await self.stop()
    
    async def _periodic_metrics_export(self):
        """Periodically export metrics to file."""
        export_interval = 60
        
        while True:
            try:
                await asyncio.sleep(export_interval)
                if hasattr(self, '_metrics_exporter') and hasattr(self, '_metrics_collector'):
                    self._metrics_exporter.export_to_file(self._metrics_collector._metrics)
                    self.logger.debug(f"Metrics exported to file")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"Failed to export metrics: {e}")
    
    def _signal_handler(self):
        """Handle shutdown signals."""
        self.logger.info("Shutdown signal received")
        self._shutdown_event.set()


class ManagedSystemOperations:
    """Operations for managing connected systems."""
    
    def __init__(self, manager: PolarisFrameworkManager):
        self.manager = manager
    
    async def list_systems(self) -> List[Dict[str, Any]]:
        systems = []
        if self.manager.framework and hasattr(self.manager.framework, 'configuration'):
            try:
                all_systems = self.manager.framework.configuration.get_all_managed_systems()
                for sys_id, sys_config in all_systems.items():
                    systems.append({
                        "system_id": sys_id,
                        "connector_type": sys_config.connector_type,
                        "enabled": sys_config.enabled,
                        "connection": sys_config.connection_params,
                        "monitoring": sys_config.monitoring_config
                    })
            except Exception as e:
                print(f"Error listing systems: {e}")
        return systems
    
    async def collect_metrics(self, system_id: str) -> Dict[str, Any]:
        if not self.manager.framework:
            return {"error": "Framework not running"}
        
        try:
            registry = self.manager.framework.plugin_registry
            connectors = registry.get_loaded_connectors()
            connector = connectors.get(system_id)
            
            if connector:
                metrics = await connector.collect_metrics()
                return {
                    "system_id": system_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metrics": {k: {"value": v.value, "unit": v.unit, "name": v.name} 
                               for k, v in metrics.items()}
                }
        except Exception as e:
            return {"error": str(e)}
        
        return {"error": f"System {system_id} not found"}

    # Include other necessary operations (execute_action, etc.) as needed, 
    # copying from original file if robust functionality is required there.
    # For now, sticking to what's used in dashboard/shell.

    async def execute_action(self, system_id: str, action_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an adaptation action on a managed system."""
        if not self.manager.framework:
            return {"error": "Framework not running"}
        
        try:
            from domain.models import AdaptationAction
            import uuid
            
            registry = self.manager.framework.plugin_registry
            connectors = registry.get_loaded_connectors()
            connector = connectors.get(system_id)
            
            if connector:
                action = AdaptationAction(
                    action_id=str(uuid.uuid4()),
                    action_type=action_type,
                    target_system=system_id,
                    parameters=parameters
                )
                
                # Validate first
                is_valid = await connector.validate_action(action)
                if not is_valid:
                    return {"error": f"Action {action_type} validation failed"}
                
                # Execute
                result = await connector.execute_action(action)
                return {
                    "action_id": result.action_id,
                    "status": result.status.value,
                    "result_data": result.result_data,
                    "execution_time_ms": getattr(result, 'execution_time_ms', None)
                }
        except Exception as e:
            return {"error": str(e)}
        
        return {"error": f"System {system_id} not found"}
    
    async def get_supported_actions(self, system_id: str) -> List[str]:
        """Get list of supported actions for a managed system."""
        if not self.manager.framework:
            return []
        
        try:
            registry = self.manager.framework.plugin_registry
            connectors = registry.get_loaded_connectors()
            connector = connectors.get(system_id)
            
            if connector:
                return await connector.get_supported_actions()
        except Exception:
            pass
        
        return []
