#!/usr/bin/env python3
"""
POLARIS Framework CLI - Comprehensive Management Interface

A complete command-line interface for the POLARIS (Proactive Optimization & Learning
Architecture for Resilient Intelligent Systems) framework.

Features:
- Start/stop framework with different configurations
- Managed system management (list, connect, disconnect, status)
- Real-time metrics and observability dashboard
- Adaptation monitoring and control
- Configuration management
- Interactive shell mode

Usage:
    python start_polaris_framework.py start --config config/swim_system_config.yaml
    python start_polaris_framework.py status
    python start_polaris_framework.py systems list
    python start_polaris_framework.py metrics --system swim
    python start_polaris_framework.py shell
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


# ============================================================================
# Constants and Configuration
# ============================================================================

VERSION = "2.0.0"
BANNER = r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██████╗  ██████╗ ██╗      █████╗ ██████╗ ██╗███████╗                        ║
║   ██╔══██╗██╔═══██╗██║     ██╔══██╗██╔══██╗██║██╔════╝                        ║
║   ██████╔╝██║   ██║██║     ███████║██████╔╝██║███████╗                        ║
║   ██╔═══╝ ██║   ██║██║     ██╔══██║██╔══██╗██║╚════██║                        ║
║   ██║     ╚██████╔╝███████╗██║  ██║██║  ██║██║███████║                        ║
║   ╚═╝      ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝                        ║
║                                                                               ║
║   Proactive Optimization & Learning Architecture for Resilient               ║
║   Intelligent Systems                                                         ║
║                                                                               ║
║   Version: {version:<10}                                                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""".format(version=VERSION)

# Base directory for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()

DEFAULT_CONFIGS = {
    "swim": str(SCRIPT_DIR / "config/swim_system_config.yaml"),
    "mock": str(SCRIPT_DIR / "config/mock_system_config.yaml"),
    "llm": str(SCRIPT_DIR / "config/llm_integration_config.yaml"),
}

SCENARIOS = {
    "normal": str(SCRIPT_DIR / "config/scenarios/normal_operation_config.yaml"),
    "high_load": str(SCRIPT_DIR / "config/scenarios/high_load_config.yaml"),
    "failure": str(SCRIPT_DIR / "config/scenarios/failure_recovery_config.yaml"),
    "mixed": str(SCRIPT_DIR / "config/scenarios/mixed_workload_config.yaml"),
    "resource": str(SCRIPT_DIR / "config/scenarios/resource_constraint_config.yaml"),
}


# ============================================================================
# Data Classes for State Management
# ============================================================================

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


# ============================================================================
# POLARIS Framework Manager
# ============================================================================

class PolarisFrameworkManager:
    """
    High-level manager for the POLARIS framework.
    
    Provides a user-friendly interface for:
    - Starting/stopping the framework
    - Managing managed systems
    - Monitoring metrics and adaptations
    - Configuration management
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
            
            # Configure logging first
            from infrastructure.observability import configure_logging, get_framework_logger, LogLevel
            from framework.configuration.models import LoggingConfiguration
            
            # Convert string log level to LogLevel enum
            log_level_enum = LogLevel[log_level.upper()] if isinstance(log_level, str) else log_level
            
            log_config = LoggingConfiguration(level=log_level, format="text", output="console")
            configure_logging(log_config)
            self.logger = get_framework_logger("manager")
            
            self.logger.info(f"Initializing POLARIS framework with config: {config_path}")
            
            # Load configuration
            from framework.configuration import ConfigurationBuilder
            from framework.configuration.utils import load_configuration_from_file
            
            configuration = load_configuration_from_file(config_path)
            
            # Initialize DI container and components
            from infrastructure.di import DIContainer
            from infrastructure.message_bus import PolarisMessageBus
            from infrastructure.data_storage import DataStoreFactory
            from framework.plugin_management import PolarisPluginRegistry
            from framework.events import PolarisEventBus
            from framework.polaris_framework import PolarisFramework
            from infrastructure.observability import ObservabilityConfig
            
            container = DIContainer()
            message_bus = PolarisMessageBus()
            data_store = DataStoreFactory.create_in_memory_store()
            plugin_registry = PolarisPluginRegistry()
            event_bus = PolarisEventBus()
            
            # Get observability config
            framework_config = configuration.get_framework_config()
            obs_config = ObservabilityConfig(
                service_name=getattr(framework_config, 'service_name', 'polaris'),
                log_level=log_level_enum
            )
            
            # Create framework instance
            self.framework = PolarisFramework(
                container=container,
                configuration=configuration,
                message_bus=message_bus,
                data_store=data_store,
                plugin_registry=plugin_registry,
                event_bus=event_bus,
                observability_config=obs_config
            )
            
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
        
        if self.framework:
            status = self.framework.get_status()
            components = status.get("components", [])
            
            # Get managed system info from configuration
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
            metrics_summary={s.system_id: s for s in self._metrics_history[-5:]}
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
                # Windows doesn't support add_signal_handler
                signal.signal(sig, lambda s, f: self._signal_handler())
        
        self.logger.info("Framework running. Press Ctrl+C to stop.")
        
        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass
        
        await self.stop()
    
    def _signal_handler(self):
        """Handle shutdown signals."""
        self.logger.info("Shutdown signal received")
        self._shutdown_event.set()


# ============================================================================
# Managed System Operations
# ============================================================================

class ManagedSystemOperations:
    """Operations for managing connected systems."""
    
    def __init__(self, manager: PolarisFrameworkManager):
        self.manager = manager
    
    async def list_systems(self) -> List[Dict[str, Any]]:
        """List all configured managed systems."""
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
    
    async def get_system_status(self, system_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a specific managed system."""
        if not self.manager.framework:
            return None
        
        try:
            # Get connector from plugin registry
            registry = self.manager.framework.plugin_registry
            connectors = registry.get_loaded_connectors()
            connector = connectors.get(system_id)
            
            if connector:
                state = await connector.get_system_state()
                return {
                    "system_id": system_id,
                    "connected": True,
                    "health_status": state.health_status.value,
                    "metrics": {k: {"value": v.value, "unit": v.unit} 
                               for k, v in state.metrics.items()},
                    "metadata": state.metadata,
                    "timestamp": state.timestamp.isoformat()
                }
        except Exception as e:
            return {
                "system_id": system_id,
                "connected": False,
                "error": str(e)
            }
        
        return None
    
    async def collect_metrics(self, system_id: str) -> Dict[str, Any]:
        """Collect current metrics from a managed system."""
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
    
    async def execute_action(self, system_id: str, action_type: str, 
                            parameters: Dict[str, Any]) -> Dict[str, Any]:
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


# ============================================================================
# Observability and Monitoring
# ============================================================================

class ObservabilityDashboard:
    """Real-time observability and monitoring dashboard."""
    
    def __init__(self, manager: PolarisFrameworkManager):
        self.manager = manager
        self.refresh_interval = 5.0  # seconds
    
    async def display_live_metrics(self, system_id: Optional[str] = None, 
                                   duration: int = 60):
        """Display live metrics in the terminal."""
        ops = ManagedSystemOperations(self.manager)
        end_time = time.time() + duration
        
        print("\n" + "=" * 70)
        print("POLARIS Live Metrics Dashboard")
        print("=" * 70)
        print(f"Refresh interval: {self.refresh_interval}s | Duration: {duration}s")
        print("Press Ctrl+C to stop\n")
        
        try:
            while time.time() < end_time:
                # Clear screen (cross-platform)
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print(BANNER)
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 70)
                
                # Get framework status
                status = self.manager.get_status()
                print(f"Framework State: {status.state.value.upper()}")
                print(f"Uptime: {self._format_uptime(status.uptime_seconds)}")
                print(f"Active Components: {len(status.components)}")
                print("-" * 70)
                
                # Get metrics for each system
                systems = await ops.list_systems()
                target_systems = [s for s in systems if s['enabled']]
                
                if system_id:
                    target_systems = [s for s in target_systems 
                                     if s['system_id'] == system_id]
                
                for sys_info in target_systems:
                    sid = sys_info['system_id']
                    print(f"\n📊 System: {sid} ({sys_info['connector_type']})")
                    print("-" * 40)
                    
                    metrics_data = await ops.collect_metrics(sid)
                    
                    if "error" in metrics_data:
                        print(f"  ⚠️  Error: {metrics_data['error']}")
                    else:
                        metrics = metrics_data.get("metrics", {})
                        for name, data in metrics.items():
                            value = data.get('value', 'N/A')
                            unit = data.get('unit', '')
                            bar = self._create_bar(value) if isinstance(value, (int, float)) else ""
                            print(f"  {name:25} {value:>10} {unit:5} {bar}")
                
                print("\n" + "-" * 70)
                print("Press Ctrl+C to stop")
                
                await asyncio.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\nMetrics display stopped.")
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def _create_bar(self, value: float, max_val: float = 100, width: int = 20) -> str:
        """Create a simple ASCII progress bar."""
        if not isinstance(value, (int, float)) or value < 0:
            return ""
        
        # Normalize value
        if max_val > 0:
            ratio = min(value / max_val, 1.0)
        else:
            ratio = 0
        
        filled = int(width * ratio)
        empty = width - filled
        
        # Color coding based on value
        if ratio > 0.8:
            color = "🔴"
        elif ratio > 0.6:
            color = "🟡"
        else:
            color = "🟢"
        
        return f"[{'█' * filled}{'░' * empty}] {color}"
    
    async def show_adaptation_history(self, limit: int = 20):
        """Show recent adaptation history."""
        print("\n" + "=" * 70)
        print("POLARIS Adaptation History")
        print("=" * 70)
        
        # Get from event bus if available
        if self.manager.framework and hasattr(self.manager.framework, 'event_bus'):
            try:
                event_bus = self.manager.framework.event_bus
                if hasattr(event_bus, 'get_event_history'):
                    history = event_bus.get_event_history(limit=limit)
                    
                    if not history:
                        print("No adaptation events recorded yet.")
                        return
                    
                    for event in history:
                        print(f"\n{event.get('timestamp', 'N/A')}")
                        print(f"  Type: {event.get('event_type', 'N/A')}")
                        print(f"  System: {event.get('system_id', 'N/A')}")
                        print(f"  Details: {event.get('data', {})}")
                else:
                    print("Event history not available.")
            except Exception as e:
                print(f"Error retrieving history: {e}")
        else:
            print("Framework not running.")


# ============================================================================
# Configuration Management
# ============================================================================

class ConfigurationManager:
    """Configuration management utilities."""
    
    @staticmethod
    def list_configs() -> Dict[str, str]:
        """List available configuration files."""
        configs = {}
        config_dir = Path(__file__).parent / "config"
        
        if config_dir.exists():
            for yaml_file in config_dir.glob("*.yaml"):
                configs[yaml_file.stem] = str(yaml_file)
            
            scenarios_dir = config_dir / "scenarios"
            if scenarios_dir.exists():
                for yaml_file in scenarios_dir.glob("*.yaml"):
                    configs[f"scenario/{yaml_file.stem}"] = str(yaml_file)
        
        return configs
    
    @staticmethod
    def validate_config(config_path: str) -> Dict[str, Any]:
        """Validate a configuration file."""
        try:
            import yaml
            
            # Load raw YAML first for quick validation
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            managed_systems = list(raw_config.get("managed_systems", {}).keys())
            
            return {
                "valid": True,
                "config_path": config_path,
                "framework_config": "framework" in raw_config,
                "managed_systems": managed_systems
            }
        except Exception as e:
            return {
                "valid": False,
                "config_path": config_path,
                "error": str(e)
            }
    
    @staticmethod
    def show_config(config_path: str) -> Dict[str, Any]:
        """Display configuration details."""
        try:
            import yaml
            
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            return {
                "path": config_path,
                "framework": raw_config.get("framework", {}),
                "managed_systems": list(raw_config.get("managed_systems", {}).keys()),
                "control_reasoning": raw_config.get("control_reasoning", {}).get(
                    "adaptive_controller", {}).get("control_strategies", []),
                "observability": raw_config.get("observability", {})
            }
        except Exception as e:
            return {"error": str(e)}


# ============================================================================
# Interactive Shell
# ============================================================================

class InteractiveShell:
    """Interactive shell for POLARIS management."""
    
    COMMANDS = {
        "help": "Show available commands",
        "status": "Show framework status",
        "systems": "List managed systems",
        "metrics <system_id>": "Show metrics for a system",
        "action <system_id> <action_type> [params]": "Execute an action",
        "actions <system_id>": "List supported actions for a system",
        "config": "Show current configuration",
        "dashboard [duration]": "Show live metrics dashboard",
        "history": "Show adaptation history",
        "quit": "Exit the shell"
    }
    
    def __init__(self, manager: PolarisFrameworkManager):
        self.manager = manager
        self.ops = ManagedSystemOperations(manager)
        self.dashboard = ObservabilityDashboard(manager)
        self.running = True
    
    async def run(self):
        """Run the interactive shell."""
        print("\n" + "=" * 70)
        print("POLARIS Interactive Shell")
        print("=" * 70)
        print("Type 'help' for available commands, 'quit' to exit.\n")
        
        while self.running:
            try:
                cmd_input = input("polaris> ").strip()
                if not cmd_input:
                    continue
                
                await self._process_command(cmd_input)
                
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit.")
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Goodbye!")
    
    async def _process_command(self, cmd_input: str):
        """Process a shell command."""
        parts = cmd_input.split()
        cmd = parts[0].lower()
        args = parts[1:]
        
        if cmd == "help":
            self._show_help()
        elif cmd == "quit" or cmd == "exit":
            self.running = False
        elif cmd == "status":
            await self._show_status()
        elif cmd == "systems":
            await self._list_systems()
        elif cmd == "metrics":
            if args:
                await self._show_metrics(args[0])
            else:
                print("Usage: metrics <system_id>")
        elif cmd == "action":
            if len(args) >= 2:
                params = {}
                if len(args) > 2:
                    # Parse key=value params
                    for arg in args[2:]:
                        if "=" in arg:
                            k, v = arg.split("=", 1)
                            try:
                                params[k] = json.loads(v)
                            except json.JSONDecodeError:
                                params[k] = v
                await self._execute_action(args[0], args[1], params)
            else:
                print("Usage: action <system_id> <action_type> [key=value ...]")
        elif cmd == "actions":
            if args:
                await self._list_actions(args[0])
            else:
                print("Usage: actions <system_id>")
        elif cmd == "config":
            self._show_config()
        elif cmd == "dashboard":
            duration = int(args[0]) if args else 60
            await self.dashboard.display_live_metrics(duration=duration)
        elif cmd == "history":
            await self.dashboard.show_adaptation_history()
        else:
            print(f"Unknown command: {cmd}. Type 'help' for available commands.")
    
    def _show_help(self):
        """Show help message."""
        print("\nAvailable Commands:")
        print("-" * 50)
        for cmd, desc in self.COMMANDS.items():
            print(f"  {cmd:35} {desc}")
        print()
    
    async def _show_status(self):
        """Show framework status."""
        status = self.manager.get_status()
        print(f"\nFramework State: {status.state.value.upper()}")
        print(f"Uptime: {status.uptime_seconds:.1f} seconds")
        print(f"Components: {', '.join(status.components) or 'None'}")
        print(f"Managed Systems: {', '.join(status.managed_systems.keys()) or 'None'}")
    
    async def _list_systems(self):
        """List managed systems."""
        systems = await self.ops.list_systems()
        if not systems:
            print("No managed systems configured.")
            return
        
        print(f"\n{'System ID':<20} {'Type':<15} {'Enabled':<10}")
        print("-" * 50)
        for sys in systems:
            enabled = "✓" if sys['enabled'] else "✗"
            print(f"{sys['system_id']:<20} {sys['connector_type']:<15} {enabled:<10}")
    
    async def _show_metrics(self, system_id: str):
        """Show metrics for a system."""
        data = await self.ops.collect_metrics(system_id)
        if "error" in data:
            print(f"Error: {data['error']}")
            return
        
        print(f"\nMetrics for {system_id}:")
        print("-" * 40)
        for name, metric in data.get("metrics", {}).items():
            print(f"  {name}: {metric['value']} {metric.get('unit', '')}")
    
    async def _execute_action(self, system_id: str, action_type: str, 
                             params: Dict[str, Any]):
        """Execute an action."""
        print(f"Executing {action_type} on {system_id}...")
        result = await self.ops.execute_action(system_id, action_type, params)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Status: {result['status']}")
            print(f"Result: {json.dumps(result.get('result_data', {}), indent=2)}")
    
    async def _list_actions(self, system_id: str):
        """List supported actions."""
        actions = await self.ops.get_supported_actions(system_id)
        if not actions:
            print(f"No actions available for {system_id}")
            return
        
        print(f"\nSupported actions for {system_id}:")
        for action in actions:
            print(f"  - {action}")
    
    def _show_config(self):
        """Show current configuration."""
        if self.manager.config_path:
            config = ConfigurationManager.show_config(self.manager.config_path)
            print(f"\nConfiguration: {config.get('path', 'N/A')}")
            print(f"Managed Systems: {', '.join(config.get('managed_systems', []))}")
            print(f"Control Strategies: {', '.join(config.get('control_reasoning', []))}")
        else:
            print("No configuration loaded.")


# ============================================================================
# CLI Command Handlers
# ============================================================================

async def cmd_start(args):
    """Handle the 'start' command."""
    print(BANNER)
    
    # Resolve config path
    config_path = args.config
    if config_path in DEFAULT_CONFIGS:
        config_path = DEFAULT_CONFIGS[config_path]
    elif config_path in SCENARIOS:
        config_path = SCENARIOS[config_path]
    
    # Check if config exists
    if not Path(config_path).exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        print("\nAvailable configurations:")
        for name, path in {**DEFAULT_CONFIGS, **SCENARIOS}.items():
            exists = "✓" if Path(path).exists() else "✗"
            print(f"  {exists} {name}: {path}")
        return 1
    
    print(f"Starting POLARIS with configuration: {config_path}")
    print("-" * 70)
    
    manager = PolarisFrameworkManager()
    
    # Initialize
    if not await manager.initialize(config_path, args.log_level):
        return 1
    
    # Start
    if not await manager.start():
        return 1
    
    # Run until shutdown
    if args.daemon:
        await manager.run_until_shutdown()
    else:
        # Interactive mode - start shell
        shell = InteractiveShell(manager)
        await shell.run()
        await manager.stop()
    
    return 0


async def cmd_status(args):
    """Handle the 'status' command."""
    print("\nPOLARIS Framework Status")
    print("=" * 50)
    
    # Check if framework is running (simplified check)
    print("Note: For detailed status, start the framework first.")
    print("\nAvailable configurations:")
    configs = ConfigurationManager.list_configs()
    for name, path in configs.items():
        print(f"  - {name}: {path}")
    
    return 0


async def cmd_systems(args):
    """Handle the 'systems' command."""
    if args.action == "list":
        print("\nConfigured Managed Systems")
        print("=" * 50)
        
        # List from config files
        for config_name, config_path in DEFAULT_CONFIGS.items():
            if Path(config_path).exists():
                config = ConfigurationManager.show_config(config_path)
                systems = config.get("managed_systems", [])
                if systems:
                    print(f"\n{config_name}:")
                    for sys in systems:
                        print(f"  - {sys}")
    
    elif args.action == "info":
        if not args.system_id:
            print("ERROR: system_id required for 'info' action")
            return 1
        
        print(f"\nSystem Info: {args.system_id}")
        print("=" * 50)
        print("Start the framework to get live system information.")
    
    return 0


async def cmd_config(args):
    """Handle the 'config' command."""
    if args.action == "list":
        print("\nAvailable Configurations")
        print("=" * 50)
        
        print("\nDefault Configurations:")
        for name, path in DEFAULT_CONFIGS.items():
            exists = "✓" if Path(path).exists() else "✗"
            print(f"  {exists} {name}: {path}")
        
        print("\nScenario Configurations:")
        for name, path in SCENARIOS.items():
            exists = "✓" if Path(path).exists() else "✗"
            print(f"  {exists} {name}: {path}")
    
    elif args.action == "validate":
        if not args.config_file:
            print("ERROR: config_file required for 'validate' action")
            return 1
        
        result = ConfigurationManager.validate_config(args.config_file)
        if result["valid"]:
            print(f"✓ Configuration valid: {args.config_file}")
            print(f"  Managed systems: {', '.join(result.get('managed_systems', []))}")
        else:
            print(f"✗ Configuration invalid: {args.config_file}")
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    elif args.action == "show":
        if not args.config_file:
            print("ERROR: config_file required for 'show' action")
            return 1
        
        config = ConfigurationManager.show_config(args.config_file)
        if "error" in config:
            print(f"Error: {config['error']}")
            return 1
        
        print(f"\nConfiguration: {config['path']}")
        print("=" * 50)
        print(f"Managed Systems: {', '.join(config.get('managed_systems', []))}")
        print(f"Control Strategies: {', '.join(config.get('control_reasoning', []))}")
        
        obs = config.get("observability", {})
        print(f"Observability: metrics={obs.get('enable_metrics', 'N/A')}, "
              f"tracing={obs.get('enable_tracing', 'N/A')}")
    
    return 0


async def cmd_metrics(args):
    """Handle the 'metrics' command."""
    print("\nMetrics Collection")
    print("=" * 50)
    print("Start the framework to collect live metrics.")
    print("\nUsage:")
    print("  python start_polaris_framework.py start --config swim")
    print("  Then use 'metrics <system_id>' in the interactive shell")
    return 0


async def cmd_shell(args):
    """Handle the 'shell' command."""
    print(BANNER)
    
    if not args.config:
        print("ERROR: Configuration required for shell mode")
        print("Usage: python start_polaris_framework.py shell --config <config>")
        return 1
    
    config_path = args.config
    if config_path in DEFAULT_CONFIGS:
        config_path = DEFAULT_CONFIGS[config_path]
    
    manager = PolarisFrameworkManager()
    
    if not await manager.initialize(config_path, args.log_level):
        return 1
    
    if not await manager.start():
        return 1
    
    shell = InteractiveShell(manager)
    await shell.run()
    await manager.stop()
    
    return 0


async def cmd_dashboard(args):
    """Handle the 'dashboard' command."""
    print(BANNER)
    
    if not args.config:
        print("ERROR: Configuration required for dashboard")
        print("Usage: python start_polaris_framework.py dashboard --config <config>")
        return 1
    
    config_path = args.config
    if config_path in DEFAULT_CONFIGS:
        config_path = DEFAULT_CONFIGS[config_path]
    
    manager = PolarisFrameworkManager()
    
    if not await manager.initialize(config_path, args.log_level):
        return 1
    
    if not await manager.start():
        return 1
    
    dashboard = ObservabilityDashboard(manager)
    await dashboard.display_live_metrics(
        system_id=args.system,
        duration=args.duration
    )
    
    await manager.stop()
    return 0


# ============================================================================
# CLI Argument Parser
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="start_polaris_framework.py",
        description="POLARIS Framework CLI - Comprehensive Management Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with SWIM configuration
  python start_polaris_framework.py start --config swim
  
  # Start with custom config file
  python start_polaris_framework.py start --config config/swim_system_config.yaml
  
  # Start in daemon mode (no interactive shell)
  python start_polaris_framework.py start --config swim --daemon
  
  # Validate a configuration
  python start_polaris_framework.py config validate --config-file config/swim_system_config.yaml
  
  # List available configurations
  python start_polaris_framework.py config list
  
  # Start interactive shell
  python start_polaris_framework.py shell --config swim
  
  # Start live metrics dashboard
  python start_polaris_framework.py dashboard --config swim --duration 300

Available Configurations:
  swim     - SWIM system with threshold + LLM reasoning
  mock     - Mock system for testing
  llm      - LLM integration configuration
  
Scenario Configurations:
  normal   - Normal operation scenario
  high_load - High load testing scenario
  failure  - Failure recovery scenario
  mixed    - Mixed workload scenario
  resource - Resource constraint scenario
"""
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"POLARIS Framework v{VERSION}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser(
        "start",
        help="Start the POLARIS framework"
    )
    start_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Configuration file or preset (swim, mock, llm, or path)"
    )
    start_parser.add_argument(
        "--log-level", "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    start_parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Run in daemon mode (no interactive shell)"
    )
    
    # Status command
    subparsers.add_parser(
        "status",
        help="Show framework status"
    )
    
    # Systems command
    systems_parser = subparsers.add_parser(
        "systems",
        help="Manage managed systems"
    )
    systems_parser.add_argument(
        "action",
        choices=["list", "info"],
        help="Action to perform"
    )
    systems_parser.add_argument(
        "--system-id", "-s",
        help="System ID for info action"
    )
    
    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration management"
    )
    config_parser.add_argument(
        "action",
        choices=["list", "validate", "show"],
        help="Action to perform"
    )
    config_parser.add_argument(
        "--config-file", "-f",
        help="Configuration file for validate/show actions"
    )
    
    # Metrics command
    metrics_parser = subparsers.add_parser(
        "metrics",
        help="Collect and display metrics"
    )
    metrics_parser.add_argument(
        "--system", "-s",
        help="System ID to collect metrics from"
    )
    
    # Shell command
    shell_parser = subparsers.add_parser(
        "shell",
        help="Start interactive shell"
    )
    shell_parser.add_argument(
        "--config", "-c",
        help="Configuration file or preset"
    )
    shell_parser.add_argument(
        "--log-level", "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Start live metrics dashboard"
    )
    dashboard_parser.add_argument(
        "--config", "-c",
        help="Configuration file or preset"
    )
    dashboard_parser.add_argument(
        "--system", "-s",
        help="Filter to specific system"
    )
    dashboard_parser.add_argument(
        "--duration", "-d",
        type=int,
        default=60,
        help="Dashboard duration in seconds (default: 60)"
    )
    dashboard_parser.add_argument(
        "--log-level", "-l",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING for cleaner dashboard)"
    )
    
    return parser


# ============================================================================
# Main Entry Point
# ============================================================================

async def async_main():
    """Async main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Route to appropriate command handler
    handlers = {
        "start": cmd_start,
        "status": cmd_status,
        "systems": cmd_systems,
        "config": cmd_config,
        "metrics": cmd_metrics,
        "shell": cmd_shell,
        "dashboard": cmd_dashboard,
    }
    
    handler = handlers.get(args.command)
    if handler:
        return await handler(args)
    else:
        parser.print_help()
        return 1


def main():
    """Main entry point."""
    try:
        return asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
