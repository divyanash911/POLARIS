"""
POLARIS Integration Test Harness

This module provides a comprehensive integration testing harness for POLARIS
that allows testing component interactions in a controlled environment.
"""

import asyncio
import json
import tempfile
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
import yaml

from src.domain.interfaces import ManagedSystemConnector
from src.domain.models import (
    SystemState, AdaptationAction,
    ExecutionResult, HealthStatus, 
    ExecutionStatus, MetricValue
)
from src.framework.events import TelemetryEvent, AdaptationEvent
from src.framework.polaris_framework import PolarisFramework
from src.framework.configuration.builder import ConfigurationBuilder
from src.infrastructure.di import DIContainer
from src.infrastructure.message_bus import PolarisMessageBus
from src.infrastructure.data_storage.data_store import PolarisDataStore
from src.infrastructure.observability.logging import PolarisLogger
from src.infrastructure.observability.metrics import PolarisMetricsCollector

from tests.fixtures.mock_objects import (
    MockManagedSystemConnector, MockMessageBroker, MockDataStore,
    MockMetricsCollector, MockLogger, TestDataBuilder
)


@dataclass
class IntegrationTestConfig:
    """Configuration for integration tests."""
    test_name: str
    systems: List[str] = field(default_factory=list)
    enable_real_message_broker: bool = False
    enable_real_data_store: bool = False
    enable_observability: bool = True
    test_timeout: float = 30.0
    cleanup_on_failure: bool = True
    temp_dir: Optional[Path] = None


@dataclass
class TestSystemConfig:
    """Configuration for a test system."""
    system_id: str
    connector_type: str = "mock"
    initial_state: Optional[Dict[str, Any]] = None
    failure_modes: List[str] = field(default_factory=list)
    custom_metrics: Optional[Dict[str, MetricValue]] = None


class PolarisIntegrationTestHarness:
    """
    Comprehensive integration test harness for POLARIS framework.
    
    This harness provides:
    - Component lifecycle management
    - Test environment isolation
    - Mock and real component integration
    - Event flow validation
    - Performance measurement
    - Cleanup and teardown
    """
    
    def __init__(self, config: IntegrationTestConfig):
        self.config = config
        self.temp_dir: Optional[Path] = None
        self.framework: Optional[PolarisFramework] = None
        self.di_container: Optional[DIContainer] = None
        self.test_systems: Dict[str, TestSystemConfig] = {}
        self.connectors: Dict[str, ManagedSystemConnector] = {}
        self.event_history: List[Dict[str, Any]] = []
        self.metrics_history: List[Dict[str, Any]] = []
        self.test_start_time: Optional[datetime] = None
        self.test_end_time: Optional[datetime] = None
        
        # Event tracking
        self.published_events: List[Any] = []
        self.received_events: List[Any] = []
        self.adaptation_results: List[ExecutionResult] = []
        
        # Validation callbacks
        self.event_validators: List[Callable[[Any], bool]] = []
        self.state_validators: List[Callable[[SystemState], bool]] = []
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.setup()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.teardown(exc_type is not None)
        
    async def setup(self) -> None:
        """Set up the integration test environment."""
        self.test_start_time = datetime.now()
        
        # Create temporary directory for test artifacts
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"polaris_test_{self.config.test_name}_"))
        
        # Initialize DI container
        self.di_container = DIContainer()
        
        # Set up infrastructure components
        await self._setup_infrastructure()
        
        # Set up test systems
        await self._setup_test_systems()
        
        # Initialize POLARIS framework
        await self._setup_framework()
        
        # Start event monitoring
        await self._start_event_monitoring()
        
    async def teardown(self, failed: bool = False) -> None:
        """Tear down the test environment."""
        self.test_end_time = datetime.now()
        
        try:
            # Stop framework if running
            if self.framework:
                await self.framework.stop()
            
            # Disconnect all connectors
            for connector in self.connectors.values():
                try:
                    if hasattr(connector, 'connected') and connector.connected:
                        await connector.disconnect()
                except Exception:
                    pass  # Ignore cleanup errors
            
            # Clean up infrastructure
            await self._cleanup_infrastructure()
            
            # Remove temporary directory
            if self.temp_dir and self.temp_dir.exists():
                if not failed or self.config.cleanup_on_failure:
                    shutil.rmtree(self.temp_dir)
                    
        except Exception as e:
            # Log cleanup errors but don't fail the test
            print(f"Warning: Cleanup error in integration test harness: {e}")
    
    async def _setup_infrastructure(self) -> None:
        """Set up infrastructure components based on configuration."""
        # Message broker
        if self.config.enable_real_message_broker:
            # Use real NATS broker (requires NATS server running)
            from src.infrastructure.message_bus import NATSMessageBroker
            message_broker = NATSMessageBroker("nats://localhost:4222")
        else:
            message_broker = MockMessageBroker()
        
        self.di_container.register("message_broker", message_broker)
        
        # Data store
        if self.config.enable_real_data_store:
            # Use real data store (in-memory for tests)
            data_store = PolarisDataStore(connection_string="memory://")
        else:
            data_store = MockDataStore()
        
        self.di_container.register("data_store", data_store)
        
        # Observability components
        if self.config.enable_observability:
            logger = PolarisLogger(level="DEBUG")
            metrics_collector = PolarisMetricsCollector()
        else:
            logger = MockLogger()
            metrics_collector = MockMetricsCollector()
        
        self.di_container.register("logger", logger)
        self.di_container.register("metrics_collector", metrics_collector)
        
        # Connect infrastructure
        await message_broker.connect()
        await data_store.connect()
    
    async def _setup_test_systems(self) -> None:
        """Set up test systems and their connectors."""
        for system_id in self.config.systems:
            system_config = self.test_systems.get(system_id, TestSystemConfig(system_id=system_id))
            
            # Create connector based on type
            if system_config.connector_type == "mock":
                connector = MockManagedSystemConnector(system_id)
                
                # Configure failure modes
                for failure_mode in system_config.failure_modes:
                    if failure_mode == "connection":
                        connector.should_fail_connection = True
                    elif failure_mode == "metrics":
                        connector.should_fail_metrics = True
                    elif failure_mode == "execution":
                        connector.should_fail_execution = True
            else:
                # For real connectors, would load from plugin registry
                raise NotImplementedError(f"Connector type {system_config.connector_type} not implemented in test harness")
            
            self.connectors[system_id] = connector
    
    async def _setup_framework(self) -> None:
        """Initialize and configure the POLARIS framework."""
        # Create test configuration
        config_data = {
            "framework": {
                "name": f"polaris_integration_test_{self.config.test_name}",
                "log_level": "DEBUG"
            },
            "adapters": {
                "monitor": {
                    "collection_interval": 0.1,  # Fast collection for tests
                    "batch_size": 5
                },
                "execution": {
                    "timeout": 5.0,
                    "retry_attempts": 2
                }
            },
            "systems": {}
        }
        
        # Add system configurations
        for system_id in self.config.systems:
            config_data["systems"][system_id] = {
                "enabled": True,
                "connector_type": "mock"
            }
        
        # Write configuration to temp file
        config_file = self.temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Build configuration
        configuration = (ConfigurationBuilder()
                        .add_yaml_source(str(config_file))
                        .build())
        
        # Initialize framework
        self.framework = PolarisFramework(configuration, self.di_container)
        
        # Register test connectors
        for system_id, connector in self.connectors.items():
            self.framework.plugin_registry.register_connector(system_id, connector)
        
        # Start framework
        await self.framework.start()
    
    async def _start_event_monitoring(self) -> None:
        """Start monitoring events for validation."""
        message_broker = self.di_container.get("message_broker")
        
        # Subscribe to all telemetry events
        await message_broker.subscribe("telemetry.*", self._handle_telemetry_event)
        
        # Subscribe to all adaptation events
        await message_broker.subscribe("adaptation.*", self._handle_adaptation_event)
        
        # Subscribe to execution results
        await message_broker.subscribe("execution.*", self._handle_execution_result)
    
    async def _handle_telemetry_event(self, message: bytes) -> None:
        """Handle received telemetry events."""
        try:
            event_data = json.loads(message.decode())
            self.received_events.append({
                "type": "telemetry",
                "data": event_data,
                "timestamp": datetime.now()
            })
            
            # Run event validators
            for validator in self.event_validators:
                validator(event_data)
                
        except Exception as e:
            print(f"Error handling telemetry event: {e}")
    
    async def _handle_adaptation_event(self, message: bytes) -> None:
        """Handle received adaptation events."""
        try:
            event_data = json.loads(message.decode())
            self.received_events.append({
                "type": "adaptation",
                "data": event_data,
                "timestamp": datetime.now()
            })
            
        except Exception as e:
            print(f"Error handling adaptation event: {e}")
    
    async def _handle_execution_result(self, message: bytes) -> None:
        """Handle execution result events."""
        try:
            result_data = json.loads(message.decode())
            self.received_events.append({
                "type": "execution_result",
                "data": result_data,
                "timestamp": datetime.now()
            })
            
        except Exception as e:
            print(f"Error handling execution result: {e}")
    
    async def _cleanup_infrastructure(self) -> None:
        """Clean up infrastructure components."""
        try:
            message_broker = self.di_container.get("message_broker")
            if message_broker:
                await message_broker.disconnect()
                
            data_store = self.di_container.get("data_store")
            if data_store:
                await data_store.disconnect()
                
        except Exception as e:
            print(f"Error during infrastructure cleanup: {e}")
    
    # Test system management methods
    
    def add_test_system(self, system_config: TestSystemConfig) -> None:
        """Add a test system configuration."""
        self.test_systems[system_config.system_id] = system_config
        if system_config.system_id not in self.config.systems:
            self.config.systems.append(system_config.system_id)
    
    def configure_system_failure(self, system_id: str, failure_modes: List[str]) -> None:
        """Configure failure modes for a test system."""
        if system_id in self.test_systems:
            self.test_systems[system_id].failure_modes = failure_modes
        else:
            self.test_systems[system_id] = TestSystemConfig(
                system_id=system_id,
                failure_modes=failure_modes
            )
    
    async def inject_telemetry(self, system_id: str, metrics: Dict[str, MetricValue]) -> None:
        """Inject telemetry data for a specific system."""
        if system_id not in self.connectors:
            raise ValueError(f"System {system_id} not configured in test harness")
        
        connector = self.connectors[system_id]
        if isinstance(connector, MockManagedSystemConnector):
            # Override the metrics that will be returned
            connector.collected_metrics = [metrics]
        
        # Trigger telemetry collection
        collected_metrics = await connector.collect_metrics()
        
        # Create telemetry event
        event = TelemetryEvent(
            system_id=system_id,
            metrics=collected_metrics,
            timestamp=datetime.now(),
            correlation_id=f"test_{len(self.published_events)}"
        )
        
        # Publish event
        message_broker = self.di_container.get("message_broker")
        event_data = json.dumps({
            "system_id": event.system_id,
            "metrics": {k: {"value": v.value, "unit": v.unit, "timestamp": v.timestamp.isoformat()} 
                       for k, v in event.metrics.items()},
            "timestamp": event.timestamp.isoformat(),
            "correlation_id": event.correlation_id
        }).encode()
        
        await message_broker.publish(f"telemetry.{system_id}", event_data)
        self.published_events.append(event)
    
    async def trigger_adaptation(self, system_id: str, actions: List[AdaptationAction]) -> None:
        """Trigger an adaptation for a specific system."""
        event = AdaptationEvent(
            system_id=system_id,
            trigger_reason="test_triggered",
            actions=actions,
            timestamp=datetime.now(),
            correlation_id=f"adaptation_{len(self.published_events)}"
        )
        
        # Publish adaptation event
        message_broker = self.di_container.get("message_broker")
        event_data = json.dumps({
            "system_id": event.system_id,
            "trigger_reason": event.trigger_reason,
            "actions": [
                {
                    "action_id": action.action_id,
                    "action_type": action.action_type,
                    "target_system": action.target_system,
                    "parameters": action.parameters
                }
                for action in event.actions
            ],
            "timestamp": event.timestamp.isoformat(),
            "correlation_id": event.correlation_id
        }).encode()
        
        await message_broker.publish(f"adaptation.{system_id}", event_data)
        self.published_events.append(event)
    
    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        """Execute an adaptation action on a target system."""
        if action.target_system not in self.connectors:
            raise ValueError(f"Target system {action.target_system} not configured")
        
        connector = self.connectors[action.target_system]
        result = await connector.execute_action(action)
        self.adaptation_results.append(result)
        
        return result
    
    # Validation and assertion methods
    
    def add_event_validator(self, validator: Callable[[Any], bool]) -> None:
        """Add an event validator function."""
        self.event_validators.append(validator)
    
    def add_state_validator(self, validator: Callable[[SystemState], bool]) -> None:
        """Add a state validator function."""
        self.state_validators.append(validator)
    
    async def wait_for_events(self, event_type: str, count: int, timeout: float = 5.0) -> List[Dict[str, Any]]:
        """Wait for a specific number of events of a given type."""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            matching_events = [
                event for event in self.received_events
                if event["type"] == event_type
            ]
            
            if len(matching_events) >= count:
                return matching_events[:count]
            
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Timeout waiting for {count} {event_type} events")
    
    async def wait_for_system_state(self, system_id: str, condition: Callable[[SystemState], bool], timeout: float = 5.0) -> SystemState:
        """Wait for a system to reach a specific state."""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            # Get current system state
            connector = self.connectors.get(system_id)
            if connector and hasattr(connector, 'get_current_state'):
                state = await connector.get_current_state()
                if condition(state):
                    return state
            
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Timeout waiting for system {system_id} to reach expected state")
    
    def assert_event_published(self, event_type: str, system_id: str = None) -> None:
        """Assert that an event of a specific type was published."""
        matching_events = [
            event for event in self.received_events
            if event["type"] == event_type and (system_id is None or event["data"].get("system_id") == system_id)
        ]
        
        assert len(matching_events) > 0, f"No {event_type} events found for system {system_id}"
    
    def assert_adaptation_executed(self, action_id: str) -> None:
        """Assert that a specific adaptation action was executed."""
        matching_results = [
            result for result in self.adaptation_results
            if result.action_id == action_id
        ]
        
        assert len(matching_results) > 0, f"No execution results found for action {action_id}"
        assert matching_results[0].status == ExecutionStatus.SUCCESS, f"Action {action_id} failed: {matching_results[0].result_data}"
    
    def assert_no_errors_logged(self) -> None:
        """Assert that no error-level logs were recorded."""
        logger = self.di_container.get("logger")
        if hasattr(logger, 'logs'):
            error_logs = [log for log in logger.logs if log["level"] == "ERROR"]
            assert len(error_logs) == 0, f"Found {len(error_logs)} error logs: {error_logs}"
    
    def get_test_metrics(self) -> Dict[str, Any]:
        """Get comprehensive test execution metrics."""
        duration = None
        if self.test_start_time and self.test_end_time:
            duration = (self.test_end_time - self.test_start_time).total_seconds()
        
        return {
            "test_name": self.config.test_name,
            "duration": duration,
            "systems_tested": len(self.config.systems),
            "events_published": len(self.published_events),
            "events_received": len(self.received_events),
            "adaptations_executed": len(self.adaptation_results),
            "successful_adaptations": len([r for r in self.adaptation_results if r.status == ExecutionStatus.SUCCESS]),
            "failed_adaptations": len([r for r in self.adaptation_results if r.status == ExecutionStatus.FAILED]),
            "temp_dir": str(self.temp_dir) if self.temp_dir else None
        }
    
    def generate_test_report(self) -> str:
        """Generate a comprehensive test report."""
        metrics = self.get_test_metrics()
        
        report_lines = [
            f"POLARIS Integration Test Report: {self.config.test_name}",
            "=" * 60,
            "",
            f"Test Duration: {metrics['duration']:.2f}s" if metrics['duration'] else "Test Duration: N/A",
            f"Systems Tested: {metrics['systems_tested']}",
            f"Events Published: {metrics['events_published']}",
            f"Events Received: {metrics['events_received']}",
            f"Adaptations Executed: {metrics['adaptations_executed']}",
            f"Successful Adaptations: {metrics['successful_adaptations']}",
            f"Failed Adaptations: {metrics['failed_adaptations']}",
            "",
            "System Details:",
            "-" * 20
        ]
        
        for system_id in self.config.systems:
            connector = self.connectors.get(system_id)
            if connector and isinstance(connector, MockManagedSystemConnector):
                report_lines.extend([
                    f"• {system_id}:",
                    f"  - Connection attempts: {connector.connection_attempts}",
                    f"  - Metrics collected: {len(connector.collected_metrics)}",
                    f"  - Actions executed: {len(connector.executed_actions)}"
                ])
        
        if self.received_events:
            report_lines.extend([
                "",
                "Event Timeline:",
                "-" * 15
            ])
            
            for event in self.received_events[-10:]:  # Show last 10 events
                timestamp = event["timestamp"].strftime("%H:%M:%S.%f")[:-3]
                event_type = event["type"]
                system_id = event["data"].get("system_id", "unknown")
                report_lines.append(f"  {timestamp} - {event_type} from {system_id}")
        
        return "\n".join(report_lines)


# Helper functions for creating test harnesses

def create_simple_harness(test_name: str, systems: List[str]) -> PolarisIntegrationTestHarness:
    """Create a simple integration test harness with mock components."""
    config = IntegrationTestConfig(
        test_name=test_name,
        systems=systems,
        enable_real_message_broker=False,
        enable_real_data_store=False,
        enable_observability=True
    )
    
    return PolarisIntegrationTestHarness(config)


def create_performance_harness(test_name: str, systems: List[str]) -> PolarisIntegrationTestHarness:
    """Create an integration test harness optimized for performance testing."""
    config = IntegrationTestConfig(
        test_name=test_name,
        systems=systems,
        enable_real_message_broker=True,  # Use real broker for performance testing
        enable_real_data_store=True,
        enable_observability=True,
        test_timeout=60.0  # Longer timeout for performance tests
    )
    
    return PolarisIntegrationTestHarness(config)


def create_failure_testing_harness(test_name: str, systems: List[str], failure_modes: Dict[str, List[str]]) -> PolarisIntegrationTestHarness:
    """Create an integration test harness configured for failure testing."""
    config = IntegrationTestConfig(
        test_name=test_name,
        systems=systems,
        enable_real_message_broker=False,
        enable_real_data_store=False,
        enable_observability=True,
        cleanup_on_failure=False  # Keep artifacts for failure analysis
    )
    
    harness = PolarisIntegrationTestHarness(config)
    
    # Configure failure modes for each system
    for system_id, modes in failure_modes.items():
        harness.configure_system_failure(system_id, modes)
    
    return harness