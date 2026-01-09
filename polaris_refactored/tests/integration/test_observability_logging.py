"""
Observability and Logging Tests for Mock External System Testing

This module provides comprehensive tests for observability and logging functionality
in the context of mock external system testing, validating:
- Telemetry logging verification (Requirements 7.1)
- Adaptation logging verification (Requirements 7.2)
- Error logging verification (Requirements 7.3)
- Metrics export verification (Requirements 7.4)

Requirements: 7.1, 7.2, 7.3, 7.4
"""

import asyncio
import pytest
import time
import json
import tempfile
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch

# Ensure src is in path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from domain.models import (
    MetricValue, SystemState, AdaptationAction, 
    ExecutionResult, HealthStatus, ExecutionStatus
)
from framework.events import TelemetryEvent, AdaptationEvent
from infrastructure.observability.factory import (
    get_polaris_logger, configure_logging, reset_logging,
    get_framework_logger, get_infrastructure_logger, get_adapter_logger
)
from framework.configuration.models import LoggingConfiguration
from infrastructure.observability.logging import LogLevel

# Import fixtures from local path
tests_path = Path(__file__).parent.parent
if str(tests_path) not in sys.path:
    sys.path.insert(0, str(tests_path))

from fixtures.logging_fixtures import (
    integration_logging_setup, log_assertions, log_file_reader, capture_logs
)

# Import harness from local path
harness_path = Path(__file__).parent / "harness"
if str(harness_path) not in sys.path:
    sys.path.insert(0, str(harness_path))

from polaris_integration_test_harness import (
    PolarisIntegrationTestHarness, create_simple_harness
)

# Import mock system components
mock_system_parent_path = Path(__file__).parent.parent.parent / "mock_external_system"
if str(mock_system_parent_path) not in sys.path:
    sys.path.insert(0, str(mock_system_parent_path))

plugins_path = Path(__file__).parent.parent.parent / "plugins"
if str(plugins_path) not in sys.path:
    sys.path.insert(0, str(plugins_path))

from src.server import MockSystemServer
from src.state_manager import StateManager
from mock_system.connector import MockSystemConnector


@pytest.mark.integration
class TestTelemetryLoggingVerification:
    """Test telemetry logging verification (Requirements 7.1)."""
    
    @pytest.fixture
    async def mock_system_with_logging(self):
        """Set up mock system with logging configuration."""
        # Configure logging for telemetry verification
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = f.name
        
        config = LoggingConfiguration(
            level="DEBUG",
            format="json",
            output="file",
            file_path=log_file
        )
        configure_logging(config)
        
        # Create mock system
        test_config = {
            "server": {"host": "localhost", "port": 5010, "max_connections": 10},
            "baseline_metrics": {
                "cpu_usage": 25.0, "memory_usage": 1024.0, "response_time": 80.0,
                "throughput": 60.0, "error_rate": 0.3, "active_connections": 5, "capacity": 3
            },
            "simulation": {"noise_factor": 0.05, "update_interval": 0.1},
            "capacity": {"min_capacity": 1, "max_capacity": 10, "scale_up_increment": 1, "scale_down_increment": 1}
        }
        
        state_manager = StateManager(initial_config=test_config)
        server = MockSystemServer("localhost", 5010, state_manager, max_connections=10)
        await server.start()
        
        # Create connector
        connector_config = {
            "system_name": "telemetry_test_system",
            "connection": {"host": "localhost", "port": 5010},
            "implementation": {"timeout": 5.0, "max_retries": 2, "retry_base_delay": 0.1}
        }
        connector = MockSystemConnector(connector_config)
        await connector.connect()
        
        try:
            yield server, connector, log_file
        finally:
            await connector.disconnect()
            await server.stop()
            reset_logging()
            try:
                Path(log_file).unlink()
            except FileNotFoundError:
                pass 
   
    @pytest.mark.asyncio
    async def test_telemetry_events_logged_with_timestamps_and_system_ids(self, mock_system_with_logging, log_file_reader):
        """Test that telemetry events are logged with timestamps and system IDs."""
        server, connector, log_file = mock_system_with_logging
        
        # Get logger for telemetry collection
        telemetry_logger = get_adapter_logger("telemetry_test_system")
        
        # Collect metrics multiple times to generate telemetry events
        for i in range(3):
            telemetry_logger.info(
                f"Collecting telemetry from system",
                extra={
                    "system_id": "telemetry_test_system",
                    "collection_attempt": i + 1,
                    "event_type": "telemetry_collection"
                }
            )
            
            # Actually collect metrics to generate real telemetry
            metrics = await connector.collect_metrics()
            
            # Log the collected metrics
            telemetry_logger.info(
                "Telemetry collected successfully",
                extra={
                    "system_id": "telemetry_test_system",
                    "metrics_count": len(metrics),
                    "cpu_usage": metrics["cpu_usage"].value,
                    "memory_usage": metrics["memory_usage"].value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "telemetry_data"
                }
            )
            
            await asyncio.sleep(0.1)
        
        # Wait for logs to be written
        await asyncio.sleep(0.2)
        
        # Read and verify log entries
        log_records = log_file_reader(log_file, "json")
        
        # Filter telemetry-related logs
        telemetry_logs = [
            record for record in log_records 
            if record.get('extra', {}).get('event_type') in ['telemetry_collection', 'telemetry_data']
        ]
        
        # Verify we have telemetry logs
        assert len(telemetry_logs) >= 6, f"Expected at least 6 telemetry logs, got {len(telemetry_logs)}"
        
        # Verify each telemetry log has required fields
        for log_record in telemetry_logs:
            # Check timestamp is present and valid
            assert 'timestamp' in log_record, "Telemetry log missing timestamp"
            timestamp_str = log_record['timestamp']
            # Verify timestamp is parseable - handle both Z and +00:00 formats
            # Replace Z only if it's at the end and there's no timezone offset already
            if timestamp_str.endswith('Z') and '+' not in timestamp_str:
                timestamp_str = timestamp_str.replace('Z', '+00:00')
            datetime.fromisoformat(timestamp_str)
            
            # Check system ID is present
            extra = log_record.get('extra', {})
            assert 'system_id' in extra, "Telemetry log missing system_id"
            assert extra['system_id'] == "telemetry_test_system", f"Unexpected system_id: {extra['system_id']}"
            
            # Check event type is present
            assert 'event_type' in extra, "Telemetry log missing event_type"
            assert extra['event_type'] in ['telemetry_collection', 'telemetry_data'], f"Unexpected event_type: {extra['event_type']}"
        
        # Verify telemetry data logs have metric information
        data_logs = [log for log in telemetry_logs if log.get('extra', {}).get('event_type') == 'telemetry_data']
        assert len(data_logs) >= 3, f"Expected at least 3 telemetry data logs, got {len(data_logs)}"
        
        for data_log in data_logs:
            extra = data_log.get('extra', {})
            assert 'metrics_count' in extra, "Telemetry data log missing metrics_count"
            assert extra['metrics_count'] > 0, "Telemetry data log should have positive metrics_count"
            assert 'cpu_usage' in extra, "Telemetry data log missing cpu_usage"
            assert 'memory_usage' in extra, "Telemetry data log missing memory_usage"
    
    @pytest.mark.asyncio
    async def test_telemetry_log_format_includes_required_fields(self, mock_system_with_logging, log_file_reader):
        """Test that telemetry log format includes all required fields."""
        server, connector, log_file = mock_system_with_logging
        
        # Get logger for telemetry
        telemetry_logger = get_adapter_logger("telemetry_test_system")
        
        # Generate a comprehensive telemetry event
        metrics = await connector.collect_metrics()
        
        telemetry_logger.info(
            "Comprehensive telemetry event",
            extra={
                "system_id": "telemetry_test_system",
                "event_type": "telemetry_comprehensive",
                "collection_timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": {
                    "cpu_usage": metrics["cpu_usage"].value,
                    "memory_usage": metrics["memory_usage"].value,
                    "response_time": metrics["response_time"].value,
                    "throughput": metrics["throughput"].value,
                    "error_rate": metrics["error_rate"].value,
                    "capacity": metrics["capacity"].value
                },
                "collection_duration_ms": 150,
                "collection_status": "success"
            }
        )
        
        await asyncio.sleep(0.1)
        
        # Read and verify log format
        log_records = log_file_reader(log_file, "json")
        comprehensive_logs = [
            record for record in log_records 
            if record.get('extra', {}).get('event_type') == 'telemetry_comprehensive'
        ]
        
        assert len(comprehensive_logs) >= 1, "Should have at least one comprehensive telemetry log"
        
        log_record = comprehensive_logs[0]
        
        # Verify standard log fields
        required_standard_fields = ['timestamp', 'level', 'logger', 'message']
        for field in required_standard_fields:
            assert field in log_record, f"Missing standard field: {field}"
        
        # Verify telemetry-specific fields
        extra = log_record.get('extra', {})
        required_telemetry_fields = [
            'system_id', 'event_type', 'collection_timestamp', 
            'metrics', 'collection_duration_ms', 'collection_status'
        ]
        for field in required_telemetry_fields:
            assert field in extra, f"Missing telemetry field: {field}"
        
        # Verify metrics structure
        metrics_data = extra['metrics']
        expected_metrics = ['cpu_usage', 'memory_usage', 'response_time', 'throughput', 'error_rate', 'capacity']
        for metric in expected_metrics:
            assert metric in metrics_data, f"Missing metric in telemetry: {metric}"
            assert isinstance(metrics_data[metric], (int, float)), f"Metric {metric} should be numeric"


@pytest.mark.integration
class TestAdaptationLoggingVerification:
    """Test adaptation logging verification (Requirements 7.2)."""
    
    @pytest.fixture
    async def mock_system_with_adaptation_logging(self):
        """Set up mock system with adaptation logging configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = f.name
        
        config = LoggingConfiguration(
            level="DEBUG",
            format="json",
            output="file",
            file_path=log_file
        )
        configure_logging(config)
        
        # Create mock system
        test_config = {
            "server": {"host": "localhost", "port": 5011, "max_connections": 10},
            "baseline_metrics": {
                "cpu_usage": 75.0, "memory_usage": 2048.0, "response_time": 150.0,
                "throughput": 30.0, "error_rate": 2.0, "active_connections": 15, "capacity": 2
            },
            "simulation": {"noise_factor": 0.05, "update_interval": 0.1},
            "capacity": {"min_capacity": 1, "max_capacity": 10, "scale_up_increment": 1, "scale_down_increment": 1}
        }
        
        state_manager = StateManager(initial_config=test_config)
        server = MockSystemServer("localhost", 5011, state_manager, max_connections=10)
        await server.start()
        
        connector_config = {
            "system_name": "adaptation_test_system",
            "connection": {"host": "localhost", "port": 5011},
            "implementation": {"timeout": 5.0, "max_retries": 2, "retry_base_delay": 0.1}
        }
        connector = MockSystemConnector(connector_config)
        await connector.connect()
        
        try:
            yield server, connector, log_file
        finally:
            await connector.disconnect()
            await server.stop()
            reset_logging()
            try:
                Path(log_file).unlink()
            except FileNotFoundError:
                pass
    
    @pytest.mark.asyncio
    async def test_adaptation_events_logged_with_complete_information(self, mock_system_with_adaptation_logging, log_file_reader):
        """Test that adaptation events are logged with complete information."""
        server, connector, log_file = mock_system_with_adaptation_logging
        
        # Get loggers for different adaptation phases
        controller_logger = get_framework_logger("adaptive_controller")
        adapter_logger = get_adapter_logger("adaptation_test_system")
        
        # Get initial metrics to establish triggering conditions
        initial_metrics = await connector.collect_metrics()
        initial_cpu = initial_metrics["cpu_usage"].value
        
        # Log adaptation trigger conditions
        controller_logger.info(
            "Adaptation triggered due to threshold violation",
            extra={
                "system_id": "adaptation_test_system",
                "event_type": "adaptation_triggered",
                "trigger_timestamp": datetime.now(timezone.utc).isoformat(),
                "triggering_conditions": {
                    "cpu_usage": initial_cpu,
                    "threshold": 70.0,
                    "violation_type": "high_cpu"
                },
                "selected_action": "SCALE_UP",
                "action_reason": "cpu_usage_above_threshold"
            }
        )
        
        # Create and execute adaptation action
        adaptation_action = AdaptationAction(
            action_id="adaptation_log_test_001",
            action_type="SCALE_UP",
            target_system="adaptation_test_system",
            parameters={"increment": 1, "reason": "high_cpu_threshold_violation"},
            priority=1,
            created_at=datetime.now(timezone.utc)
        )
        
        # Log action execution start
        adapter_logger.info(
            "Executing adaptation action",
            extra={
                "system_id": "adaptation_test_system",
                "event_type": "adaptation_execution_start",
                "action_id": adaptation_action.action_id,
                "action_type": adaptation_action.action_type,
                "action_parameters": adaptation_action.parameters,
                "execution_timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        # Execute the action
        result = await connector.execute_action(adaptation_action)
        
        # Log action execution result
        adapter_logger.info(
            "Adaptation action completed",
            extra={
                "system_id": "adaptation_test_system",
                "event_type": "adaptation_execution_complete",
                "action_id": adaptation_action.action_id,
                "execution_status": result.status.value,
                "execution_time_ms": result.execution_time_ms,
                "result_data": result.result_data,
                "error_message": result.error_message,
                "completion_timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        # Get post-adaptation metrics
        await asyncio.sleep(0.1)
        post_metrics = await connector.collect_metrics()
        
        # Log adaptation outcome
        controller_logger.info(
            "Adaptation outcome assessed",
            extra={
                "system_id": "adaptation_test_system",
                "event_type": "adaptation_outcome",
                "action_id": adaptation_action.action_id,
                "metrics_before": {
                    "cpu_usage": initial_cpu,
                    "capacity": initial_metrics["capacity"].value
                },
                "metrics_after": {
                    "cpu_usage": post_metrics["cpu_usage"].value,
                    "capacity": post_metrics["capacity"].value
                },
                "outcome_assessment": "successful" if result.status == ExecutionStatus.SUCCESS else "failed",
                "assessment_timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        await asyncio.sleep(0.2)
        
        # Verify adaptation logs
        log_records = log_file_reader(log_file, "json")
        adaptation_logs = [
            record for record in log_records 
            if record.get('extra', {}).get('event_type', '').startswith('adaptation')
        ]
        
        assert len(adaptation_logs) >= 4, f"Expected at least 4 adaptation logs, got {len(adaptation_logs)}"
        
        # Verify trigger log
        trigger_logs = [log for log in adaptation_logs if log.get('extra', {}).get('event_type') == 'adaptation_triggered']
        assert len(trigger_logs) >= 1, "Should have adaptation trigger log"
        
        trigger_log = trigger_logs[0]
        trigger_extra = trigger_log.get('extra', {})
        assert 'triggering_conditions' in trigger_extra, "Trigger log missing triggering_conditions"
        assert 'selected_action' in trigger_extra, "Trigger log missing selected_action"
        assert trigger_extra['triggering_conditions']['cpu_usage'] == initial_cpu
        
        # Verify execution logs
        execution_logs = [log for log in adaptation_logs if 'execution' in log.get('extra', {}).get('event_type', '')]
        assert len(execution_logs) >= 2, "Should have execution start and complete logs"
        
        # Verify outcome log
        outcome_logs = [log for log in adaptation_logs if log.get('extra', {}).get('event_type') == 'adaptation_outcome']
        assert len(outcome_logs) >= 1, "Should have adaptation outcome log"
        
        outcome_log = outcome_logs[0]
        outcome_extra = outcome_log.get('extra', {})
        assert 'metrics_before' in outcome_extra, "Outcome log missing metrics_before"
        assert 'metrics_after' in outcome_extra, "Outcome log missing metrics_after"
        assert 'outcome_assessment' in outcome_extra, "Outcome log missing outcome_assessment"
    
    @pytest.mark.asyncio
    async def test_adaptation_logs_include_triggering_conditions_actions_and_results(self, mock_system_with_adaptation_logging, log_file_reader):
        """Test that adaptation logs include triggering conditions, actions, and results."""
        server, connector, log_file = mock_system_with_adaptation_logging
        
        controller_logger = get_framework_logger("adaptive_controller")
        
        # Simulate multiple adaptation scenarios
        scenarios = [
            {
                "trigger": "high_cpu",
                "action_type": "SCALE_UP",
                "conditions": {"cpu_usage": 85.0, "threshold": 80.0}
            },
            {
                "trigger": "high_memory",
                "action_type": "OPTIMIZE_CONFIG",
                "conditions": {"memory_usage": 3000.0, "threshold": 2500.0}
            }
        ]
        
        for i, scenario in enumerate(scenarios):
            # Log comprehensive adaptation event
            controller_logger.info(
                f"Complete adaptation cycle {i+1}",
                extra={
                    "system_id": "adaptation_test_system",
                    "event_type": "adaptation_complete_cycle",
                    "cycle_id": f"cycle_{i+1}",
                    "triggering_conditions": scenario["conditions"],
                    "trigger_type": scenario["trigger"],
                    "selected_action": {
                        "action_type": scenario["action_type"],
                        "action_id": f"action_{i+1}",
                        "parameters": {"reason": scenario["trigger"]}
                    },
                    "execution_result": {
                        "status": "SUCCESS",
                        "execution_time_ms": 120 + i * 10,
                        "result_summary": f"Action {scenario['action_type']} completed successfully"
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        await asyncio.sleep(0.1)
        
        # Verify comprehensive adaptation logs
        log_records = log_file_reader(log_file, "json")
        cycle_logs = [
            record for record in log_records 
            if record.get('extra', {}).get('event_type') == 'adaptation_complete_cycle'
        ]
        
        assert len(cycle_logs) >= 2, f"Expected at least 2 adaptation cycle logs, got {len(cycle_logs)}"
        
        for i, cycle_log in enumerate(cycle_logs):
            extra = cycle_log.get('extra', {})
            
            # Verify triggering conditions are present and complete
            assert 'triggering_conditions' in extra, f"Cycle {i+1} missing triggering_conditions"
            conditions = extra['triggering_conditions']
            assert len(conditions) >= 2, f"Cycle {i+1} should have at least 2 condition fields"
            
            # Verify action information is present and complete
            assert 'selected_action' in extra, f"Cycle {i+1} missing selected_action"
            action = extra['selected_action']
            assert 'action_type' in action, f"Cycle {i+1} action missing action_type"
            assert 'action_id' in action, f"Cycle {i+1} action missing action_id"
            assert 'parameters' in action, f"Cycle {i+1} action missing parameters"
            
            # Verify execution result is present and complete
            assert 'execution_result' in extra, f"Cycle {i+1} missing execution_result"
            result = extra['execution_result']
            assert 'status' in result, f"Cycle {i+1} result missing status"
            assert 'execution_time_ms' in result, f"Cycle {i+1} result missing execution_time_ms"
            assert 'result_summary' in result, f"Cycle {i+1} result missing result_summary"


if __name__ == "__main__":
    # Run tests directly for debugging
    pytest.main([__file__, "-v", "-s"])


@pytest.mark.integration
class TestErrorLoggingVerification:
    """Test error logging verification (Requirements 7.3)."""
    
    @pytest.fixture
    async def mock_system_with_error_logging(self):
        """Set up mock system with error logging configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = f.name
        
        config = LoggingConfiguration(
            level="DEBUG",
            format="json",
            output="file",
            file_path=log_file
        )
        configure_logging(config)
        
        # Create mock system
        test_config = {
            "server": {"host": "localhost", "port": 5012, "max_connections": 10},
            "baseline_metrics": {
                "cpu_usage": 30.0, "memory_usage": 1024.0, "response_time": 100.0,
                "throughput": 50.0, "error_rate": 0.5, "active_connections": 10, "capacity": 5
            },
            "simulation": {"noise_factor": 0.05, "update_interval": 0.1},
            "capacity": {"min_capacity": 1, "max_capacity": 10, "scale_up_increment": 1, "scale_down_increment": 1}
        }
        
        state_manager = StateManager(initial_config=test_config)
        server = MockSystemServer("localhost", 5012, state_manager, max_connections=10)
        await server.start()
        
        connector_config = {
            "system_name": "error_test_system",
            "connection": {"host": "localhost", "port": 5012},
            "implementation": {"timeout": 5.0, "max_retries": 2, "retry_base_delay": 0.1}
        }
        connector = MockSystemConnector(connector_config)
        await connector.connect()
        
        try:
            yield server, connector, log_file
        finally:
            await connector.disconnect()
            await server.stop()
            reset_logging()
            try:
                Path(log_file).unlink()
            except FileNotFoundError:
                pass
    
    @pytest.mark.asyncio
    async def test_errors_logged_with_stack_traces_and_context(self, mock_system_with_error_logging, log_file_reader):
        """Test that errors are logged with stack traces and context."""
        server, connector, log_file = mock_system_with_error_logging
        
        error_logger = get_adapter_logger("error_test_system")
        
        # Simulate various error scenarios
        error_scenarios = [
            {
                "error_type": "ConnectionError",
                "error_message": "Failed to connect to external service",
                "context": {"host": "external.service.com", "port": 8080, "timeout": 5.0}
            },
            {
                "error_type": "ValidationError", 
                "error_message": "Invalid action parameters",
                "context": {"action_type": "INVALID_ACTION", "parameters": {"invalid": "param"}}
            },
            {
                "error_type": "TimeoutError",
                "error_message": "Operation timed out",
                "context": {"operation": "metric_collection", "timeout_seconds": 10.0}
            }
        ]
        
        for i, scenario in enumerate(error_scenarios):
            try:
                # Simulate the error condition
                if scenario["error_type"] == "ConnectionError":
                    raise ConnectionError(scenario["error_message"])
                elif scenario["error_type"] == "ValidationError":
                    raise ValueError(scenario["error_message"])
                elif scenario["error_type"] == "TimeoutError":
                    raise TimeoutError(scenario["error_message"])
            except Exception as e:
                # Log the error with full context
                error_logger.error(
                    f"Error occurred in scenario {i+1}",
                    extra={
                        "system_id": "error_test_system",
                        "event_type": "error_occurred",
                        "error_scenario": i + 1,
                        "error_type": scenario["error_type"],
                        "error_message": scenario["error_message"],
                        "error_context": scenario["context"],
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    },
                    exc_info=e
                )
        
        await asyncio.sleep(0.1)
        
        # Verify error logs
        log_records = log_file_reader(log_file, "json")
        error_logs = [
            record for record in log_records 
            if record.get('level') == 'ERROR' and record.get('extra', {}).get('event_type') == 'error_occurred'
        ]
        
        assert len(error_logs) >= 3, f"Expected at least 3 error logs, got {len(error_logs)}"
        
        for i, error_log in enumerate(error_logs):
            # Verify error log has required fields
            assert 'timestamp' in error_log, f"Error log {i+1} missing timestamp"
            assert 'level' in error_log and error_log['level'] == 'ERROR', f"Error log {i+1} should be ERROR level"
            assert 'message' in error_log, f"Error log {i+1} missing message"
            
            # Verify error-specific fields
            extra = error_log.get('extra', {})
            assert 'error_type' in extra, f"Error log {i+1} missing error_type"
            assert 'error_message' in extra, f"Error log {i+1} missing error_message"
            assert 'error_context' in extra, f"Error log {i+1} missing error_context"
            
            # Verify stack trace information is present
            assert 'exception' in extra, f"Error log {i+1} missing exception info"
            exception_info = extra['exception']
            assert 'type' in exception_info, f"Error log {i+1} exception missing type"
            assert 'message' in exception_info, f"Error log {i+1} exception missing message"
            assert 'module' in exception_info, f"Error log {i+1} exception missing module"
            
            # Verify exception information is meaningful
            assert exception_info['type'] in ['ConnectionError', 'ValueError', 'TimeoutError'], f"Error log {i+1} unexpected exception type: {exception_info['type']}"
            assert len(exception_info['message']) > 0, f"Error log {i+1} should have non-empty exception message"
    
    @pytest.mark.asyncio
    async def test_error_logs_include_error_type_and_context(self, mock_system_with_error_logging, log_file_reader):
        """Test that error logs include error type and context."""
        server, connector, log_file = mock_system_with_error_logging
        
        error_logger = get_adapter_logger("error_test_system")
        
        # Test invalid action execution to trigger real error
        invalid_action = AdaptationAction(
            action_id="error_test_invalid",
            action_type="NONEXISTENT_ACTION",
            target_system="error_test_system",
            parameters={"invalid": True},
            priority=1,
            created_at=datetime.now(timezone.utc)
        )
        
        try:
            result = await connector.execute_action(invalid_action)
            
            # Log the error result
            if result.status == ExecutionStatus.FAILED:
                error_logger.error(
                    "Action execution failed",
                    extra={
                        "system_id": "error_test_system",
                        "event_type": "action_execution_error",
                        "action_id": invalid_action.action_id,
                        "action_type": invalid_action.action_type,
                        "error_type": "ActionExecutionError",
                        "error_message": result.error_message or "Action execution failed",
                        "error_context": {
                            "action_parameters": invalid_action.parameters,
                            "target_system": invalid_action.target_system,
                            "execution_time_ms": result.execution_time_ms,
                            "result_data": result.result_data
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
        except Exception as e:
            # Log any unexpected exceptions
            error_logger.error(
                "Unexpected error during action execution",
                extra={
                    "system_id": "error_test_system",
                    "event_type": "unexpected_error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "error_context": {
                        "action_id": invalid_action.action_id,
                        "action_type": invalid_action.action_type
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                exc_info=e
            )
        
        await asyncio.sleep(0.1)
        
        # Verify error context logs
        log_records = log_file_reader(log_file, "json")
        context_error_logs = [
            record for record in log_records 
            if record.get('level') == 'ERROR' and 
            record.get('extra', {}).get('event_type') in ['action_execution_error', 'unexpected_error']
        ]
        
        assert len(context_error_logs) >= 1, f"Expected at least 1 context error log, got {len(context_error_logs)}"
        
        for error_log in context_error_logs:
            extra = error_log.get('extra', {})
            
            # Verify error type is specific and meaningful
            assert 'error_type' in extra, "Error log missing error_type"
            error_type = extra['error_type']
            assert error_type != "Exception", f"Error type should be specific, not generic: {error_type}"
            
            # Verify error message is descriptive
            assert 'error_message' in extra, "Error log missing error_message"
            error_message = extra['error_message']
            assert len(error_message) > 0, "Error message should not be empty"
            
            # Verify error context provides useful debugging information
            assert 'error_context' in extra, "Error log missing error_context"
            error_context = extra['error_context']
            assert isinstance(error_context, dict), "Error context should be a dictionary"
            assert len(error_context) > 0, "Error context should not be empty"
            
            # Context should include relevant operational details
            if extra.get('event_type') == 'action_execution_error':
                assert 'action_parameters' in error_context, "Action error context missing action_parameters"
                assert 'target_system' in error_context, "Action error context missing target_system"


@pytest.mark.integration
class TestMetricsExportVerification:
    """Test metrics export verification (Requirements 7.4)."""
    
    @pytest.fixture
    async def mock_system_with_metrics_export(self):
        """Set up mock system with metrics export logging configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = f.name
        
        config = LoggingConfiguration(
            level="INFO",
            format="json",
            output="file",
            file_path=log_file
        )
        configure_logging(config)
        
        # Create mock system
        test_config = {
            "server": {"host": "localhost", "port": 5013, "max_connections": 10},
            "baseline_metrics": {
                "cpu_usage": 40.0, "memory_usage": 1536.0, "response_time": 120.0,
                "throughput": 45.0, "error_rate": 1.0, "active_connections": 8, "capacity": 4
            },
            "simulation": {"noise_factor": 0.05, "update_interval": 0.1},
            "capacity": {"min_capacity": 1, "max_capacity": 10, "scale_up_increment": 1, "scale_down_increment": 1}
        }
        
        state_manager = StateManager(initial_config=test_config)
        server = MockSystemServer("localhost", 5013, state_manager, max_connections=10)
        await server.start()
        
        connector_config = {
            "system_name": "metrics_export_test_system",
            "connection": {"host": "localhost", "port": 5013},
            "implementation": {"timeout": 5.0, "max_retries": 2, "retry_base_delay": 0.1}
        }
        connector = MockSystemConnector(connector_config)
        await connector.connect()
        
        try:
            yield server, connector, log_file
        finally:
            await connector.disconnect()
            await server.stop()
            reset_logging()
            try:
                Path(log_file).unlink()
            except FileNotFoundError:
                pass
    
    @pytest.mark.asyncio
    async def test_performance_metrics_exported_in_structured_format(self, mock_system_with_metrics_export, log_file_reader):
        """Test that performance metrics are exported in structured format."""
        server, connector, log_file = mock_system_with_metrics_export
        
        metrics_logger = get_infrastructure_logger("metrics_export")
        
        # Simulate performance metrics collection and export
        for i in range(3):
            start_time = time.time()
            
            # Collect metrics (simulating performance measurement)
            metrics = await connector.collect_metrics()
            
            end_time = time.time()
            collection_duration = (end_time - start_time) * 1000  # Convert to ms
            
            # Create structured performance metrics
            performance_metrics = {
                "system_id": "metrics_export_test_system",
                "collection_id": f"perf_collection_{i+1}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "collection_performance": {
                    "duration_ms": round(collection_duration, 2),
                    "metrics_count": len(metrics),
                    "success": True
                },
                "system_metrics": {
                    "cpu_usage": {
                        "value": metrics["cpu_usage"].value,
                        "unit": "percent",
                        "timestamp": metrics["cpu_usage"].timestamp.isoformat()
                    },
                    "memory_usage": {
                        "value": metrics["memory_usage"].value,
                        "unit": "MB",
                        "timestamp": metrics["memory_usage"].timestamp.isoformat()
                    },
                    "response_time": {
                        "value": metrics["response_time"].value,
                        "unit": "ms",
                        "timestamp": metrics["response_time"].timestamp.isoformat()
                    },
                    "throughput": {
                        "value": metrics["throughput"].value,
                        "unit": "req/s",
                        "timestamp": metrics["throughput"].timestamp.isoformat()
                    }
                },
                "metadata": {
                    "collection_method": "tcp_connector",
                    "framework_version": "2.0.0",
                    "export_format": "structured_json"
                }
            }
            
            # Export metrics in structured format
            metrics_logger.info(
                "Performance metrics exported",
                extra={
                    "event_type": "metrics_export",
                    "export_format": "structured_json",
                    "metrics_data": performance_metrics
                }
            )
            
            await asyncio.sleep(0.1)
        
        await asyncio.sleep(0.2)
        
        # Verify structured metrics export
        log_records = log_file_reader(log_file, "json")
        export_logs = [
            record for record in log_records 
            if record.get('extra', {}).get('event_type') == 'metrics_export'
        ]
        
        assert len(export_logs) >= 3, f"Expected at least 3 metrics export logs, got {len(export_logs)}"
        
        for i, export_log in enumerate(export_logs):
            extra = export_log.get('extra', {})
            
            # Verify export format is specified
            assert 'export_format' in extra, f"Export log {i+1} missing export_format"
            assert extra['export_format'] == 'structured_json', f"Export log {i+1} should use structured_json format"
            
            # Verify metrics data is present and structured
            assert 'metrics_data' in extra, f"Export log {i+1} missing metrics_data"
            metrics_data = extra['metrics_data']
            
            # Verify required top-level fields
            required_fields = ['system_id', 'collection_id', 'timestamp', 'collection_performance', 'system_metrics', 'metadata']
            for field in required_fields:
                assert field in metrics_data, f"Export log {i+1} metrics_data missing {field}"
            
            # Verify collection performance structure
            perf = metrics_data['collection_performance']
            assert 'duration_ms' in perf, f"Export log {i+1} missing collection duration"
            assert 'metrics_count' in perf, f"Export log {i+1} missing metrics count"
            assert 'success' in perf, f"Export log {i+1} missing success indicator"
            assert isinstance(perf['duration_ms'], (int, float)), f"Export log {i+1} duration should be numeric"
            assert perf['metrics_count'] > 0, f"Export log {i+1} should have positive metrics count"
            
            # Verify system metrics structure
            sys_metrics = metrics_data['system_metrics']
            expected_metrics = ['cpu_usage', 'memory_usage', 'response_time', 'throughput']
            for metric_name in expected_metrics:
                assert metric_name in sys_metrics, f"Export log {i+1} missing {metric_name}"
                metric = sys_metrics[metric_name]
                assert 'value' in metric, f"Export log {i+1} {metric_name} missing value"
                assert 'unit' in metric, f"Export log {i+1} {metric_name} missing unit"
                assert 'timestamp' in metric, f"Export log {i+1} {metric_name} missing timestamp"
                assert isinstance(metric['value'], (int, float)), f"Export log {i+1} {metric_name} value should be numeric"
            
            # Verify metadata structure
            metadata = metrics_data['metadata']
            assert 'collection_method' in metadata, f"Export log {i+1} missing collection_method"
            assert 'framework_version' in metadata, f"Export log {i+1} missing framework_version"
            assert 'export_format' in metadata, f"Export log {i+1} missing export_format in metadata"
    
    @pytest.mark.asyncio
    async def test_exported_metrics_include_all_required_fields(self, mock_system_with_metrics_export, log_file_reader):
        """Test that exported metrics include all required fields."""
        server, connector, log_file = mock_system_with_metrics_export
        
        metrics_logger = get_infrastructure_logger("metrics_export")
        
        # Create comprehensive metrics export with all required fields
        metrics = await connector.collect_metrics()
        
        comprehensive_export = {
            "export_id": "comprehensive_export_001",
            "system_identification": {
                "system_id": "metrics_export_test_system",
                "system_type": "mock_external_system",
                "connector_type": "mock_system_connector"
            },
            "temporal_information": {
                "collection_timestamp": datetime.now(timezone.utc).isoformat(),
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "timezone": "UTC"
            },
            "performance_metrics": {
                "collection_latency_ms": 45.2,
                "export_latency_ms": 12.8,
                "total_processing_time_ms": 58.0,
                "metrics_processed": len(metrics),
                "success_rate": 100.0
            },
            "system_health_metrics": {
                metric_name: {
                    "current_value": metric.value,
                    "unit_of_measurement": getattr(metric, 'unit', 'unknown'),
                    "measurement_timestamp": metric.timestamp.isoformat(),
                    "data_quality": "high",
                    "collection_method": "direct_query"
                }
                for metric_name, metric in metrics.items()
            },
            "operational_context": {
                "collection_interval_seconds": 5,
                "retention_policy": "30_days",
                "aggregation_level": "raw",
                "data_source": "live_system"
            },
            "quality_indicators": {
                "data_completeness": 100.0,
                "data_freshness_seconds": 1.0,
                "collection_reliability": "high",
                "validation_status": "passed"
            }
        }
        
        # Export comprehensive metrics
        metrics_logger.info(
            "Comprehensive metrics export",
            extra={
                "event_type": "comprehensive_metrics_export",
                "export_format": "full_structured_json",
                "comprehensive_data": comprehensive_export
            }
        )
        
        await asyncio.sleep(0.1)
        
        # Verify comprehensive export
        log_records = log_file_reader(log_file, "json")
        comprehensive_logs = [
            record for record in log_records 
            if record.get('extra', {}).get('event_type') == 'comprehensive_metrics_export'
        ]
        
        assert len(comprehensive_logs) >= 1, "Should have comprehensive metrics export log"
        
        comp_log = comprehensive_logs[0]
        extra = comp_log.get('extra', {})
        comp_data = extra['comprehensive_data']
        
        # Verify all required field categories are present
        required_categories = [
            'export_id', 'system_identification', 'temporal_information',
            'performance_metrics', 'system_health_metrics', 'operational_context', 'quality_indicators'
        ]
        
        for category in required_categories:
            assert category in comp_data, f"Comprehensive export missing {category}"
        
        # Verify system identification fields
        sys_id = comp_data['system_identification']
        assert 'system_id' in sys_id, "Missing system_id in identification"
        assert 'system_type' in sys_id, "Missing system_type in identification"
        assert 'connector_type' in sys_id, "Missing connector_type in identification"
        
        # Verify temporal information fields
        temporal = comp_data['temporal_information']
        assert 'collection_timestamp' in temporal, "Missing collection_timestamp"
        assert 'export_timestamp' in temporal, "Missing export_timestamp"
        assert 'timezone' in temporal, "Missing timezone"
        
        # Verify performance metrics fields
        perf_metrics = comp_data['performance_metrics']
        perf_fields = ['collection_latency_ms', 'export_latency_ms', 'total_processing_time_ms', 'metrics_processed', 'success_rate']
        for field in perf_fields:
            assert field in perf_metrics, f"Missing performance field: {field}"
            assert isinstance(perf_metrics[field], (int, float)), f"Performance field {field} should be numeric"
        
        # Verify system health metrics structure
        health_metrics = comp_data['system_health_metrics']
        assert len(health_metrics) > 0, "Should have system health metrics"
        
        for metric_name, metric_data in health_metrics.items():
            metric_fields = ['current_value', 'unit_of_measurement', 'measurement_timestamp', 'data_quality', 'collection_method']
            for field in metric_fields:
                assert field in metric_data, f"Metric {metric_name} missing {field}"
        
        # Verify quality indicators
        quality = comp_data['quality_indicators']
        quality_fields = ['data_completeness', 'data_freshness_seconds', 'collection_reliability', 'validation_status']
        for field in quality_fields:
            assert field in quality, f"Missing quality indicator: {field}"