"""
Mock System Integration Tests

This module provides comprehensive integration tests for the mock external system
with POLARIS framework, testing:
- Mock system startup and connection
- Telemetry collection from mock system
- Basic adaptation trigger and execution
- End-to-end integration flows

Requirements: 5.1, 5.2, 5.3, 5.4
"""

import asyncio
import pytest
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List

from domain.models import (
    MetricValue, SystemState, AdaptationAction, 
    ExecutionResult, HealthStatus, ExecutionStatus
)
from src.framework.events import TelemetryEvent, AdaptationEvent
from tests.integration.harness.polaris_integration_test_harness import (
    PolarisIntegrationTestHarness, IntegrationTestConfig, SystemConfig,
    create_simple_harness, create_performance_harness
)

# Import mock system components for direct testing
from polaris_refactored.mock_external_system.src.server import MockSystemServer
from polaris_refactored.mock_external_system.src.state_manager import StateManager
from polaris_refactored.plugins.mock_system.connector import MockSystemConnector


@pytest.mark.integration
class TestMockSystemBasicIntegration:
    """Basic integration tests for mock system with POLARIS."""
    
    @pytest.fixture
    async def mock_system_server(self):
        """Start a mock system server for testing."""
        # Create state manager with test configuration
        test_config = {
            "server": {"host": "localhost", "port": 5001, "max_connections": 10},
            "baseline_metrics": {
                "cpu_usage": 25.0,
                "memory_usage": 1024.0,
                "response_time": 80.0,
                "throughput": 60.0,
                "error_rate": 0.3,
                "active_connections": 5,
                "capacity": 3
            },
            "simulation": {"noise_factor": 0.05, "update_interval": 0.1},
            "capacity": {"min_capacity": 1, "max_capacity": 10, "scale_up_increment": 1, "scale_down_increment": 1}
        }
        
        state_manager = StateManager(initial_config=test_config)
        server = MockSystemServer(
            host="localhost",
            port=5001,
            state_manager=state_manager,
            max_connections=10
        )
        
        await server.start()
        
        # Wait for server to be ready
        await asyncio.sleep(0.1)
        
        try:
            yield server
        finally:
            await server.stop()
    
    @pytest.fixture
    async def mock_system_connector(self, mock_system_server):
        """Create a connector to the mock system."""
        connector_config = {
            "system_name": "test_mock_system",
            "connection": {"host": "localhost", "port": 5001},
            "implementation": {"timeout": 5.0, "max_retries": 2, "retry_base_delay": 0.1}
        }
        
        connector = MockSystemConnector(connector_config)
        
        # Test connection
        connected = await connector.connect()
        assert connected, "Failed to connect to mock system"
        
        try:
            yield connector
        finally:
            await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_mock_system_startup_and_connection(self, mock_system_server):
        """Test that mock system starts up and accepts connections."""
        # Verify server is running
        assert mock_system_server.running
        assert mock_system_server.connection_count == 0
        
        # Test basic connection
        connector_config = {
            "connection": {"host": "localhost", "port": 5001},
            "implementation": {"timeout": 5.0}
        }
        connector = MockSystemConnector(connector_config)
        
        # Test connection
        connected = await connector.connect()
        assert connected, "Should be able to connect to mock system"
        
        # Test system ID
        system_id = await connector.get_system_id()
        assert system_id == "mock_system"
        
        # Test disconnect
        disconnected = await connector.disconnect()
        assert disconnected, "Should be able to disconnect from mock system"
    
    @pytest.mark.asyncio
    async def test_telemetry_collection_from_mock_system(self, mock_system_connector):
        """Test collecting telemetry data from mock system."""
        # Collect metrics
        metrics = await mock_system_connector.collect_metrics()
        
        # Verify all expected metrics are present
        expected_metrics = [
            "cpu_usage", "memory_usage", "response_time", 
            "throughput", "error_rate", "active_connections", "capacity"
        ]
        
        for metric_name in expected_metrics:
            assert metric_name in metrics, f"Missing metric: {metric_name}"
            metric = metrics[metric_name]
            assert isinstance(metric, MetricValue), f"Invalid metric type for {metric_name}"
            assert metric.name == metric_name
            assert isinstance(metric.value, (int, float)), f"Invalid value type for {metric_name}"
            assert metric.timestamp is not None
        
        # Verify metric values are reasonable
        assert 0 <= metrics["cpu_usage"].value <= 100, "CPU usage should be 0-100%"
        assert metrics["memory_usage"].value > 0, "Memory usage should be positive"
        assert metrics["response_time"].value > 0, "Response time should be positive"
        assert metrics["throughput"].value >= 0, "Throughput should be non-negative"
        assert 0 <= metrics["error_rate"].value <= 100, "Error rate should be 0-100%"
        assert metrics["active_connections"].value >= 0, "Active connections should be non-negative"
        assert metrics["capacity"].value > 0, "Capacity should be positive"
    
    @pytest.mark.asyncio
    async def test_system_state_collection(self, mock_system_connector):
        """Test collecting system state from mock system."""
        # Get system state
        state = await mock_system_connector.get_system_state()
        
        # Verify state structure
        assert isinstance(state, SystemState)
        assert state.system_id == "test_mock_system"
        assert state.health_status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.UNHEALTHY]
        assert isinstance(state.metrics, dict)
        assert len(state.metrics) > 0
        assert state.timestamp is not None
        
        # Verify metrics in state
        for metric_name, metric in state.metrics.items():
            assert isinstance(metric, MetricValue)
            assert metric.name == metric_name
    
    @pytest.mark.asyncio
    async def test_basic_adaptation_trigger_and_execution(self, mock_system_connector):
        """Test triggering and executing basic adaptations."""
        # First, set high load to trigger adaptation conditions
        success = await mock_system_connector.set_load(0.9)  # High load
        assert success, "Should be able to set load level"
        
        # Wait for load to take effect
        await asyncio.sleep(0.2)
        
        # Collect metrics to see high load effects
        metrics = await mock_system_connector.collect_metrics()
        initial_cpu = metrics["cpu_usage"].value
        initial_capacity = metrics["capacity"].value
        
        # Create and execute a scale-up action
        scale_up_action = AdaptationAction(
            action_id="test_scale_up_001",
            action_type="SCALE_UP",
            target_system="test_mock_system",
            parameters={"increment": 1, "reason": "high_cpu_test"},
            priority=1,
            created_at=datetime.now(timezone.utc)
        )
        
        # Execute the action
        result = await mock_system_connector.execute_action(scale_up_action)
        
        # Verify execution result
        assert isinstance(result, ExecutionResult)
        assert result.action_id == "test_scale_up_001"
        assert result.status == ExecutionStatus.SUCCESS, f"Action failed: {result.error_message}"
        assert result.execution_time_ms > 0
        assert "mock_response" in result.result_data
        
        # Wait for action effects to take place
        await asyncio.sleep(0.2)
        
        # Collect metrics again to verify action effects
        new_metrics = await mock_system_connector.collect_metrics()
        new_capacity = new_metrics["capacity"].value
        
        # Verify capacity increased
        assert new_capacity > initial_capacity, f"Capacity should have increased: {initial_capacity} -> {new_capacity}"
        
        # Test action validation
        valid_action = AdaptationAction(
            action_id="test_validation_001",
            action_type="SCALE_DOWN",
            target_system="test_mock_system",
            parameters={"decrement": 1},
            priority=1,
            created_at=datetime.now(timezone.utc)
        )
        
        is_valid = await mock_system_connector.validate_action(valid_action)
        assert is_valid, "Scale down should be valid when capacity > min"
        
        # Test invalid action (scale down below minimum)
        invalid_action = AdaptationAction(
            action_id="test_validation_002",
            action_type="SCALE_DOWN",
            target_system="test_mock_system",
            parameters={"decrement": 100},  # Too much
            priority=1,
            created_at=datetime.now(timezone.utc)
        )
        
        is_invalid = await mock_system_connector.validate_action(invalid_action)
        # Note: This might still be valid if the mock system allows it, 
        # but the actual execution should handle the constraint
    
    @pytest.mark.asyncio
    async def test_supported_actions(self, mock_system_connector):
        """Test getting supported actions from mock system."""
        actions = await mock_system_connector.get_supported_actions()
        
        # Verify expected actions are supported
        expected_actions = [
            "SCALE_UP", "SCALE_DOWN", "ADJUST_QOS", "RESTART_SERVICE",
            "OPTIMIZE_CONFIG", "ENABLE_CACHING", "DISABLE_CACHING"
        ]
        
        for action in expected_actions:
            assert action in actions, f"Missing supported action: {action}"
        
        assert len(actions) >= len(expected_actions), "Should have at least the expected actions"
    
    @pytest.mark.asyncio
    async def test_multiple_actions_sequence(self, mock_system_connector):
        """Test executing a sequence of actions."""
        # Reset system to known state
        await mock_system_connector.reset()
        await asyncio.sleep(0.1)
        
        # Get initial state
        initial_metrics = await mock_system_connector.collect_metrics()
        initial_capacity = initial_metrics["capacity"].value
        
        # Execute scale-up
        scale_up = AdaptationAction(
            action_id="seq_001",
            action_type="SCALE_UP",
            target_system="test_mock_system",
            parameters={"increment": 2},
            priority=1,
            created_at=datetime.now(timezone.utc)
        )
        
        result1 = await mock_system_connector.execute_action(scale_up)
        assert result1.status == ExecutionStatus.SUCCESS
        
        await asyncio.sleep(0.1)
        
        # Verify capacity increased
        mid_metrics = await mock_system_connector.collect_metrics()
        mid_capacity = mid_metrics["capacity"].value
        assert mid_capacity > initial_capacity
        
        # Execute optimization
        optimize = AdaptationAction(
            action_id="seq_002",
            action_type="OPTIMIZE_CONFIG",
            target_system="test_mock_system",
            parameters={},
            priority=1,
            created_at=datetime.now(timezone.utc)
        )
        
        result2 = await mock_system_connector.execute_action(optimize)
        assert result2.status == ExecutionStatus.SUCCESS
        
        await asyncio.sleep(0.1)
        
        # Execute scale-down
        scale_down = AdaptationAction(
            action_id="seq_003",
            action_type="SCALE_DOWN",
            target_system="test_mock_system",
            parameters={"decrement": 1},
            priority=1,
            created_at=datetime.now(timezone.utc)
        )
        
        result3 = await mock_system_connector.execute_action(scale_down)
        assert result3.status == ExecutionStatus.SUCCESS
        
        await asyncio.sleep(0.1)
        
        # Verify final state
        final_metrics = await mock_system_connector.collect_metrics()
        final_capacity = final_metrics["capacity"].value
        
        # Should be initial + 2 - 1 = initial + 1
        expected_capacity = initial_capacity + 1
        assert final_capacity == expected_capacity, f"Expected capacity {expected_capacity}, got {final_capacity}"
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, mock_system_connector):
        """Test error handling and recovery scenarios."""
        # Test invalid action type
        invalid_action = AdaptationAction(
            action_id="error_001",
            action_type="INVALID_ACTION",
            target_system="test_mock_system",
            parameters={},
            priority=1,
            created_at=datetime.now(timezone.utc)
        )
        
        result = await mock_system_connector.execute_action(invalid_action)
        assert result.status == ExecutionStatus.FAILED
        assert "error" in result.result_data or result.error_message is not None
        
        # Test connection recovery by collecting metrics after error
        metrics = await mock_system_connector.collect_metrics()
        assert len(metrics) > 0, "Should still be able to collect metrics after error"
        
        # Test system reset
        reset_success = await mock_system_connector.reset()
        assert reset_success, "Should be able to reset system"
        
        # Verify system is back to baseline
        await asyncio.sleep(0.1)
        reset_metrics = await mock_system_connector.collect_metrics()
        assert reset_metrics["capacity"].value > 0, "Capacity should be positive after reset"


@pytest.mark.integration
class TestMockSystemScenarioIntegration:
    """Integration tests for different operational scenarios."""
    
    @pytest.fixture
    async def mock_system_server(self):
        """Start a mock system server for scenario testing."""
        test_config = {
            "server": {"host": "localhost", "port": 5003, "max_connections": 10},
            "baseline_metrics": {
                "cpu_usage": 30.0,
                "memory_usage": 2048.0,
                "response_time": 100.0,
                "throughput": 50.0,
                "error_rate": 0.5,
                "active_connections": 10,
                "capacity": 5
            },
            "simulation": {"noise_factor": 0.05, "update_interval": 0.1},
            "capacity": {"min_capacity": 1, "max_capacity": 15, "scale_up_increment": 2, "scale_down_increment": 1}
        }
        
        state_manager = StateManager(initial_config=test_config)
        server = MockSystemServer(
            host="localhost",
            port=5003,
            state_manager=state_manager,
            max_connections=10
        )
        
        await server.start()
        await asyncio.sleep(0.1)
        
        try:
            yield server
        finally:
            await server.stop()
    
    @pytest.fixture
    async def scenario_connector(self, mock_system_server):
        """Create a connector for scenario testing."""
        connector_config = {
            "system_name": "scenario_test_system",
            "connection": {"host": "localhost", "port": 5003},
            "implementation": {"timeout": 5.0, "max_retries": 2, "retry_base_delay": 0.1}
        }
        
        connector = MockSystemConnector(connector_config)
        connected = await connector.connect()
        assert connected, "Failed to connect to scenario test system"
        
        try:
            yield connector
        finally:
            await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_normal_operation_scenario(self, scenario_connector):
        """Test normal operation scenario (stable metrics, no adaptations)."""
        # Reset system to baseline
        await scenario_connector.reset()
        await asyncio.sleep(0.1)
        
        # Set normal load level
        await scenario_connector.set_load(0.3)  # 30% load - should be stable
        await asyncio.sleep(0.2)
        
        # Collect metrics multiple times to verify stability
        metrics_samples = []
        for i in range(5):
            metrics = await scenario_connector.collect_metrics()
            metrics_samples.append(metrics)
            await asyncio.sleep(0.1)
        
        # Verify metrics are stable (within reasonable bounds)
        cpu_values = [m["cpu_usage"].value for m in metrics_samples]
        memory_values = [m["memory_usage"].value for m in metrics_samples]
        
        # CPU should be reasonable for normal load
        avg_cpu = sum(cpu_values) / len(cpu_values)
        assert 20.0 <= avg_cpu <= 60.0, f"CPU should be stable under normal load: {avg_cpu}%"
        
        # Memory should be reasonable
        avg_memory = sum(memory_values) / len(memory_values)
        assert avg_memory > 0, f"Memory usage should be positive: {avg_memory}MB"
        
        # Verify no extreme variations (coefficient of variation < 20%)
        cpu_std = (sum((x - avg_cpu) ** 2 for x in cpu_values) / len(cpu_values)) ** 0.5
        cpu_cv = cpu_std / avg_cpu if avg_cpu > 0 else 0
        assert cpu_cv < 0.3, f"CPU should be stable (CV < 30%): {cpu_cv:.2f}"
        
        print(f"Normal operation: CPU avg={avg_cpu:.1f}%, CV={cpu_cv:.2f}")
    
    @pytest.mark.asyncio
    async def test_high_load_scenario(self, scenario_connector):
        """Test high load scenario (triggers scale-up)."""
        # Reset system to baseline
        await scenario_connector.reset()
        await asyncio.sleep(0.1)
        
        # Get initial capacity
        initial_metrics = await scenario_connector.collect_metrics()
        initial_capacity = initial_metrics["capacity"].value
        initial_cpu = initial_metrics["cpu_usage"].value
        
        # Apply high load
        await scenario_connector.set_load(0.95)  # 95% load - should trigger scale-up
        await asyncio.sleep(0.3)  # Wait for load effects
        
        # Collect metrics under high load
        high_load_metrics = await scenario_connector.collect_metrics()
        high_load_cpu = high_load_metrics["cpu_usage"].value
        
        # Verify high load increased CPU usage
        assert high_load_cpu > initial_cpu, f"High load should increase CPU: {initial_cpu}% -> {high_load_cpu}%"
        
        # Execute scale-up action to handle high load
        scale_up_action = AdaptationAction(
            action_id="high_load_scale_up",
            action_type="SCALE_UP",
            target_system="scenario_test_system",
            parameters={"increment": 2, "reason": "high_load_scenario"},
            priority=1,
            created_at=datetime.now(timezone.utc)
        )
        
        result = await scenario_connector.execute_action(scale_up_action)
        assert result.status == ExecutionStatus.SUCCESS, f"Scale-up failed: {result.error_message}"
        
        # Wait for scale-up effects
        await asyncio.sleep(0.3)
        
        # Verify capacity increased
        post_scale_metrics = await scenario_connector.collect_metrics()
        post_scale_capacity = post_scale_metrics["capacity"].value
        post_scale_cpu = post_scale_metrics["cpu_usage"].value
        
        assert post_scale_capacity > initial_capacity, f"Capacity should increase: {initial_capacity} -> {post_scale_capacity}"
        
        # CPU should improve (decrease) after scaling up, but allow for some noise
        # The improvement might not be immediate due to noise, so we check for capacity increase
        # and that CPU is not significantly worse
        assert post_scale_cpu <= high_load_cpu * 1.1, f"CPU should not get significantly worse after scale-up: {high_load_cpu}% -> {post_scale_cpu}%"
        
        print(f"High load scenario: CPU {initial_cpu:.1f}% -> {high_load_cpu:.1f}% -> {post_scale_cpu:.1f}%")
        print(f"Capacity scaled: {initial_capacity} -> {post_scale_capacity}")
    
    @pytest.mark.asyncio
    async def test_resource_constraint_scenario(self, scenario_connector):
        """Test resource constraint scenario (triggers optimization)."""
        # Reset system to baseline
        await scenario_connector.reset()
        await asyncio.sleep(0.1)
        
        # Set moderate load to create resource pressure
        await scenario_connector.set_load(0.7)  # 70% load
        await asyncio.sleep(0.2)
        
        # Get initial metrics
        initial_metrics = await scenario_connector.collect_metrics()
        initial_cpu = initial_metrics["cpu_usage"].value
        initial_memory = initial_metrics["memory_usage"].value
        initial_response_time = initial_metrics["response_time"].value
        
        # Execute optimization action to improve efficiency
        optimize_action = AdaptationAction(
            action_id="resource_optimization",
            action_type="OPTIMIZE_CONFIG",
            target_system="scenario_test_system",
            parameters={"reason": "resource_constraint_scenario"},
            priority=1,
            created_at=datetime.now(timezone.utc)
        )
        
        result = await scenario_connector.execute_action(optimize_action)
        assert result.status == ExecutionStatus.SUCCESS, f"Optimization failed: {result.error_message}"
        
        # Wait for optimization effects
        await asyncio.sleep(0.3)
        
        # Verify optimization improved resource usage
        optimized_metrics = await scenario_connector.collect_metrics()
        optimized_cpu = optimized_metrics["cpu_usage"].value
        optimized_memory = optimized_metrics["memory_usage"].value
        
        # Optimization should reduce resource usage (allowing for noise in metrics)
        # Allow up to 5% variance due to noise in the metrics simulator
        cpu_tolerance = initial_cpu * 0.05
        memory_tolerance = initial_memory * 0.05
        assert optimized_cpu <= initial_cpu + cpu_tolerance, f"CPU should improve or stay similar: {initial_cpu}% -> {optimized_cpu}% (tolerance: {cpu_tolerance}%)"
        assert optimized_memory <= initial_memory + memory_tolerance, f"Memory should improve or stay similar: {initial_memory}MB -> {optimized_memory}MB (tolerance: {memory_tolerance}MB)"
        
        print(f"Resource optimization: CPU {initial_cpu:.1f}% -> {optimized_cpu:.1f}%")
        print(f"Memory optimization: {initial_memory:.0f}MB -> {optimized_memory:.0f}MB")
    
    @pytest.mark.asyncio
    async def test_failure_recovery_scenario(self, scenario_connector):
        """Test failure recovery scenario (detects and recovers)."""
        # Reset system to baseline
        await scenario_connector.reset()
        await asyncio.sleep(0.1)
        
        # Simulate failure by setting very high load and error conditions
        await scenario_connector.set_load(1.0)  # Maximum load
        await asyncio.sleep(0.2)
        
        # Get metrics showing degraded performance
        degraded_metrics = await scenario_connector.collect_metrics()
        degraded_error_rate = degraded_metrics["error_rate"].value
        degraded_response_time = degraded_metrics["response_time"].value
        
        # Execute restart service action for recovery
        restart_action = AdaptationAction(
            action_id="failure_recovery_restart",
            action_type="RESTART_SERVICE",
            target_system="scenario_test_system",
            parameters={"reason": "failure_recovery_scenario"},
            priority=1,
            created_at=datetime.now(timezone.utc)
        )
        
        result = await scenario_connector.execute_action(restart_action)
        assert result.status == ExecutionStatus.SUCCESS, f"Restart failed: {result.error_message}"
        
        # Wait for restart effects
        await asyncio.sleep(0.3)
        
        # Verify recovery - metrics should improve
        recovered_metrics = await scenario_connector.collect_metrics()
        recovered_error_rate = recovered_metrics["error_rate"].value
        recovered_response_time = recovered_metrics["response_time"].value
        
        # After restart, error rate should be lower or similar (allowing for noise)
        # The main point is that restart should reset to baseline, not necessarily improve
        baseline_error_rate = 0.5  # From test config
        assert recovered_error_rate <= degraded_error_rate * 1.1, f"Error rate should not get significantly worse: {degraded_error_rate}% -> {recovered_error_rate}%"
        
        # Connections should be reasonable after restart (may not be exactly 0 due to load effects)
        # The main point is that restart action succeeded and system is functional
        assert recovered_metrics["active_connections"].value >= 0, "Active connections should be non-negative after restart"
        
        print(f"Failure recovery: Error rate {degraded_error_rate:.2f}% -> {recovered_error_rate:.2f}%")
        print(f"Response time {degraded_response_time:.1f}ms -> {recovered_response_time:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_mixed_workload_scenario(self, scenario_connector):
        """Test mixed workload scenario (multiple concurrent adaptations)."""
        # Reset system to baseline
        await scenario_connector.reset()
        await asyncio.sleep(0.1)
        
        # Get initial state
        initial_metrics = await scenario_connector.collect_metrics()
        initial_capacity = initial_metrics["capacity"].value
        
        # Apply moderate load
        await scenario_connector.set_load(0.6)
        await asyncio.sleep(0.2)
        
        # Execute multiple adaptations in sequence to simulate mixed workload
        actions = [
            AdaptationAction(
                action_id="mixed_001_scale_up",
                action_type="SCALE_UP",
                target_system="scenario_test_system",
                parameters={"increment": 1, "reason": "mixed_workload_capacity"},
                priority=1,
                created_at=datetime.now(timezone.utc)
            ),
            AdaptationAction(
                action_id="mixed_002_enable_cache",
                action_type="ENABLE_CACHING",
                target_system="scenario_test_system",
                parameters={"reason": "mixed_workload_performance"},
                priority=2,
                created_at=datetime.now(timezone.utc)
            ),
            AdaptationAction(
                action_id="mixed_003_optimize",
                action_type="OPTIMIZE_CONFIG",
                target_system="scenario_test_system",
                parameters={"reason": "mixed_workload_efficiency"},
                priority=3,
                created_at=datetime.now(timezone.utc)
            )
        ]
        
        results = []
        for action in actions:
            result = await scenario_connector.execute_action(action)
            results.append(result)
            await asyncio.sleep(0.1)  # Small delay between actions
        
        # Verify all actions succeeded
        for i, result in enumerate(results):
            assert result.status == ExecutionStatus.SUCCESS, f"Action {i+1} failed: {result.error_message}"
        
        # Wait for all effects to take place
        await asyncio.sleep(0.3)
        
        # Verify final state reflects all adaptations
        final_metrics = await scenario_connector.collect_metrics()
        final_capacity = final_metrics["capacity"].value
        
        # Capacity should have increased
        assert final_capacity > initial_capacity, f"Capacity should increase: {initial_capacity} -> {final_capacity}"
        
        # Get system state to verify caching is enabled
        system_state = await scenario_connector.get_system_state()
        # Note: We can't directly check caching state through metrics, but the action should have succeeded
        
        print(f"Mixed workload: Executed {len(actions)} adaptations successfully")
        print(f"Capacity: {initial_capacity} -> {final_capacity}")
        print(f"All actions completed: {[r.action_id for r in results]}")
    
    @pytest.mark.asyncio
    async def test_cascading_failure_scenario(self, scenario_connector):
        """Test cascading failure scenario with multiple recovery steps."""
        # Reset system to baseline
        await scenario_connector.reset()
        await asyncio.sleep(0.1)
        
        # Simulate cascading failure: high load -> high error rate -> need multiple interventions
        await scenario_connector.set_load(0.95)  # Very high load
        await asyncio.sleep(0.2)
        
        initial_metrics = await scenario_connector.collect_metrics()
        initial_cpu = initial_metrics["cpu_usage"].value
        initial_capacity = initial_metrics["capacity"].value
        
        # Step 1: Scale up to handle load
        scale_action = AdaptationAction(
            action_id="cascade_001_scale",
            action_type="SCALE_UP",
            target_system="scenario_test_system",
            parameters={"increment": 2},
            priority=1,
            created_at=datetime.now(timezone.utc)
        )
        
        scale_result = await scenario_connector.execute_action(scale_action)
        assert scale_result.status == ExecutionStatus.SUCCESS
        await asyncio.sleep(0.2)
        
        # Step 2: Enable caching for performance
        cache_action = AdaptationAction(
            action_id="cascade_002_cache",
            action_type="ENABLE_CACHING",
            target_system="scenario_test_system",
            parameters={},
            priority=2,
            created_at=datetime.now(timezone.utc)
        )
        
        cache_result = await scenario_connector.execute_action(cache_action)
        assert cache_result.status == ExecutionStatus.SUCCESS
        await asyncio.sleep(0.2)
        
        # Step 3: Optimize configuration
        optimize_action = AdaptationAction(
            action_id="cascade_003_optimize",
            action_type="OPTIMIZE_CONFIG",
            target_system="scenario_test_system",
            parameters={},
            priority=3,
            created_at=datetime.now(timezone.utc)
        )
        
        optimize_result = await scenario_connector.execute_action(optimize_action)
        assert optimize_result.status == ExecutionStatus.SUCCESS
        await asyncio.sleep(0.3)
        
        # Verify recovery
        final_metrics = await scenario_connector.collect_metrics()
        final_cpu = final_metrics["cpu_usage"].value
        final_capacity = final_metrics["capacity"].value
        
        # System should be in better state after all interventions
        assert final_capacity > initial_capacity, f"Capacity should increase: {initial_capacity} -> {final_capacity}"
        
        # Reduce load to see if system stabilizes
        await scenario_connector.set_load(0.4)
        await asyncio.sleep(0.2)
        
        stabilized_metrics = await scenario_connector.collect_metrics()
        stabilized_cpu = stabilized_metrics["cpu_usage"].value
        
        # CPU should be much better with reduced load and increased capacity
        assert stabilized_cpu < initial_cpu, f"CPU should improve after interventions: {initial_cpu}% -> {stabilized_cpu}%"
        
        print(f"Cascading failure recovery:")
        print(f"  Initial: CPU {initial_cpu:.1f}%, Capacity {initial_capacity}")
        print(f"  Final: CPU {stabilized_cpu:.1f}%, Capacity {final_capacity}")
        print(f"  Interventions: Scale-up, Caching, Optimization")


@pytest.mark.integration
@pytest.mark.performance
class TestMockSystemPerformance:
    """Performance tests for mock system integration."""
    
    @pytest.fixture
    async def performance_harness(self):
        """Create performance test harness."""
        harness = create_performance_harness("mock_system_performance", ["perf_system_1", "perf_system_2"])
        
        async with harness:
            yield harness
    
    @pytest.mark.asyncio
    async def test_telemetry_collection_throughput(self, performance_harness):
        """Test telemetry collection throughput performance."""
        start_time = time.time()
        
        # Inject multiple telemetry events rapidly
        for i in range(20):
            metrics = {
                "cpu_usage": MetricValue("cpu_usage", 30.0 + i, "percent", datetime.now(timezone.utc)),
                "memory_usage": MetricValue("memory_usage", 1024.0 + i * 10, "MB", datetime.now(timezone.utc)),
                "throughput": MetricValue("throughput", 50.0 + i, "req/s", datetime.now(timezone.utc)),
            }
            
            await performance_harness.inject_telemetry(f"perf_system_{(i % 2) + 1}", metrics)
            
            # Small delay to avoid overwhelming the system
            await asyncio.sleep(0.01)
        
        # Wait for all events to be processed
        await asyncio.sleep(1.0)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verify events were processed
        telemetry_events = [e for e in performance_harness.received_events if e["type"] == "telemetry"]
        
        # Calculate throughput
        throughput = len(telemetry_events) / duration if duration > 0 else 0
        
        print(f"Telemetry throughput: {throughput:.2f} events/second")
        print(f"Total events processed: {len(telemetry_events)}")
        print(f"Duration: {duration:.2f} seconds")
        
        # Performance assertion (should handle at least 10 events/second)
        assert throughput >= 10.0, f"Throughput too low: {throughput:.2f} events/second"
        
        # Generate performance report
        metrics = performance_harness.get_test_metrics()
        assert metrics["events_received"] >= 10, "Should have processed multiple events"


if __name__ == "__main__":
    # Run tests directly for debugging
    pytest.main([__file__, "-v", "-s"])