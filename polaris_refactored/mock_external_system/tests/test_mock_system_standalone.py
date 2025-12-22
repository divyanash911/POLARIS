"""
Standalone Mock System Tests

This test module verifies that the mock system works independently without POLARIS.
It tests:
- State manager initialization and updates
- Metrics simulator value generation
- Action handler execution and validation
- TCP server command processing
- Configuration loading

Requirements: 3.2
"""

import asyncio
import json
import pytest
from datetime import datetime
from typing import Dict, Any

from mock_external_system.src.state_manager import StateManager, MockSystemState
from mock_external_system.src.metrics_simulator import MetricsSimulator
from mock_external_system.src.action_handler import ActionHandler, ActionType
from mock_external_system.src.server import MockSystemServer
from mock_external_system.src.protocol import Protocol


class TestStateManager:
    """Test StateManager functionality."""
    
    def test_state_manager_initialization(self):
        """Test that state manager initializes with default config."""
        manager = StateManager()
        state = manager.get_state()
        
        assert state is not None
        assert "capacity" in state
        assert "cpu_usage" in state
        assert "memory_usage" in state
        assert state["capacity"] == 5
        assert state["cpu_usage"] == 30.0
    
    def test_state_manager_with_custom_config(self):
        """Test state manager with custom configuration."""
        custom_config = {
            "baseline_metrics": {
                "cpu_usage": 50.0,
                "memory_usage": 4096.0,
                "response_time": 200.0,
                "throughput": 100.0,
                "error_rate": 1.0,
                "active_connections": 20,
                "capacity": 10
            }
        }
        
        manager = StateManager(initial_config=custom_config)
        state = manager.get_state()
        
        assert state["cpu_usage"] == 50.0
        assert state["capacity"] == 10
        assert state["memory_usage"] == 4096.0
    
    def test_state_update(self):
        """Test updating state."""
        manager = StateManager()
        
        manager.update_state({"cpu_usage": 75.0}, reason="test_update")
        state = manager.get_state()
        
        assert state["cpu_usage"] == 75.0
    
    def test_state_validation_cpu_bounds(self):
        """Test that state validation enforces CPU bounds."""
        manager = StateManager()
        
        with pytest.raises(ValueError):
            manager.update_state({"cpu_usage": 150.0})
        
        with pytest.raises(ValueError):
            manager.update_state({"cpu_usage": -10.0})
    
    def test_state_validation_capacity_bounds(self):
        """Test that state validation enforces capacity bounds."""
        manager = StateManager()
        
        with pytest.raises(ValueError):
            manager.update_state({"capacity": 0})
        
        with pytest.raises(ValueError):
            manager.update_state({"capacity": 100})
    
    def test_state_history_tracking(self):
        """Test that state changes are tracked in history."""
        manager = StateManager()
        
        manager.update_state({"cpu_usage": 40.0}, reason="first_change")
        manager.update_state({"cpu_usage": 50.0}, reason="second_change")
        
        history = manager.get_history()
        
        assert len(history) >= 2
        assert history[-1]["change_reason"] == "second_change"
    
    def test_reset_to_baseline(self):
        """Test resetting state to baseline."""
        manager = StateManager()
        
        manager.update_state({"cpu_usage": 80.0})
        manager.reset_to_baseline()
        
        state = manager.get_state()
        assert state["cpu_usage"] == 30.0
    
    def test_state_validity_check(self):
        """Test state validity checking."""
        manager = StateManager()
        
        assert manager.is_state_valid()
        
        manager.update_state({"cpu_usage": 50.0})
        assert manager.is_state_valid()


class TestMetricsSimulator:
    """Test MetricsSimulator functionality."""
    
    def test_metrics_generation(self):
        """Test that metrics are generated."""
        manager = StateManager()
        simulator = MetricsSimulator(manager)
        
        metrics = simulator.generate_metrics()
        
        assert metrics is not None
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert "response_time" in metrics
        assert "throughput" in metrics
        assert "error_rate" in metrics
        assert "active_connections" in metrics
        assert "capacity" in metrics
    
    def test_metrics_have_required_fields(self):
        """Test that each metric has required fields."""
        manager = StateManager()
        simulator = MetricsSimulator(manager)
        
        metrics = simulator.generate_metrics()
        
        for metric_name, metric_value in metrics.items():
            assert hasattr(metric_value, "name")
            assert hasattr(metric_value, "value")
            assert hasattr(metric_value, "unit")
            assert hasattr(metric_value, "timestamp")
            assert metric_value.name == metric_name
            assert isinstance(metric_value.value, (int, float))
            assert isinstance(metric_value.timestamp, datetime)
    
    def test_metrics_within_bounds(self):
        """Test that metrics stay within reasonable bounds."""
        manager = StateManager()
        simulator = MetricsSimulator(manager)
        
        # Generate multiple times to check consistency
        for _ in range(10):
            metrics = simulator.generate_metrics()
            
            # CPU and error rate should be 0-100
            assert 0 <= metrics["cpu_usage"].value <= 100
            assert 0 <= metrics["error_rate"].value <= 100
            
            # Other metrics should be non-negative
            assert metrics["memory_usage"].value >= 0
            assert metrics["response_time"].value >= 0
            assert metrics["throughput"].value >= 0
            assert metrics["active_connections"].value >= 0
            assert metrics["capacity"].value >= 0
    
    def test_metrics_dict_generation(self):
        """Test generating metrics as simple dictionary."""
        manager = StateManager()
        simulator = MetricsSimulator(manager)
        
        metrics_dict = simulator.generate_metrics_dict()
        
        assert isinstance(metrics_dict, dict)
        assert "cpu_usage" in metrics_dict
        assert isinstance(metrics_dict["cpu_usage"], (int, float))
    
    def test_load_effect_on_metrics(self):
        """Test that load affects metrics."""
        manager = StateManager()
        simulator = MetricsSimulator(manager)
        
        # Get baseline metrics
        baseline_metrics = simulator.generate_metrics_dict()
        baseline_cpu = baseline_metrics["cpu_usage"]
        
        # Apply high load
        simulator.apply_load(0.9)
        high_load_metrics = simulator.generate_metrics_dict()
        high_load_cpu = high_load_metrics["cpu_usage"]
        
        # CPU should increase with load
        assert high_load_cpu > baseline_cpu
    
    def test_action_effect_scale_up(self):
        """Test that scale-up action affects metrics."""
        manager = StateManager()
        simulator = MetricsSimulator(manager)
        
        # Get baseline
        baseline_metrics = simulator.generate_metrics_dict()
        baseline_capacity = baseline_metrics["capacity"]
        baseline_cpu = baseline_metrics["cpu_usage"]
        
        # Apply scale-up effect
        simulator.apply_action_effect("SCALE_UP", {"increment": 2})
        scaled_metrics = simulator.generate_metrics_dict()
        
        # Capacity should increase
        assert scaled_metrics["capacity"] > baseline_capacity
    
    def test_reproducibility_with_seed(self):
        """Test that metrics are reproducible with same seed."""
        manager1 = StateManager()
        simulator1 = MetricsSimulator(manager1, seed=42)
        metrics1 = simulator1.generate_metrics_dict()
        
        manager2 = StateManager()
        simulator2 = MetricsSimulator(manager2, seed=42)
        metrics2 = simulator2.generate_metrics_dict()
        
        # Metrics should be identical with same seed
        for key in metrics1:
            assert metrics1[key] == metrics2[key]


class TestActionHandler:
    """Test ActionHandler functionality."""
    
    def test_action_handler_initialization(self):
        """Test that action handler initializes."""
        manager = StateManager()
        simulator = MetricsSimulator(manager)
        handler = ActionHandler(manager, simulator)
        
        assert handler is not None
    
    def test_get_supported_actions(self):
        """Test getting list of supported actions."""
        manager = StateManager()
        simulator = MetricsSimulator(manager)
        handler = ActionHandler(manager, simulator)
        
        actions = handler.get_supported_actions()
        
        assert isinstance(actions, list)
        assert len(actions) > 0
        assert "SCALE_UP" in actions
        assert "SCALE_DOWN" in actions
    
    def test_validate_action_scale_up(self):
        """Test validating scale-up action."""
        manager = StateManager()
        simulator = MetricsSimulator(manager)
        handler = ActionHandler(manager, simulator)
        
        result = handler.validate_action("SCALE_UP", {})
        
        assert result.valid
    
    def test_validate_action_scale_down_at_minimum(self):
        """Test that scale-down is invalid at minimum capacity."""
        manager = StateManager()
        manager.update_state({"capacity": 1})
        simulator = MetricsSimulator(manager)
        handler = ActionHandler(manager, simulator)
        
        result = handler.validate_action("SCALE_DOWN", {})
        
        assert not result.valid
    
    def test_execute_scale_up_action(self):
        """Test executing scale-up action."""
        manager = StateManager()
        simulator = MetricsSimulator(manager)
        handler = ActionHandler(manager, simulator)
        
        initial_capacity = manager.get_state()["capacity"]
        
        result = handler.execute_action("SCALE_UP", {})
        
        assert result.success
        new_capacity = manager.get_state()["capacity"]
        assert new_capacity > initial_capacity
    
    def test_execute_scale_down_action(self):
        """Test executing scale-down action."""
        manager = StateManager()
        manager.update_state({"capacity": 10})
        simulator = MetricsSimulator(manager)
        handler = ActionHandler(manager, simulator)
        
        initial_capacity = manager.get_state()["capacity"]
        
        result = handler.execute_action("SCALE_DOWN", {})
        
        assert result.success
        new_capacity = manager.get_state()["capacity"]
        assert new_capacity < initial_capacity
    
    def test_action_result_has_required_fields(self):
        """Test that action result has all required fields."""
        manager = StateManager()
        simulator = MetricsSimulator(manager)
        handler = ActionHandler(manager, simulator)
        
        result = handler.execute_action("SCALE_UP", {})
        
        assert hasattr(result, "action_type")
        assert hasattr(result, "success")
        assert hasattr(result, "message")
        assert hasattr(result, "changes")
        assert hasattr(result, "timestamp")
        assert hasattr(result, "execution_time_ms")


class TestProtocol:
    """Test Protocol functionality."""
    
    def test_protocol_initialization(self):
        """Test that protocol initializes."""
        protocol = Protocol()
        assert protocol is not None
    
    def test_protocol_format_response_ok(self):
        """Test formatting OK response."""
        from mock_external_system.src.protocol import ProtocolResponse, ResponseStatus
        
        response = ProtocolResponse.ok({"status": "healthy"})
        wire_format = response.to_wire_format()
        
        assert "OK" in wire_format
        assert "status" in wire_format
    
    def test_protocol_format_response_error(self):
        """Test formatting error response."""
        from mock_external_system.src.protocol import ProtocolResponse
        
        response = ProtocolResponse.error("invalid_command", {"reason": "invalid_command"})
        wire_format = response.to_wire_format()
        
        assert "ERROR" in wire_format
        assert "invalid_command" in wire_format
    
    def test_protocol_parse_command(self):
        """Test parsing command."""
        protocol = Protocol()
        command_str = "get_metrics"
        
        parsed = protocol.parse_command(command_str)
        
        assert parsed.command == "get_metrics"
        assert parsed.args == []
    
    def test_protocol_parse_command_with_params(self):
        """Test parsing command with parameters."""
        protocol = Protocol()
        command_str = "execute_action SCALE_UP"
        
        parsed = protocol.parse_command(command_str)
        
        assert parsed.command == "execute_action"
        assert "SCALE_UP" in parsed.args


@pytest.mark.asyncio
class TestMockSystemServer:
    """Test MockSystemServer functionality."""
    
    async def test_server_initialization(self):
        """Test that server initializes."""
        server = MockSystemServer(host="localhost", port=5555)
        
        assert server is not None
        assert server.host == "localhost"
        assert server.port == 5555
    
    async def test_server_start_stop(self):
        """Test starting and stopping server."""
        server = MockSystemServer(host="localhost", port=5556)
        
        # Start server
        await server.start()
        assert server._running
        
        # Stop server
        await server.stop()
        assert not server._running
    
    async def test_server_health_check(self):
        """Test server health check."""
        server = MockSystemServer(host="localhost", port=5557)
        
        await server.start()
        
        try:
            # Connect and send health_check command
            reader, writer = await asyncio.open_connection("localhost", 5557)
            
            writer.write(b"health_check\n")
            await writer.drain()
            
            response = await reader.readline()
            response_str = response.decode().strip()
            
            assert "OK" in response_str or "healthy" in response_str
            
            writer.close()
            await writer.wait_closed()
        finally:
            await server.stop()
    
    async def test_server_get_metrics(self):
        """Test server get_metrics command."""
        server = MockSystemServer(host="localhost", port=5558)
        
        await server.start()
        
        try:
            reader, writer = await asyncio.open_connection("localhost", 5558)
            
            writer.write(b"get_metrics\n")
            await writer.drain()
            
            response = await reader.readline()
            response_str = response.decode().strip()
            
            assert "OK" in response_str
            assert "cpu_usage" in response_str or "metrics" in response_str
            
            writer.close()
            await writer.wait_closed()
        finally:
            await server.stop()
    
    async def test_server_get_state(self):
        """Test server get_state command."""
        server = MockSystemServer(host="localhost", port=5559)
        
        await server.start()
        
        try:
            reader, writer = await asyncio.open_connection("localhost", 5559)
            
            writer.write(b"get_state\n")
            await writer.drain()
            
            response = await reader.readline()
            response_str = response.decode().strip()
            
            assert "OK" in response_str
            
            writer.close()
            await writer.wait_closed()
        finally:
            await server.stop()


class TestIntegration:
    """Integration tests for mock system components."""
    
    def test_full_workflow(self):
        """Test complete workflow: initialize, generate metrics, execute action."""
        # Initialize
        config = {
            "baseline_metrics": {
                "cpu_usage": 30.0,
                "memory_usage": 2048.0,
                "response_time": 100.0,
                "throughput": 50.0,
                "error_rate": 0.5,
                "active_connections": 10,
                "capacity": 5
            }
        }
        
        manager = StateManager(initial_config=config)
        simulator = MetricsSimulator(manager)
        handler = ActionHandler(manager, simulator)
        
        # Get initial metrics
        initial_metrics = simulator.generate_metrics_dict()
        assert initial_metrics["capacity"] == 5
        
        # Execute scale-up
        result = handler.execute_action("SCALE_UP", {})
        assert result.success
        
        # Verify state changed
        new_metrics = simulator.generate_metrics_dict()
        assert new_metrics["capacity"] > initial_metrics["capacity"]
    
    def test_state_consistency_across_components(self):
        """Test that state is consistent across all components."""
        manager = StateManager()
        simulator = MetricsSimulator(manager)
        handler = ActionHandler(manager, simulator)
        
        # Update state
        manager.update_state({"cpu_usage": 60.0})
        
        # Get state from different components
        state1 = manager.get_state()
        metrics = simulator.generate_metrics_dict()
        
        # CPU usage should be consistent
        assert state1["cpu_usage"] == 60.0
        # Metrics should reflect the state
        assert metrics["cpu_usage"] >= 50.0  # Allow some noise
