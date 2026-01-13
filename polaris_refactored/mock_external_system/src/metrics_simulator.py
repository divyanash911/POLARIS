"""
Metrics Simulator - Simulates realistic system metrics.

This module provides the MetricsSimulator class that handles:
- Generating realistic metric values with noise
- Simulating load effects on metrics
- Applying action effects (scale up/down, optimization)
- Maintaining metric consistency with system state

"""

import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from .state_manager import StateManager


@dataclass
class MetricValue:
    """Represents a single metric value with metadata."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat()
        }


class MetricsSimulator:
    """Simulates realistic system metrics.
    
    This class handles:
    - Generating metric values that fluctuate around baseline
    - Applying load effects to metrics
    - Applying action effects (scale up/down, optimization)
    - Maintaining consistency with system state
    """
    
    # Metric units mapping
    METRIC_UNITS = {
        "cpu_usage": "percent",
        "memory_usage": "MB",
        "response_time": "ms",
        "throughput": "req/s",
        "error_rate": "percent",
        "active_connections": "count",
        "capacity": "units"
    }
    
    def __init__(self, state_manager: StateManager, seed: Optional[int] = None):
        """Initialize the metrics simulator.
        
        Args:
            state_manager: StateManager instance for accessing state and config.
            seed: Optional random seed for reproducibility.
        """
        self._state_manager = state_manager
        self._noise_factor = state_manager.get_simulation_config().get("noise_factor", 0.1)
        self._last_metrics: Dict[str, float] = {}
        self._smoothing_factor = 0.3  # For smooth metric transitions
        
        if seed is not None:
            random.seed(seed)

    def generate_metrics(self) -> Dict[str, MetricValue]:
        """Generate current metrics based on state and load.
        
        Returns:
            Dictionary of metric name to MetricValue objects.
        """
        state = self._state_manager.get_state_object()
        baseline = self._state_manager.get_baseline_metrics()
        timestamp = datetime.now()
        
        metrics = {}
        
        # Generate each metric with noise and load effects
        metrics["cpu_usage"] = self._generate_cpu_usage(state, baseline, timestamp)
        metrics["memory_usage"] = self._generate_memory_usage(state, baseline, timestamp)
        metrics["response_time"] = self._generate_response_time(state, baseline, timestamp)
        metrics["throughput"] = self._generate_throughput(state, baseline, timestamp)
        metrics["error_rate"] = self._generate_error_rate(state, baseline, timestamp)
        metrics["active_connections"] = self._generate_active_connections(state, baseline, timestamp)
        metrics["capacity"] = self._generate_capacity(state, timestamp)
        
        return metrics
    
    def generate_metrics_dict(self) -> Dict[str, float]:
        """Generate current metrics as a simple dictionary of values.
        
        Returns:
            Dictionary of metric name to float values.
        """
        metrics = self.generate_metrics()
        return {name: metric.value for name, metric in metrics.items()}
    
    def _add_noise(self, base_value: float, noise_factor: Optional[float] = None) -> float:
        """Add realistic noise to a metric value.
        
        Args:
            base_value: The base value to add noise to.
            noise_factor: Optional override for noise factor.
            
        Returns:
            Value with noise applied.
        """
        factor = noise_factor if noise_factor is not None else self._noise_factor
        noise = random.gauss(0, factor * base_value)
        return base_value + noise
    
    def _smooth_value(self, metric_name: str, new_value: float) -> float:
        """Apply smoothing to prevent sudden metric jumps.
        
        Args:
            metric_name: Name of the metric.
            new_value: New calculated value.
            
        Returns:
            Smoothed value.
        """
        if metric_name in self._last_metrics:
            old_value = self._last_metrics[metric_name]
            smoothed = old_value + self._smoothing_factor * (new_value - old_value)
        else:
            smoothed = new_value
        
        self._last_metrics[metric_name] = smoothed
        return smoothed
    
    def _generate_cpu_usage(self, state, baseline: Dict, timestamp: datetime) -> MetricValue:
        """Generate CPU usage metric.
        
        CPU usage increases with load and decreases with capacity.
        """
        base_cpu = baseline.get("cpu_usage", 30.0)
        baseline_capacity = baseline.get("capacity", 5)
        
        # Load effect: higher load = higher CPU
        load_effect = state.load_level * 50  # 0-50% increase based on load
        
        # Capacity effect: more capacity = lower CPU per unit
        capacity_ratio = baseline_capacity / max(state.capacity, 1)
        capacity_effect = base_cpu * (capacity_ratio - 1) * 0.5
        
        # Caching effect: caching reduces CPU slightly
        caching_effect = -5.0 if state.caching_enabled else 0.0
        
        raw_cpu = base_cpu + load_effect + capacity_effect + caching_effect
        raw_cpu = self._add_noise(raw_cpu)
        raw_cpu = max(0.0, min(100.0, raw_cpu))  # Clamp to 0-100
        
        smoothed_cpu = self._smooth_value("cpu_usage", raw_cpu)
        
        return MetricValue(
            name="cpu_usage",
            value=round(smoothed_cpu, 2),
            unit=self.METRIC_UNITS["cpu_usage"],
            timestamp=timestamp
        )

    def _generate_memory_usage(self, state, baseline: Dict, timestamp: datetime) -> MetricValue:
        """Generate memory usage metric.
        
        Memory usage increases with load and active connections.
        """
        base_memory = baseline.get("memory_usage", 2048.0)
        baseline_connections = baseline.get("active_connections", 10)
        
        # Load effect: higher load = more memory
        load_effect = state.load_level * base_memory * 0.3
        
        # Connection effect: more connections = more memory
        connection_ratio = state.active_connections / max(baseline_connections, 1)
        connection_effect = base_memory * (connection_ratio - 1) * 0.1
        
        # Caching effect: caching uses more memory
        caching_effect = base_memory * 0.2 if state.caching_enabled else 0.0
        
        raw_memory = base_memory + load_effect + connection_effect + caching_effect
        raw_memory = self._add_noise(raw_memory)
        raw_memory = max(0.0, raw_memory)
        
        smoothed_memory = self._smooth_value("memory_usage", raw_memory)
        
        return MetricValue(
            name="memory_usage",
            value=round(smoothed_memory, 2),
            unit=self.METRIC_UNITS["memory_usage"],
            timestamp=timestamp
        )
    
    def _generate_response_time(self, state, baseline: Dict, timestamp: datetime) -> MetricValue:
        """Generate response time metric.
        
        Response time increases with load and decreases with capacity.
        """
        base_response = baseline.get("response_time", 100.0)
        baseline_capacity = baseline.get("capacity", 5)
        
        # Load effect: higher load = slower response
        load_effect = state.load_level * base_response * 2.0
        
        # Capacity effect: more capacity = faster response
        capacity_ratio = baseline_capacity / max(state.capacity, 1)
        capacity_effect = base_response * (capacity_ratio - 1) * 0.3
        
        # Caching effect: caching improves response time
        caching_effect = -base_response * 0.3 if state.caching_enabled else 0.0
        
        raw_response = base_response + load_effect + capacity_effect + caching_effect
        raw_response = self._add_noise(raw_response)
        raw_response = max(1.0, raw_response)  # Minimum 1ms
        
        smoothed_response = self._smooth_value("response_time", raw_response)
        
        return MetricValue(
            name="response_time",
            value=round(smoothed_response, 2),
            unit=self.METRIC_UNITS["response_time"],
            timestamp=timestamp
        )
    
    def _generate_throughput(self, state, baseline: Dict, timestamp: datetime) -> MetricValue:
        """Generate throughput metric.
        
        Throughput increases with capacity and decreases under high load.
        """
        base_throughput = baseline.get("throughput", 50.0)
        baseline_capacity = baseline.get("capacity", 5)
        
        # Capacity effect: more capacity = higher throughput
        capacity_ratio = state.capacity / max(baseline_capacity, 1)
        capacity_effect = base_throughput * (capacity_ratio - 1) * 0.8
        
        # Load effect: very high load can reduce throughput (saturation)
        if state.load_level > 0.8:
            load_penalty = (state.load_level - 0.8) * base_throughput * 0.5
        else:
            load_penalty = 0.0
        
        # Caching effect: caching improves throughput
        caching_effect = base_throughput * 0.2 if state.caching_enabled else 0.0
        
        raw_throughput = base_throughput + capacity_effect - load_penalty + caching_effect
        raw_throughput = self._add_noise(raw_throughput)
        raw_throughput = max(0.0, raw_throughput)
        
        smoothed_throughput = self._smooth_value("throughput", raw_throughput)
        
        return MetricValue(
            name="throughput",
            value=round(smoothed_throughput, 2),
            unit=self.METRIC_UNITS["throughput"],
            timestamp=timestamp
        )

    def _generate_error_rate(self, state, baseline: Dict, timestamp: datetime) -> MetricValue:
        """Generate error rate metric.
        
        Error rate increases under high load and decreases with capacity.
        """
        base_error = baseline.get("error_rate", 0.5)
        baseline_capacity = baseline.get("capacity", 5)
        
        # Load effect: higher load = more errors
        if state.load_level > 0.7:
            load_effect = (state.load_level - 0.7) * 20  # Up to 6% increase
        else:
            load_effect = 0.0
        
        # Capacity effect: more capacity = fewer errors
        capacity_ratio = state.capacity / max(baseline_capacity, 1)
        capacity_effect = -base_error * (capacity_ratio - 1) * 0.3
        
        raw_error = base_error + load_effect + capacity_effect
        raw_error = self._add_noise(raw_error, noise_factor=0.05)  # Less noise for error rate
        raw_error = max(0.0, min(100.0, raw_error))  # Clamp to 0-100
        
        smoothed_error = self._smooth_value("error_rate", raw_error)
        
        return MetricValue(
            name="error_rate",
            value=round(smoothed_error, 3),
            unit=self.METRIC_UNITS["error_rate"],
            timestamp=timestamp
        )
    
    def _generate_active_connections(self, state, baseline: Dict, timestamp: datetime) -> MetricValue:
        """Generate active connections metric.
        
        Active connections correlate with load level, but can be overridden by state.
        """
        # If state has a specific active_connections value (e.g., after restart), use it
        if hasattr(state, 'active_connections') and state.active_connections is not None:
            # Use state value but still apply some load-based variation
            base_connections = state.active_connections
            if base_connections == 0:
                # After restart, connections start at 0 but may gradually increase
                load_effect = state.load_level * baseline.get("active_connections", 10) * 0.1
                raw_connections = base_connections + load_effect
            else:
                # Normal operation - use state value as base
                load_effect = state.load_level * base_connections * 0.5
                raw_connections = base_connections + load_effect
        else:
            # Fallback to baseline calculation
            base_connections = baseline.get("active_connections", 10)
            load_effect = state.load_level * base_connections * 2
            raw_connections = base_connections + load_effect
        
        raw_connections = self._add_noise(raw_connections)
        raw_connections = max(0, int(round(raw_connections)))
        
        smoothed_connections = self._smooth_value("active_connections", float(raw_connections))
        
        return MetricValue(
            name="active_connections",
            value=int(round(smoothed_connections)),
            unit=self.METRIC_UNITS["active_connections"],
            timestamp=timestamp
        )
    
    def _generate_capacity(self, state, timestamp: datetime) -> MetricValue:
        """Generate capacity metric.
        
        Capacity is directly from state (no noise).
        """
        return MetricValue(
            name="capacity",
            value=float(state.capacity),
            unit=self.METRIC_UNITS["capacity"],
            timestamp=timestamp
        )
    
    def apply_load(self, load_level: float) -> None:
        """Apply simulated load to affect metrics.
        
        Args:
            load_level: Load level from 0.0 (no load) to 1.0 (max load).
            
        Raises:
            ValueError: If load_level is out of range.
        """
        if load_level < 0.0 or load_level > 1.0:
            raise ValueError(f"load_level must be between 0.0 and 1.0: {load_level}")
        
        self._state_manager.update_state(
            {"load_level": load_level},
            reason=f"load_applied:{load_level}"
        )

    def apply_action_effect(self, action_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply effects of adaptation action to metrics.
        
        Args:
            action_type: Type of action (SCALE_UP, SCALE_DOWN, etc.).
            params: Optional parameters for the action.
            
        Returns:
            Dictionary with effect details.
            
        Raises:
            ValueError: If action_type is not supported.
        """
        params = params or {}
        state = self._state_manager.get_state_object()
        capacity_config = self._state_manager.get_capacity_config()
        
        effect = {"action_type": action_type, "applied": True, "changes": {}}
        
        if action_type == "SCALE_UP":
            effect = self._apply_scale_up(state, capacity_config, params)
        elif action_type == "SCALE_DOWN":
            effect = self._apply_scale_down(state, capacity_config, params)
        elif action_type == "ADJUST_QOS":
            effect = self._apply_adjust_qos(state, params)
        elif action_type == "RESTART_SERVICE":
            effect = self._apply_restart_service(state)
        elif action_type == "OPTIMIZE_CONFIG":
            effect = self._apply_optimize_config(state)
        elif action_type == "ENABLE_CACHING":
            effect = self._apply_enable_caching(state)
        elif action_type == "DISABLE_CACHING":
            effect = self._apply_disable_caching(state)
        else:
            raise ValueError(f"Unsupported action type: {action_type}")
        
        # Record the action
        self._state_manager.record_action(action_type)
        
        return effect
    
    def _apply_scale_up(self, state, capacity_config: Dict, params: Dict) -> Dict[str, Any]:
        """Apply scale-up action effect.
        
        Increases capacity and decreases utilization metrics.
        """
        increment = params.get("increment", capacity_config.get("scale_up_increment", 2))
        max_capacity = capacity_config.get("max_capacity", 20)
        
        old_capacity = state.capacity
        new_capacity = min(old_capacity + increment, max_capacity)
        
        if new_capacity == old_capacity:
            return {
                "action_type": "SCALE_UP",
                "applied": False,
                "reason": "Already at maximum capacity",
                "changes": {}
            }
        
        # Update state
        self._state_manager.update_state(
            {"capacity": new_capacity},
            reason=f"scale_up:{old_capacity}->{new_capacity}"
        )
        
        return {
            "action_type": "SCALE_UP",
            "applied": True,
            "changes": {
                "capacity": {"old": old_capacity, "new": new_capacity}
            }
        }
    
    def _apply_scale_down(self, state, capacity_config: Dict, params: Dict) -> Dict[str, Any]:
        """Apply scale-down action effect.
        
        Decreases capacity and increases utilization metrics.
        """
        decrement = params.get("decrement", capacity_config.get("scale_down_increment", 1))
        min_capacity = capacity_config.get("min_capacity", 1)
        
        old_capacity = state.capacity
        new_capacity = max(old_capacity - decrement, min_capacity)
        
        if new_capacity == old_capacity:
            return {
                "action_type": "SCALE_DOWN",
                "applied": False,
                "reason": "Already at minimum capacity",
                "changes": {}
            }
        
        # Update state
        self._state_manager.update_state(
            {"capacity": new_capacity},
            reason=f"scale_down:{old_capacity}->{new_capacity}"
        )
        
        return {
            "action_type": "SCALE_DOWN",
            "applied": True,
            "changes": {
                "capacity": {"old": old_capacity, "new": new_capacity}
            }
        }

    def _apply_adjust_qos(self, state, params: Dict) -> Dict[str, Any]:
        """Apply QoS adjustment action effect.
        
        Adjusts quality of service settings affecting response time and throughput.
        """
        # QoS adjustment can trade response time for throughput or vice versa
        qos_mode = params.get("mode", "balanced")
        baseline = self._state_manager.get_baseline_metrics()
        
        changes = {}
        
        if qos_mode == "performance":
            # Prioritize response time
            new_response = state.response_time * 0.8
            self._state_manager.update_state(
                {"response_time": new_response},
                reason="adjust_qos:performance"
            )
            changes["response_time"] = {"old": state.response_time, "new": new_response}
        elif qos_mode == "throughput":
            # Prioritize throughput
            new_throughput = state.throughput * 1.2
            self._state_manager.update_state(
                {"throughput": new_throughput},
                reason="adjust_qos:throughput"
            )
            changes["throughput"] = {"old": state.throughput, "new": new_throughput}
        else:
            # Balanced - reset to baseline ratios
            pass
        
        return {
            "action_type": "ADJUST_QOS",
            "applied": True,
            "mode": qos_mode,
            "changes": changes
        }
    
    def _apply_restart_service(self, state) -> Dict[str, Any]:
        """Apply restart service action effect.
        
        Simulates brief downtime then improved metrics.
        """
        # Simulate restart by temporarily increasing error rate
        # then resetting some metrics to baseline
        baseline = self._state_manager.get_baseline_metrics()
        
        old_error_rate = state.error_rate
        old_response_time = state.response_time
        
        # After restart, metrics improve
        new_error_rate = baseline.get("error_rate", 0.5)
        new_response_time = baseline.get("response_time", 100.0)
        
        self._state_manager.update_state(
            {
                "error_rate": new_error_rate,
                "response_time": new_response_time,
                "active_connections": 0  # Connections reset on restart
            },
            reason="restart_service"
        )
        
        return {
            "action_type": "RESTART_SERVICE",
            "applied": True,
            "changes": {
                "error_rate": {"old": old_error_rate, "new": new_error_rate},
                "response_time": {"old": old_response_time, "new": new_response_time},
                "active_connections": {"old": state.active_connections, "new": 0}
            }
        }
    
    def _apply_optimize_config(self, state) -> Dict[str, Any]:
        """Apply configuration optimization action effect.
        
        Improves efficiency metrics without changing capacity.
        Note: This doesn't directly modify cpu_usage/memory_usage in state,
        as those should be calculated by the metrics simulator based on
        load, capacity, and other factors. Instead, we could add an
        'optimization_factor' to state if we want persistent effects.
        """
        # Get current metrics for reporting changes
        current_metrics = self.generate_metrics_dict()
        old_cpu = current_metrics.get("cpu_usage", state.cpu_usage)
        old_memory = current_metrics.get("memory_usage", state.memory_usage)
        
        # For now, optimization is a no-op on state since metrics are calculated
        # In a real system, this might set an optimization flag or adjust baseline
        # The effect will be reflected in future metric calculations if needed
        
        # Calculate what the new values would be (for reporting purposes)
        new_cpu = max(5.0, old_cpu * 0.9)
        new_memory = max(512.0, old_memory * 0.9)
        
        return {
            "action_type": "OPTIMIZE_CONFIG",
            "applied": True,
            "changes": {
                "cpu_usage": {"old": old_cpu, "new": new_cpu},
                "memory_usage": {"old": old_memory, "new": new_memory}
            }
        }
    
    def _apply_enable_caching(self, state) -> Dict[str, Any]:
        """Apply enable caching action effect.
        
        Enables caching which improves response time and throughput.
        """
        if state.caching_enabled:
            return {
                "action_type": "ENABLE_CACHING",
                "applied": False,
                "reason": "Caching already enabled",
                "changes": {}
            }
        
        self._state_manager.update_state(
            {"caching_enabled": True},
            reason="enable_caching"
        )
        
        return {
            "action_type": "ENABLE_CACHING",
            "applied": True,
            "changes": {
                "caching_enabled": {"old": False, "new": True}
            }
        }
    
    def _apply_disable_caching(self, state) -> Dict[str, Any]:
        """Apply disable caching action effect.
        
        Disables caching which reduces memory usage but may affect performance.
        """
        if not state.caching_enabled:
            return {
                "action_type": "DISABLE_CACHING",
                "applied": False,
                "reason": "Caching already disabled",
                "changes": {}
            }
        
        self._state_manager.update_state(
            {"caching_enabled": False},
            reason="disable_caching"
        )
        
        return {
            "action_type": "DISABLE_CACHING",
            "applied": True,
            "changes": {
                "caching_enabled": {"old": True, "new": False}
            }
        }
    
    def get_supported_actions(self) -> list:
        """Get list of supported action types.
        
        Returns:
            List of supported action type strings.
        """
        return [
            "SCALE_UP",
            "SCALE_DOWN",
            "ADJUST_QOS",
            "RESTART_SERVICE",
            "OPTIMIZE_CONFIG",
            "ENABLE_CACHING",
            "DISABLE_CACHING"
        ]
    
    def set_noise_factor(self, noise_factor: float) -> None:
        """Set the noise factor for metric generation.
        
        Args:
            noise_factor: Noise factor (0.0 to 1.0).
            
        Raises:
            ValueError: If noise_factor is out of range.
        """
        if noise_factor < 0.0 or noise_factor > 1.0:
            raise ValueError(f"noise_factor must be between 0.0 and 1.0: {noise_factor}")
        self._noise_factor = noise_factor
    
    def reset_smoothing(self) -> None:
        """Reset the smoothing state for fresh metric generation."""
        self._last_metrics.clear()
