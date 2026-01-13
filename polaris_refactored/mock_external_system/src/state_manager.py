"""
State Manager - Manages mock system state.

This module provides the StateManager class that handles:
- System state initialization from configuration
- State updates and history tracking
- State validation and consistency checks
- Baseline configuration loading from YAML

"""

import copy
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class MockSystemState:
    """Internal state of mock system."""
    capacity: int = 5
    cpu_usage: float = 30.0
    memory_usage: float = 2048.0
    response_time: float = 100.0
    throughput: float = 50.0
    error_rate: float = 0.5
    active_connections: int = 10
    load_level: float = 0.5
    caching_enabled: bool = False
    last_action: Optional[str] = None
    last_action_time: Optional[datetime] = None
    uptime_seconds: float = 0.0


@dataclass
class StateHistoryEntry:
    """Entry in state change history."""
    timestamp: datetime
    previous_state: Dict[str, Any]
    new_state: Dict[str, Any]
    change_reason: str


class StateManager:
    """Manages mock system state.
    
    This class handles:
    - Loading baseline configuration from YAML files
    - Managing current system state
    - Tracking state change history
    - Validating state consistency
    """
    
    def __init__(self, initial_config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        """Initialize the state manager.
        
        Args:
            initial_config: Optional dictionary with initial configuration.
                           If not provided, loads from config_path or default.
            config_path: Optional path to YAML configuration file.
        """
        self._config = self._load_config(initial_config, config_path)
        self._state = self._initialize_state()
        self._history: List[StateHistoryEntry] = []
        self._start_time = datetime.now()
        self._max_history_size = 1000

    def _load_config(self, initial_config: Optional[Dict[str, Any]], config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from provided dict, file, or default.
        
        Args:
            initial_config: Optional dictionary with configuration.
            config_path: Optional path to YAML configuration file.
            
        Returns:
            Configuration dictionary.
        """
        if initial_config is not None:
            return copy.deepcopy(initial_config)
        
        if config_path is not None:
            return self._load_yaml_config(config_path)
        
        # Try to load default config
        default_path = self._get_default_config_path()
        if default_path and os.path.exists(default_path):
            return self._load_yaml_config(default_path)
        
        # Return hardcoded defaults if no config file found
        return self._get_default_config()
    
    def _get_default_config_path(self) -> Optional[str]:
        """Get the path to the default configuration file.
        
        Returns:
            Path to default_config.yaml or None if not found.
        """
        # Try relative to this file
        current_dir = Path(__file__).parent.parent
        config_path = current_dir / "config" / "default_config.yaml"
        if config_path.exists():
            return str(config_path)
        return None
    
    def _load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file.
            
        Returns:
            Configuration dictionary.
            
        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid YAML.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config if config else {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values.
        
        Returns:
            Default configuration dictionary.
        """
        return {
            "server": {
                "host": "localhost",
                "port": 5000,
                "max_connections": 100
            },
            "baseline_metrics": {
                "cpu_usage": 30.0,
                "memory_usage": 2048.0,
                "response_time": 100.0,
                "throughput": 50.0,
                "error_rate": 0.5,
                "active_connections": 10,
                "capacity": 5
            },
            "simulation": {
                "noise_factor": 0.1,
                "update_interval": 1.0,
                "load_response_time": 2.0
            },
            "capacity": {
                "min_capacity": 1,
                "max_capacity": 20,
                "scale_up_increment": 2,
                "scale_down_increment": 1
            }
        }

    def _initialize_state(self) -> MockSystemState:
        """Initialize state from baseline configuration.
        
        Returns:
            Initialized MockSystemState object.
        """
        baseline = self._config.get("baseline_metrics", {})
        
        return MockSystemState(
            capacity=int(baseline.get("capacity", 5)),
            cpu_usage=float(baseline.get("cpu_usage", 30.0)),
            memory_usage=float(baseline.get("memory_usage", 2048.0)),
            response_time=float(baseline.get("response_time", 100.0)),
            throughput=float(baseline.get("throughput", 50.0)),
            error_rate=float(baseline.get("error_rate", 0.5)),
            active_connections=int(baseline.get("active_connections", 10)),
            load_level=0.5,
            caching_enabled=False,
            last_action=None,
            last_action_time=None,
            uptime_seconds=0.0
        )
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration.
        
        Returns:
            Configuration dictionary.
        """
        return copy.deepcopy(self._config)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current system state as dictionary.
        
        Returns:
            Dictionary representation of current state.
        """
        # Update uptime
        self._state.uptime_seconds = (datetime.now() - self._start_time).total_seconds()
        
        return {
            "capacity": self._state.capacity,
            "cpu_usage": self._state.cpu_usage,
            "memory_usage": self._state.memory_usage,
            "response_time": self._state.response_time,
            "throughput": self._state.throughput,
            "error_rate": self._state.error_rate,
            "active_connections": self._state.active_connections,
            "load_level": self._state.load_level,
            "caching_enabled": self._state.caching_enabled,
            "last_action": self._state.last_action,
            "last_action_time": self._state.last_action_time.isoformat() if self._state.last_action_time else None,
            "uptime_seconds": self._state.uptime_seconds
        }
    
    def get_state_object(self) -> MockSystemState:
        """Get the current state object.
        
        Returns:
            MockSystemState object.
        """
        self._state.uptime_seconds = (datetime.now() - self._start_time).total_seconds()
        return self._state
    
    def update_state(self, updates: Dict[str, Any], reason: str = "manual_update") -> None:
        """Update system state with validation.
        
        Args:
            updates: Dictionary of state fields to update.
            reason: Reason for the state change.
            
        Raises:
            ValueError: If updates would result in invalid state.
        """
        # Capture previous state
        previous_state = self.get_state()
        
        # Validate updates
        self._validate_updates(updates)
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)
        
        # Record history
        new_state = self.get_state()
        self._record_history(previous_state, new_state, reason)

    def _validate_updates(self, updates: Dict[str, Any]) -> None:
        """Validate state updates for consistency.
        
        Args:
            updates: Dictionary of state fields to update.
            
        Raises:
            ValueError: If updates would result in invalid state.
        """
        capacity_config = self._config.get("capacity", {})
        min_capacity = capacity_config.get("min_capacity", 1)
        max_capacity = capacity_config.get("max_capacity", 20)
        
        # Validate capacity bounds
        if "capacity" in updates:
            capacity = updates["capacity"]
            if capacity < min_capacity:
                raise ValueError(f"Capacity {capacity} below minimum {min_capacity}")
            if capacity > max_capacity:
                raise ValueError(f"Capacity {capacity} above maximum {max_capacity}")
        
        # Validate percentage values (0-100)
        percentage_fields = ["cpu_usage", "error_rate"]
        for field in percentage_fields:
            if field in updates:
                value = updates[field]
                if value < 0:
                    raise ValueError(f"{field} cannot be negative: {value}")
                if value > 100:
                    raise ValueError(f"{field} cannot exceed 100: {value}")
        
        # Validate non-negative values
        non_negative_fields = ["memory_usage", "response_time", "throughput", "active_connections"]
        for field in non_negative_fields:
            if field in updates:
                value = updates[field]
                if value < 0:
                    raise ValueError(f"{field} cannot be negative: {value}")
        
        # Validate load_level (0.0 to 1.0)
        if "load_level" in updates:
            load_level = updates["load_level"]
            if load_level < 0.0 or load_level > 1.0:
                raise ValueError(f"load_level must be between 0.0 and 1.0: {load_level}")
    
    def _record_history(self, previous_state: Dict[str, Any], new_state: Dict[str, Any], reason: str) -> None:
        """Record state change in history.
        
        Args:
            previous_state: State before change.
            new_state: State after change.
            reason: Reason for the change.
        """
        entry = StateHistoryEntry(
            timestamp=datetime.now(),
            previous_state=previous_state,
            new_state=new_state,
            change_reason=reason
        )
        self._history.append(entry)
        
        # Trim history if too large
        if len(self._history) > self._max_history_size:
            self._history = self._history[-self._max_history_size:]
    
    def reset_to_baseline(self) -> None:
        """Reset state to baseline configuration."""
        previous_state = self.get_state()
        self._state = self._initialize_state()
        self._start_time = datetime.now()
        new_state = self.get_state()
        self._record_history(previous_state, new_state, "reset_to_baseline")
    
    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get state change history.
        
        Args:
            limit: Maximum number of history entries to return.
            
        Returns:
            List of history entries as dictionaries.
        """
        entries = self._history[-limit:] if limit > 0 else self._history
        return [
            {
                "timestamp": entry.timestamp.isoformat(),
                "previous_state": entry.previous_state,
                "new_state": entry.new_state,
                "change_reason": entry.change_reason
            }
            for entry in entries
        ]
    
    def get_baseline_metrics(self) -> Dict[str, Any]:
        """Get baseline metrics from configuration.
        
        Returns:
            Baseline metrics dictionary.
        """
        return copy.deepcopy(self._config.get("baseline_metrics", {}))
    
    def get_capacity_config(self) -> Dict[str, Any]:
        """Get capacity configuration.
        
        Returns:
            Capacity configuration dictionary.
        """
        return copy.deepcopy(self._config.get("capacity", {}))
    
    def get_simulation_config(self) -> Dict[str, Any]:
        """Get simulation configuration.
        
        Returns:
            Simulation configuration dictionary.
        """
        return copy.deepcopy(self._config.get("simulation", {}))
    
    def is_state_valid(self) -> bool:
        """Check if current state is valid.
        
        Returns:
            True if state is valid, False otherwise.
        """
        try:
            state = self.get_state()
            self._validate_updates(state)
            return True
        except ValueError:
            return False
    
    def record_action(self, action_type: str) -> None:
        """Record that an action was executed.
        
        Args:
            action_type: Type of action that was executed.
        """
        self._state.last_action = action_type
        self._state.last_action_time = datetime.now()
