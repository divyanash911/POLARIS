"""
Action Handler - Handles adaptation actions from POLARIS.

This module provides the ActionHandler class that handles:
- Action execution with validation
- All supported actions (SCALE_UP, SCALE_DOWN, ADJUST_QOS, etc.)
- Action validation with constraint checking
- Integration with state manager and metrics simulator

Requirements: 1.4, 2.2, 2.3
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .state_manager import StateManager
from .metrics_simulator import MetricsSimulator


class ActionType(Enum):
    """Supported action types."""
    SCALE_UP = "SCALE_UP"
    SCALE_DOWN = "SCALE_DOWN"
    ADJUST_QOS = "ADJUST_QOS"
    RESTART_SERVICE = "RESTART_SERVICE"
    OPTIMIZE_CONFIG = "OPTIMIZE_CONFIG"
    ENABLE_CACHING = "ENABLE_CACHING"
    DISABLE_CACHING = "DISABLE_CACHING"


@dataclass
class ActionResult:
    """Result of an action execution."""
    action_type: str
    success: bool
    message: str
    changes: Dict[str, Any]
    timestamp: datetime
    execution_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "action_type": self.action_type,
            "success": self.success,
            "message": self.message,
            "changes": self.changes,
            "timestamp": self.timestamp.isoformat(),
            "execution_time_ms": self.execution_time_ms
        }


@dataclass
class ValidationResult:
    """Result of action validation."""
    valid: bool
    reason: str
    constraints_checked: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "valid": self.valid,
            "reason": self.reason,
            "constraints_checked": self.constraints_checked
        }


class ActionHandler:
    """Handles adaptation actions from POLARIS.
    
    This class handles:
    - Executing adaptation actions with validation
    - Supporting all action types (SCALE_UP, SCALE_DOWN, etc.)
    - Validating actions against system constraints
    - Integrating with state manager and metrics simulator
    """
    
    def __init__(self, state_manager: StateManager, metrics_simulator: MetricsSimulator):
        """Initialize the action handler.
        
        Args:
            state_manager: StateManager instance for state access.
            metrics_simulator: MetricsSimulator instance for applying effects.
        """
        self._state_manager = state_manager
        self._metrics_simulator = metrics_simulator
        self._action_history: List[ActionResult] = []
        self._max_history_size = 1000
    
    def execute_action(self, action_type: str, parameters: Optional[Dict[str, Any]] = None) -> ActionResult:
        """Execute an adaptation action and return result.
        
        Args:
            action_type: Type of action to execute.
            parameters: Optional parameters for the action.
            
        Returns:
            ActionResult with execution details.
        """
        import time
        start_time = time.time()
        parameters = parameters or {}
        
        # Validate action first
        validation = self.validate_action(action_type, parameters)
        if not validation.valid:
            result = ActionResult(
                action_type=action_type,
                success=False,
                message=f"Validation failed: {validation.reason}",
                changes={},
                timestamp=datetime.now(),
                execution_time_ms=0.0
            )
            self._record_action(result)
            return result
        
        try:
            # Execute the action through metrics simulator
            effect = self._metrics_simulator.apply_action_effect(action_type, parameters)
            
            execution_time = (time.time() - start_time) * 1000
            
            if effect.get("applied", False):
                result = ActionResult(
                    action_type=action_type,
                    success=True,
                    message=f"Action {action_type} executed successfully",
                    changes=effect.get("changes", {}),
                    timestamp=datetime.now(),
                    execution_time_ms=execution_time
                )
            else:
                result = ActionResult(
                    action_type=action_type,
                    success=False,
                    message=effect.get("reason", "Action not applied"),
                    changes={},
                    timestamp=datetime.now(),
                    execution_time_ms=execution_time
                )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            result = ActionResult(
                action_type=action_type,
                success=False,
                message=f"Action execution failed: {str(e)}",
                changes={},
                timestamp=datetime.now(),
                execution_time_ms=execution_time
            )
        
        self._record_action(result)
        return result

    def validate_action(self, action_type: str, parameters: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate if action can be executed.
        
        Args:
            action_type: Type of action to validate.
            parameters: Optional parameters for the action.
            
        Returns:
            ValidationResult with validation details.
        """
        parameters = parameters or {}
        constraints_checked = []
        
        # Check if action type is supported
        constraints_checked.append("action_type_supported")
        if action_type not in self.get_supported_actions():
            return ValidationResult(
                valid=False,
                reason=f"Unsupported action type: {action_type}",
                constraints_checked=constraints_checked
            )
        
        # Get current state and config
        state = self._state_manager.get_state_object()
        capacity_config = self._state_manager.get_capacity_config()
        
        # Action-specific validation
        if action_type == "SCALE_UP":
            return self._validate_scale_up(state, capacity_config, parameters, constraints_checked)
        elif action_type == "SCALE_DOWN":
            return self._validate_scale_down(state, capacity_config, parameters, constraints_checked)
        elif action_type == "ADJUST_QOS":
            return self._validate_adjust_qos(parameters, constraints_checked)
        elif action_type == "RESTART_SERVICE":
            return self._validate_restart_service(state, constraints_checked)
        elif action_type == "OPTIMIZE_CONFIG":
            return self._validate_optimize_config(constraints_checked)
        elif action_type == "ENABLE_CACHING":
            return self._validate_enable_caching(state, constraints_checked)
        elif action_type == "DISABLE_CACHING":
            return self._validate_disable_caching(state, constraints_checked)
        
        return ValidationResult(
            valid=True,
            reason="Action is valid",
            constraints_checked=constraints_checked
        )
    
    def _validate_scale_up(self, state, capacity_config: Dict, parameters: Dict, 
                           constraints_checked: List[str]) -> ValidationResult:
        """Validate scale-up action."""
        constraints_checked.append("capacity_not_at_max")
        
        max_capacity = capacity_config.get("max_capacity", 20)
        increment = parameters.get("increment", capacity_config.get("scale_up_increment", 2))
        
        if state.capacity >= max_capacity:
            return ValidationResult(
                valid=False,
                reason=f"Already at maximum capacity ({max_capacity})",
                constraints_checked=constraints_checked
            )
        
        constraints_checked.append("increment_positive")
        if increment <= 0:
            return ValidationResult(
                valid=False,
                reason=f"Scale increment must be positive: {increment}",
                constraints_checked=constraints_checked
            )
        
        return ValidationResult(
            valid=True,
            reason="Scale-up action is valid",
            constraints_checked=constraints_checked
        )
    
    def _validate_scale_down(self, state, capacity_config: Dict, parameters: Dict,
                             constraints_checked: List[str]) -> ValidationResult:
        """Validate scale-down action."""
        constraints_checked.append("capacity_not_at_min")
        
        min_capacity = capacity_config.get("min_capacity", 1)
        decrement = parameters.get("decrement", capacity_config.get("scale_down_increment", 1))
        
        if state.capacity <= min_capacity:
            return ValidationResult(
                valid=False,
                reason=f"Already at minimum capacity ({min_capacity})",
                constraints_checked=constraints_checked
            )
        
        constraints_checked.append("decrement_positive")
        if decrement <= 0:
            return ValidationResult(
                valid=False,
                reason=f"Scale decrement must be positive: {decrement}",
                constraints_checked=constraints_checked
            )
        
        # Check if scaling down would violate load constraints
        constraints_checked.append("load_allows_scale_down")
        if state.load_level > 0.8 and state.capacity <= min_capacity + 1:
            return ValidationResult(
                valid=False,
                reason="Cannot scale down during high load with near-minimum capacity",
                constraints_checked=constraints_checked
            )
        
        return ValidationResult(
            valid=True,
            reason="Scale-down action is valid",
            constraints_checked=constraints_checked
        )

    def _validate_adjust_qos(self, parameters: Dict, 
                             constraints_checked: List[str]) -> ValidationResult:
        """Validate QoS adjustment action."""
        constraints_checked.append("qos_mode_valid")
        
        mode = parameters.get("mode", "balanced")
        valid_modes = ["balanced", "performance", "throughput"]
        
        if mode not in valid_modes:
            return ValidationResult(
                valid=False,
                reason=f"Invalid QoS mode: {mode}. Valid modes: {valid_modes}",
                constraints_checked=constraints_checked
            )
        
        return ValidationResult(
            valid=True,
            reason="QoS adjustment action is valid",
            constraints_checked=constraints_checked
        )
    
    def _validate_restart_service(self, state, 
                                  constraints_checked: List[str]) -> ValidationResult:
        """Validate restart service action."""
        constraints_checked.append("not_recently_restarted")
        
        # Check if service was recently restarted (within last 60 seconds)
        if state.last_action == "RESTART_SERVICE" and state.last_action_time:
            time_since_restart = (datetime.now() - state.last_action_time).total_seconds()
            if time_since_restart < 60:
                return ValidationResult(
                    valid=False,
                    reason=f"Service was restarted {time_since_restart:.0f}s ago. Wait at least 60s.",
                    constraints_checked=constraints_checked
                )
        
        return ValidationResult(
            valid=True,
            reason="Restart service action is valid",
            constraints_checked=constraints_checked
        )
    
    def _validate_optimize_config(self, constraints_checked: List[str]) -> ValidationResult:
        """Validate optimize config action."""
        constraints_checked.append("optimization_allowed")
        
        # Optimization is always allowed
        return ValidationResult(
            valid=True,
            reason="Optimize config action is valid",
            constraints_checked=constraints_checked
        )
    
    def _validate_enable_caching(self, state, 
                                 constraints_checked: List[str]) -> ValidationResult:
        """Validate enable caching action."""
        constraints_checked.append("caching_not_enabled")
        
        if state.caching_enabled:
            return ValidationResult(
                valid=False,
                reason="Caching is already enabled",
                constraints_checked=constraints_checked
            )
        
        # Check if there's enough memory for caching
        constraints_checked.append("memory_available_for_caching")
        baseline = self._state_manager.get_baseline_metrics()
        memory_threshold = baseline.get("memory_usage", 2048) * 1.5
        
        if state.memory_usage > memory_threshold:
            return ValidationResult(
                valid=False,
                reason=f"Memory usage too high for caching: {state.memory_usage:.0f}MB",
                constraints_checked=constraints_checked
            )
        
        return ValidationResult(
            valid=True,
            reason="Enable caching action is valid",
            constraints_checked=constraints_checked
        )
    
    def _validate_disable_caching(self, state, 
                                  constraints_checked: List[str]) -> ValidationResult:
        """Validate disable caching action."""
        constraints_checked.append("caching_enabled")
        
        if not state.caching_enabled:
            return ValidationResult(
                valid=False,
                reason="Caching is already disabled",
                constraints_checked=constraints_checked
            )
        
        return ValidationResult(
            valid=True,
            reason="Disable caching action is valid",
            constraints_checked=constraints_checked
        )

    def get_supported_actions(self) -> List[str]:
        """Return list of supported action types.
        
        Returns:
            List of action type strings.
        """
        return [action.value for action in ActionType]
    
    def get_action_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get action execution history.
        
        Args:
            limit: Maximum number of history entries to return.
            
        Returns:
            List of action results as dictionaries.
        """
        entries = self._action_history[-limit:] if limit > 0 else self._action_history
        return [entry.to_dict() for entry in entries]
    
    def _record_action(self, result: ActionResult) -> None:
        """Record action result in history.
        
        Args:
            result: ActionResult to record.
        """
        self._action_history.append(result)
        
        # Trim history if too large
        if len(self._action_history) > self._max_history_size:
            self._action_history = self._action_history[-self._max_history_size:]
    
    def get_action_stats(self) -> Dict[str, Any]:
        """Get statistics about action execution.
        
        Returns:
            Dictionary with action statistics.
        """
        if not self._action_history:
            return {
                "total_actions": 0,
                "successful_actions": 0,
                "failed_actions": 0,
                "success_rate": 0.0,
                "avg_execution_time_ms": 0.0,
                "actions_by_type": {}
            }
        
        total = len(self._action_history)
        successful = sum(1 for a in self._action_history if a.success)
        failed = total - successful
        
        avg_time = sum(a.execution_time_ms for a in self._action_history) / total
        
        actions_by_type = {}
        for action in self._action_history:
            if action.action_type not in actions_by_type:
                actions_by_type[action.action_type] = {"total": 0, "successful": 0}
            actions_by_type[action.action_type]["total"] += 1
            if action.success:
                actions_by_type[action.action_type]["successful"] += 1
        
        return {
            "total_actions": total,
            "successful_actions": successful,
            "failed_actions": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_execution_time_ms": avg_time,
            "actions_by_type": actions_by_type
        }
    
    def clear_history(self) -> None:
        """Clear action history."""
        self._action_history.clear()
    
    def can_execute_action(self, action_type: str, parameters: Optional[Dict[str, Any]] = None) -> bool:
        """Check if an action can be executed without actually executing it.
        
        Args:
            action_type: Type of action to check.
            parameters: Optional parameters for the action.
            
        Returns:
            True if action can be executed, False otherwise.
        """
        validation = self.validate_action(action_type, parameters)
        return validation.valid
