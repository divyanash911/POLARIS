"""
Mock External System source modules.

This package contains the core implementation of the mock external system:
- server: TCP server implementation
- metrics_simulator: Metrics generation and simulation
- action_handler: Adaptation action processing
- state_manager: System state management
- protocol: Communication protocol
"""

from .state_manager import StateManager, MockSystemState, StateHistoryEntry
from .metrics_simulator import MetricsSimulator, MetricValue
from .action_handler import ActionHandler, ActionResult, ValidationResult, ActionType
from .protocol import Protocol, ProtocolError, ProtocolResponse, ParsedCommand, ResponseStatus
from .server import MockSystemServer, ConnectionInfo, run_server
from .config_validator import ConfigValidator, ConfigValidationError, validate_and_report

__all__ = [
    # State Manager
    "StateManager",
    "MockSystemState",
    "StateHistoryEntry",
    # Metrics Simulator
    "MetricsSimulator",
    "MetricValue",
    # Action Handler
    "ActionHandler",
    "ActionResult",
    "ValidationResult",
    "ActionType",
    # Protocol
    "Protocol",
    "ProtocolError",
    "ProtocolResponse",
    "ParsedCommand",
    "ResponseStatus",
    # Server
    "MockSystemServer",
    "ConnectionInfo",
    "run_server",
    # Config Validator
    "ConfigValidator",
    "ConfigValidationError",
    "validate_and_report",
]
