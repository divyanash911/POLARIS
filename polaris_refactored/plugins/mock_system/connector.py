"""
Mock System TCP Connector for POLARIS Framework.

This module implements the managed system connector for the mock external system,
handling TCP communication with retry logic and error handling.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone

from src.domain.interfaces import ManagedSystemConnector
from src.domain.models import (
    SystemState, AdaptationAction, ExecutionResult, MetricValue, 
    HealthStatus, ExecutionStatus
)


class MockSystemConnector(ManagedSystemConnector):
    """
    TCP connector for Mock External System.
    
    This connector implements the communication protocol for the mock system's
    external control interface via TCP socket connections.
    
    Protocol Format:
    - Request: COMMAND [arg1] [arg2] ... [key=value]
    - Response: STATUS|{"data": ..., "message": ...}
    """
    
    def __init__(self, system_config: Optional[Dict[str, Any]] = None):
        """Initialize the Mock System TCP connector.
        
        Args:
            system_config: Complete configuration for mock system (optional for basic usage)
        """
        self.config = system_config or {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Extract connection parameters
        connection_config = self.config.get("connection", {})
        self.host = connection_config.get("host", "localhost")
        self.port = connection_config.get("port", 5000)
        
        # Extract implementation parameters
        implementation_config = self.config.get("implementation", {})
        self.timeout = implementation_config.get("timeout", 10.0)
        self.max_retries = implementation_config.get("max_retries", 3)
        self.retry_base_delay = implementation_config.get("retry_base_delay", 1.0)
        self.retry_max_delay = implementation_config.get("retry_max_delay", 5.0)
        
        # Connection state
        self._connected = False
        self._system_id = self.config.get("system_name", "mock_system")
        
        self.logger.info(
            "Mock system connector initialized",
            extra={
                "host": self.host,
                "port": self.port,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "system_id": self._system_id
            }
        )
    
    async def connect(self) -> bool:
        """Establish connection to mock system.
        
        Tests connection with a health check command.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Test connection with health check
            response = await self._send_command("health_check")
            status, data, message = self._parse_response(response)
            
            if status == "OK":
                self._connected = True
                self.logger.info(
                    "Mock system connection verified",
                    extra={
                        "host": self.host,
                        "port": self.port,
                        "health_data": data
                    }
                )
                return True
            else:
                self._connected = False
                self.logger.error(
                    "Mock system health check failed",
                    extra={
                        "host": self.host,
                        "port": self.port,
                        "error": message
                    }
                )
                return False
                
        except Exception as e:
            self._connected = False
            self.logger.error(
                "Failed to connect to mock system",
                extra={
                    "host": self.host,
                    "port": self.port,
                    "error": str(e)
                }
            )
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from mock system.
        
        Since we don't maintain persistent connections, this just updates state.
        
        Returns:
            bool: True if disconnection successful
        """
        self._connected = False
        self.logger.info("Mock system connector disconnected")
        return True
    
    async def get_system_id(self) -> str:
        """Get the unique identifier for this managed system.
        
        Returns:
            str: Unique system identifier
        """
        return self._system_id

    async def collect_metrics(self) -> Dict[str, MetricValue]:
        """Collect current metrics from the mock system.
        
        Returns:
            Dict[str, MetricValue]: Dictionary of metric name to metric value
        """
        try:
            response = await self._send_command("get_metrics")
            status, data, message = self._parse_response(response)
            
            if status != "OK":
                self.logger.error(f"Failed to collect metrics: {message}")
                return {}
            
            metrics = {}
            timestamp = datetime.now(timezone.utc)
            
            # Convert response data to MetricValue objects
            for metric_name, metric_data in data.items():
                if isinstance(metric_data, dict):
                    metrics[metric_name] = MetricValue(
                        name=metric_data.get("name", metric_name),
                        value=metric_data.get("value", 0),
                        unit=metric_data.get("unit"),
                        timestamp=timestamp,
                        tags=metric_data.get("tags", {})
                    )
                else:
                    # Handle simple value format
                    metrics[metric_name] = MetricValue(
                        name=metric_name,
                        value=metric_data,
                        timestamp=timestamp
                    )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            return {}
    
    async def get_system_state(self) -> SystemState:
        """Get the current state of the mock system.
        
        Returns:
            SystemState: Current system state
        """
        try:
            response = await self._send_command("get_state")
            status, data, message = self._parse_response(response)
            
            # Collect metrics for the state
            metrics = await self.collect_metrics()
            
            # Determine health status
            health_status = HealthStatus.HEALTHY
            if not self._connected:
                health_status = HealthStatus.UNHEALTHY
            elif status != "OK":
                health_status = HealthStatus.WARNING
            elif not metrics:
                health_status = HealthStatus.WARNING
            
            return SystemState(
                system_id=self._system_id,
                timestamp=datetime.now(timezone.utc),
                metrics=metrics,
                health_status=health_status,
                metadata={
                    "host": self.host,
                    "port": self.port,
                    "connected": self._connected,
                    "state_data": data if status == "OK" else None
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get system state: {e}")
            return SystemState(
                system_id=self._system_id,
                timestamp=datetime.now(timezone.utc),
                metrics={},
                health_status=HealthStatus.UNHEALTHY,
                metadata={"error": str(e)}
            )
    
    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        """Execute an adaptation action on the mock system.
        
        Args:
            action: The adaptation action to execute
            
        Returns:
            ExecutionResult: Result of the action execution
        """
        start_time = time.perf_counter()
        
        try:
            action_type = action.action_type.upper()
            
            # Build command with parameters
            cmd_parts = ["execute_action", action_type]
            
            # Add parameters as key=value pairs
            for key, value in action.parameters.items():
                if isinstance(value, (dict, list)):
                    cmd_parts.append(f"{key}={json.dumps(value)}")
                else:
                    cmd_parts.append(f"{key}={value}")
            
            command = " ".join(cmd_parts)
            response = await self._send_command(command)
            status, data, message = self._parse_response(response)
            
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            
            if status == "OK":
                return ExecutionResult(
                    action_id=action.action_id,
                    status=ExecutionStatus.SUCCESS,
                    result_data={
                        "mock_response": data,
                        "action_type": action_type,
                        "parameters": action.parameters,
                        "message": message
                    },
                    execution_time_ms=execution_time_ms
                )
            else:
                return ExecutionResult(
                    action_id=action.action_id,
                    status=ExecutionStatus.FAILED,
                    result_data={
                        "error_data": data,
                        "action_type": action_type,
                        "parameters": action.parameters
                    },
                    error_message=message,
                    execution_time_ms=execution_time_ms
                )
                
        except asyncio.TimeoutError:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.TIMEOUT,
                result_data={"error": "Command timed out"},
                error_message=f"Action execution timed out after {self.timeout}s",
                execution_time_ms=execution_time_ms
            )
        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            self.logger.error(f"Failed to execute action {action.action_id}: {e}")
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={"error": str(e)},
                error_message=str(e),
                execution_time_ms=execution_time_ms
            )
    
    async def validate_action(self, action: AdaptationAction) -> bool:
        """Validate if an adaptation action can be executed.
        
        Args:
            action: The adaptation action to validate
            
        Returns:
            bool: True if action is valid, False otherwise
        """
        try:
            action_type = action.action_type.upper()
            
            # Build validation command
            cmd_parts = ["validate_action", action_type]
            
            # Add parameters as key=value pairs
            for key, value in action.parameters.items():
                if isinstance(value, (dict, list)):
                    cmd_parts.append(f"{key}={json.dumps(value)}")
                else:
                    cmd_parts.append(f"{key}={value}")
            
            command = " ".join(cmd_parts)
            response = await self._send_command(command)
            status, data, message = self._parse_response(response)
            
            if status == "OK" and isinstance(data, dict):
                return data.get("valid", False)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to validate action {action.action_id}: {e}")
            return False
    
    async def get_supported_actions(self) -> List[str]:
        """Get the list of action types supported by the mock system.
        
        Returns:
            List[str]: List of supported action type names
        """
        try:
            response = await self._send_command("get_supported_actions")
            status, data, message = self._parse_response(response)
            
            if status == "OK" and isinstance(data, dict):
                return data.get("actions", [])
            
            # Return default supported actions if command fails
            return [
                "SCALE_UP",
                "SCALE_DOWN",
                "ADJUST_QOS",
                "RESTART_SERVICE",
                "OPTIMIZE_CONFIG",
                "ENABLE_CACHING",
                "DISABLE_CACHING"
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get supported actions: {e}")
            return []

    # TCP Communication Helper Methods
    
    async def _send_command(self, command: str) -> str:
        """Send a command to the mock system via TCP with retry logic.
        
        Args:
            command: Command string to send
            
        Returns:
            str: Response from mock system
            
        Raises:
            ConnectionError: If connection fails after retries
            TimeoutError: If command times out
        """
        return await self._send_with_retries(command)
    
    async def _send_recv(self, command: str) -> str:
        """Send command and receive response via TCP.
        
        Args:
            command: Command string to send
            
        Returns:
            str: Response from mock system
            
        Raises:
            ConnectionError: If connection fails
            TimeoutError: If operation times out
        """
        start_time = time.perf_counter()
        
        try:
            # Open TCP connection with timeout
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Connection to {self.host}:{self.port} timed out")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.host}:{self.port}: {e}")
        
        try:
            # Send command
            writer.write((command + "\n").encode())
            await asyncio.wait_for(writer.drain(), timeout=self.timeout)
            
            # Receive response
            line = await asyncio.wait_for(reader.readline(), timeout=self.timeout)
            response = line.decode(errors="replace").strip()
            
            elapsed = time.perf_counter() - start_time
            self.logger.debug(
                "Mock system command executed",
                extra={
                    "command": command,
                    "response": response[:200] if len(response) > 200 else response,
                    "elapsed_ms": round(elapsed * 1000, 3)
                }
            )
            
            return response
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Command '{command}' timed out after {self.timeout}s")
        except Exception as e:
            raise RuntimeError(f"Command '{command}' failed: {e}")
        finally:
            # Always close the connection
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
    
    async def _send_with_retries(self, command: str) -> str:
        """Send command with exponential backoff retry.
        
        Args:
            command: Command string to send
            
        Returns:
            str: Response from mock system
            
        Raises:
            ConnectionError: If all retries fail
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(
                    "Sending mock system command",
                    extra={
                        "command": command,
                        "attempt": attempt,
                        "max_retries": self.max_retries
                    }
                )
                
                response = await self._send_recv(command)
                
                self.logger.debug(
                    "Mock system command successful",
                    extra={
                        "command": command,
                        "attempt": attempt
                    }
                )
                
                return response
                
            except (TimeoutError, ConnectionError, OSError, asyncio.TimeoutError) as e:
                last_error = e
                
                if attempt >= self.max_retries:
                    self.logger.error(
                        "Mock system command failed after max retries",
                        extra={
                            "command": command,
                            "attempts": attempt + 1,
                            "error": str(e)
                        }
                    )
                    raise
                
                # Calculate retry delay with jitter
                delay = min(
                    self.retry_base_delay * (2 ** attempt) + (time.time() % 1),
                    self.retry_max_delay
                )
                
                self.logger.warning(
                    "Mock system command failed, retrying",
                    extra={
                        "command": command,
                        "attempt": attempt,
                        "error": str(e),
                        "retry_in_sec": round(delay, 3)
                    }
                )
                
                await asyncio.sleep(delay)
        
        # This should not be reached, but just in case
        raise last_error or ConnectionError(
            f"Command '{command}' failed after {self.max_retries + 1} attempts"
        )
    
    def _parse_response(self, raw_response: str) -> tuple:
        """Parse a raw response string from the mock system.
        
        Args:
            raw_response: Raw response string in format STATUS|JSON_DATA
            
        Returns:
            Tuple of (status, data, message)
            
        Raises:
            ValueError: If response format is invalid
        """
        raw_response = raw_response.strip()
        
        if "|" not in raw_response:
            raise ValueError(f"Invalid response format: missing separator in '{raw_response}'")
        
        status_str, json_data = raw_response.split("|", 1)
        
        # Validate status
        if status_str not in ("OK", "ERROR"):
            raise ValueError(f"Invalid response status: {status_str}")
        
        # Parse JSON data
        try:
            parsed = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")
        
        data = parsed.get("data")
        message = parsed.get("message")
        
        return status_str, data, message
    
    # Additional helper methods for convenience
    
    async def set_load(self, load_level: float) -> bool:
        """Set the simulated load level on the mock system.
        
        Args:
            load_level: Load level between 0.0 and 1.0
            
        Returns:
            bool: True if successful
        """
        if not 0.0 <= load_level <= 1.0:
            raise ValueError(f"Load level must be between 0.0 and 1.0, got {load_level}")
        
        try:
            response = await self._send_command(f"set_load {load_level}")
            status, data, message = self._parse_response(response)
            return status == "OK"
        except Exception as e:
            self.logger.error(f"Failed to set load: {e}")
            return False
    
    async def reset(self) -> bool:
        """Reset the mock system to baseline state.
        
        Returns:
            bool: True if successful
        """
        try:
            response = await self._send_command("reset")
            status, data, message = self._parse_response(response)
            return status == "OK"
        except Exception as e:
            self.logger.error(f"Failed to reset: {e}")
            return False
    
    async def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get state change history from the mock system.
        
        Args:
            limit: Maximum number of history entries to retrieve
            
        Returns:
            List of history entries
        """
        try:
            response = await self._send_command(f"get_history limit={limit}")
            status, data, message = self._parse_response(response)
            
            if status == "OK" and isinstance(data, dict):
                return data.get("history", [])
            return []
        except Exception as e:
            self.logger.error(f"Failed to get history: {e}")
            return []
