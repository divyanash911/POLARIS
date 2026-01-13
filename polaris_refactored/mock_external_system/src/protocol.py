"""
Protocol - Communication protocol for mock system.

This module provides the Protocol class that handles:
- Command parsing from text-based protocol
- Response formatting with status and data
- JSON serialization for complex data
- Protocol validation and error handling

Protocol Format:
- Request: <command> [<param1>] [<param2>] ...
- Response: <status>|<data>
  - status: OK or ERROR
  - data: JSON-encoded response data

"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class ResponseStatus(Enum):
    """Response status codes."""
    OK = "OK"
    ERROR = "ERROR"


@dataclass
class ParsedCommand:
    """Represents a parsed command from the protocol."""
    command: str
    args: List[str]
    params: Dict[str, Any]
    raw: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "command": self.command,
            "args": self.args,
            "params": self.params,
            "raw": self.raw
        }


@dataclass
class ProtocolResponse:
    """Represents a protocol response."""
    status: ResponseStatus
    data: Any
    message: Optional[str] = None
    
    def to_wire_format(self) -> str:
        """Convert to wire format string.
        
        Returns:
            String in format: STATUS|JSON_DATA
        """
        response_data = {
            "data": self.data,
        }
        if self.message:
            response_data["message"] = self.message
            
        return f"{self.status.value}|{json.dumps(response_data)}"
    
    @classmethod
    def ok(cls, data: Any, message: Optional[str] = None) -> "ProtocolResponse":
        """Create an OK response."""
        return cls(status=ResponseStatus.OK, data=data, message=message)
    
    @classmethod
    def error(cls, message: str, data: Any = None) -> "ProtocolResponse":
        """Create an ERROR response."""
        return cls(status=ResponseStatus.ERROR, data=data, message=message)


class ProtocolError(Exception):
    """Exception raised for protocol errors."""
    
    def __init__(self, message: str, command: Optional[str] = None):
        self.message = message
        self.command = command
        super().__init__(message)


class Protocol:
    """Communication protocol for mock system.
    
    This class handles:
    - Parsing incoming commands from text format
    - Formatting responses with status and JSON data
    - Validating command syntax
    - Serializing complex data types
    
    Protocol Format:
    - Request: COMMAND [arg1] [arg2] ... [key=value] [key=value]
    - Response: STATUS|{"data": ..., "message": ...}
    """
    
    # Supported commands
    SUPPORTED_COMMANDS = {
        "get_metrics": {
            "description": "Get current system metrics",
            "args": [],
            "params": []
        },
        "get_state": {
            "description": "Get current system state",
            "args": [],
            "params": []
        },
        "execute_action": {
            "description": "Execute an adaptation action",
            "args": ["action_type"],
            "params": ["increment", "decrement", "mode"]
        },
        "set_load": {
            "description": "Set simulated load level",
            "args": ["level"],
            "params": []
        },
        "health_check": {
            "description": "Check system health",
            "args": [],
            "params": []
        },
        "reset": {
            "description": "Reset to baseline state",
            "args": [],
            "params": []
        },
        "get_history": {
            "description": "Get state change history",
            "args": [],
            "params": ["limit"]
        },
        "get_supported_actions": {
            "description": "Get list of supported actions",
            "args": [],
            "params": []
        },
        "validate_action": {
            "description": "Validate if an action can be executed",
            "args": ["action_type"],
            "params": ["increment", "decrement", "mode"]
        },
        "get_action_history": {
            "description": "Get action execution history",
            "args": [],
            "params": ["limit"]
        },
        "shutdown": {
            "description": "Gracefully shutdown the server",
            "args": [],
            "params": []
        }
    }
    
    def __init__(self):
        """Initialize the protocol handler."""
        self._command_handlers: Dict[str, callable] = {}
    
    def parse_command(self, raw_input: str) -> ParsedCommand:
        """Parse a raw command string into a ParsedCommand.
        
        Args:
            raw_input: Raw command string from client.
            
        Returns:
            ParsedCommand with parsed components.
            
        Raises:
            ProtocolError: If command format is invalid.
        """
        raw_input = raw_input.strip()
        
        if not raw_input:
            raise ProtocolError("Empty command received")
        
        # Split into tokens
        tokens = self._tokenize(raw_input)
        
        if not tokens:
            raise ProtocolError("No command found in input")
        
        # First token is the command
        command = tokens[0].lower()
        
        # Validate command
        if command not in self.SUPPORTED_COMMANDS:
            raise ProtocolError(
                f"Unknown command: {command}. Supported: {list(self.SUPPORTED_COMMANDS.keys())}",
                command=command
            )
        
        # Parse remaining tokens into args and params
        args = []
        params = {}
        
        for token in tokens[1:]:
            if "=" in token:
                # Key=value parameter
                key, value = token.split("=", 1)
                params[key] = self._parse_value(value)
            else:
                # Positional argument
                args.append(token)
        
        return ParsedCommand(
            command=command,
            args=args,
            params=params,
            raw=raw_input
        )
    
    def _tokenize(self, input_str: str) -> List[str]:
        """Tokenize input string, handling quoted strings.
        
        Args:
            input_str: Input string to tokenize.
            
        Returns:
            List of tokens.
        """
        tokens = []
        current_token = ""
        in_quotes = False
        quote_char = None
        
        for char in input_str:
            if char in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
            elif char == " " and not in_quotes:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    def _parse_value(self, value: str) -> Any:
        """Parse a string value into appropriate type.
        
        Args:
            value: String value to parse.
            
        Returns:
            Parsed value (int, float, bool, or string).
        """
        # Try to parse as JSON first (handles complex types)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Try boolean
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def format_response(self, status: ResponseStatus, data: Any, 
                       message: Optional[str] = None) -> str:
        """Format a response for sending to client.
        
        Args:
            status: Response status (OK or ERROR).
            data: Response data to serialize.
            message: Optional message.
            
        Returns:
            Formatted response string.
        """
        response = ProtocolResponse(status=status, data=data, message=message)
        return response.to_wire_format()
    
    def format_ok_response(self, data: Any, message: Optional[str] = None) -> str:
        """Format an OK response.
        
        Args:
            data: Response data.
            message: Optional message.
            
        Returns:
            Formatted OK response string.
        """
        return self.format_response(ResponseStatus.OK, data, message)
    
    def format_error_response(self, message: str, data: Any = None) -> str:
        """Format an ERROR response.
        
        Args:
            message: Error message.
            data: Optional error data.
            
        Returns:
            Formatted ERROR response string.
        """
        return self.format_response(ResponseStatus.ERROR, data, message)
    
    def parse_response(self, raw_response: str) -> Tuple[ResponseStatus, Any, Optional[str]]:
        """Parse a raw response string.
        
        Args:
            raw_response: Raw response string from server.
            
        Returns:
            Tuple of (status, data, message).
            
        Raises:
            ProtocolError: If response format is invalid.
        """
        raw_response = raw_response.strip()
        
        if "|" not in raw_response:
            raise ProtocolError("Invalid response format: missing separator")
        
        status_str, json_data = raw_response.split("|", 1)
        
        # Parse status
        try:
            status = ResponseStatus(status_str)
        except ValueError:
            raise ProtocolError(f"Invalid response status: {status_str}")
        
        # Parse JSON data
        try:
            parsed = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise ProtocolError(f"Invalid JSON in response: {e}")
        
        data = parsed.get("data")
        message = parsed.get("message")
        
        return status, data, message
    
    def serialize_data(self, data: Any) -> str:
        """Serialize data to JSON string.
        
        Args:
            data: Data to serialize.
            
        Returns:
            JSON string.
        """
        return json.dumps(data, default=self._json_serializer)
    
    def deserialize_data(self, json_str: str) -> Any:
        """Deserialize JSON string to data.
        
        Args:
            json_str: JSON string to deserialize.
            
        Returns:
            Deserialized data.
            
        Raises:
            ProtocolError: If JSON is invalid.
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ProtocolError(f"Invalid JSON: {e}")
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for complex types.
        
        Args:
            obj: Object to serialize.
            
        Returns:
            Serializable representation.
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def get_command_help(self, command: Optional[str] = None) -> Dict[str, Any]:
        """Get help information for commands.
        
        Args:
            command: Optional specific command to get help for.
            
        Returns:
            Dictionary with command help information.
        """
        if command:
            if command not in self.SUPPORTED_COMMANDS:
                return {"error": f"Unknown command: {command}"}
            return {command: self.SUPPORTED_COMMANDS[command]}
        return self.SUPPORTED_COMMANDS
    
    def validate_command_args(self, parsed: ParsedCommand) -> Tuple[bool, Optional[str]]:
        """Validate command arguments.
        
        Args:
            parsed: Parsed command to validate.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        cmd_spec = self.SUPPORTED_COMMANDS.get(parsed.command)
        if not cmd_spec:
            return False, f"Unknown command: {parsed.command}"
        
        required_args = cmd_spec.get("args", [])
        
        if len(parsed.args) < len(required_args):
            missing = required_args[len(parsed.args):]
            return False, f"Missing required arguments: {missing}"
        
        return True, None
    
    def build_command(self, command: str, args: Optional[List[str]] = None,
                     params: Optional[Dict[str, Any]] = None) -> str:
        """Build a command string from components.
        
        Args:
            command: Command name.
            args: Optional positional arguments.
            params: Optional key=value parameters.
            
        Returns:
            Command string ready to send.
        """
        parts = [command]
        
        if args:
            parts.extend(str(arg) for arg in args)
        
        if params:
            for key, value in params.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                parts.append(f"{key}={value}")
        
        return " ".join(parts)
