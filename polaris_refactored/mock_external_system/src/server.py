"""
Mock System Server - TCP server implementation.

This module provides the MockSystemServer class that handles:
- Asyncio TCP server for receiving commands
- Connection handling and command processing
- Support for all commands (get_metrics, execute_action, health_check, etc.)
- Concurrent connection handling
- Graceful shutdown

Requirements: 1.1, 1.2
"""

import asyncio
import logging
import signal
import sys
from asyncio import StreamReader, StreamWriter
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from .protocol import Protocol, ProtocolError, ResponseStatus, ParsedCommand
from .state_manager import StateManager
from .metrics_simulator import MetricsSimulator
from .action_handler import ActionHandler


# Configure logging
logger = logging.getLogger(__name__)


class ConnectionInfo:
    """Information about a client connection."""
    
    def __init__(self, reader: StreamReader, writer: StreamWriter):
        self.reader = reader
        self.writer = writer
        self.connected_at = datetime.now()
        self.commands_processed = 0
        self.last_command_at: Optional[datetime] = None
        
        # Get peer info
        peername = writer.get_extra_info('peername')
        self.remote_host = peername[0] if peername else "unknown"
        self.remote_port = peername[1] if peername else 0
    
    @property
    def connection_id(self) -> str:
        """Get unique connection identifier."""
        return f"{self.remote_host}:{self.remote_port}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "connection_id": self.connection_id,
            "remote_host": self.remote_host,
            "remote_port": self.remote_port,
            "connected_at": self.connected_at.isoformat(),
            "commands_processed": self.commands_processed,
            "last_command_at": self.last_command_at.isoformat() if self.last_command_at else None
        }


class MockSystemServer:
    """TCP server for mock external system.
    
    This class handles:
    - Starting and stopping the TCP server
    - Accepting and managing client connections
    - Processing commands and sending responses
    - Concurrent connection handling
    - Graceful shutdown with cleanup
    """
    
    def __init__(self, host: str = "localhost", port: int = 5000,
                 state_manager: Optional[StateManager] = None,
                 max_connections: int = 100):
        """Initialize the mock system server.
        
        Args:
            host: Host address to bind to.
            port: Port number to listen on.
            state_manager: Optional StateManager instance. Creates one if not provided.
            max_connections: Maximum number of concurrent connections.
        """
        self.host = host
        self.port = port
        self.max_connections = max_connections
        
        # Initialize components
        self._state_manager = state_manager or StateManager()
        self._metrics_simulator = MetricsSimulator(self._state_manager)
        self._action_handler = ActionHandler(self._state_manager, self._metrics_simulator)
        self._protocol = Protocol()
        
        # Server state
        self._server: Optional[asyncio.AbstractServer] = None
        self._running = False
        self._connections: Set[ConnectionInfo] = set()
        self._start_time: Optional[datetime] = None
        
        # Shutdown event
        self._shutdown_event = asyncio.Event()
    
    @property
    def running(self) -> bool:
        """Check if server is running."""
        return self._running
    
    @property
    def connection_count(self) -> int:
        """Get current number of connections."""
        return len(self._connections)
    
    async def start(self) -> None:
        """Start the TCP server.
        
        Raises:
            RuntimeError: If server is already running.
            OSError: If port is already in use.
        """
        if self._running:
            raise RuntimeError("Server is already running")
        
        logger.info(f"Starting mock system server on {self.host}:{self.port}")
        
        try:
            self._server = await asyncio.start_server(
                self._handle_client,
                self.host,
                self.port,
                reuse_address=True
            )
            
            self._running = True
            self._start_time = datetime.now()
            self._shutdown_event.clear()
            
            # Get actual bound address
            addrs = [sock.getsockname() for sock in self._server.sockets]
            logger.info(f"Mock system server listening on {addrs}")
            
        except OSError as e:
            logger.error(f"Failed to start server: {e}")
            raise
    
    async def serve_forever(self) -> None:
        """Serve requests until shutdown is requested."""
        if not self._running or not self._server:
            raise RuntimeError("Server not started. Call start() first.")
        
        async with self._server:
            await self._shutdown_event.wait()
    
    async def stop(self) -> None:
        """Stop the TCP server gracefully.
        
        Closes all connections and stops accepting new ones.
        """
        if not self._running:
            logger.warning("Server is not running")
            return
        
        logger.info("Stopping mock system server...")
        
        # Signal shutdown
        self._shutdown_event.set()
        self._running = False
        
        # Close all active connections
        for conn in list(self._connections):
            try:
                conn.writer.close()
                await conn.writer.wait_closed()
            except Exception as e:
                logger.warning(f"Error closing connection {conn.connection_id}: {e}")
        
        self._connections.clear()
        
        # Close server
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        
        logger.info("Mock system server stopped")
    
    async def _handle_client(self, reader: StreamReader, writer: StreamWriter) -> None:
        """Handle a client connection.
        
        Args:
            reader: Stream reader for receiving data.
            writer: Stream writer for sending data.
        """
        conn = ConnectionInfo(reader, writer)
        
        # Check connection limit
        if len(self._connections) >= self.max_connections:
            logger.warning(f"Connection limit reached, rejecting {conn.connection_id}")
            response = self._protocol.format_error_response(
                "Connection limit reached",
                {"max_connections": self.max_connections}
            )
            writer.write((response + "\n").encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return
        
        self._connections.add(conn)
        logger.info(f"Client connected: {conn.connection_id}")
        
        try:
            while self._running:
                try:
                    # Read command with timeout
                    data = await asyncio.wait_for(
                        reader.readline(),
                        timeout=300.0  # 5 minute timeout
                    )
                    
                    if not data:
                        # Client disconnected
                        break
                    
                    # Process command
                    command_str = data.decode().strip()
                    if not command_str:
                        continue
                    
                    logger.debug(f"Received from {conn.connection_id}: {command_str}")
                    
                    # Process and get response
                    response = await self._process_command(command_str)
                    
                    # Update connection stats
                    conn.commands_processed += 1
                    conn.last_command_at = datetime.now()
                    
                    # Send response
                    writer.write((response + "\n").encode())
                    await writer.drain()
                    
                    logger.debug(f"Sent to {conn.connection_id}: {response[:100]}...")
                    
                except asyncio.TimeoutError:
                    logger.info(f"Connection timeout: {conn.connection_id}")
                    break
                except ConnectionResetError:
                    logger.info(f"Connection reset: {conn.connection_id}")
                    break
                except Exception as e:
                    logger.error(f"Error processing command from {conn.connection_id}: {e}")
                    try:
                        response = self._protocol.format_error_response(str(e))
                        writer.write((response + "\n").encode())
                        await writer.drain()
                    except Exception:
                        break
        
        finally:
            # Cleanup connection
            self._connections.discard(conn)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.info(f"Client disconnected: {conn.connection_id}")
    
    async def _process_command(self, command_str: str) -> str:
        """Process a command and return response.
        
        Args:
            command_str: Raw command string.
            
        Returns:
            Formatted response string.
        """
        try:
            # Parse command
            parsed = self._protocol.parse_command(command_str)
            
            # Validate arguments
            is_valid, error_msg = self._protocol.validate_command_args(parsed)
            if not is_valid:
                return self._protocol.format_error_response(error_msg)
            
            # Route to handler
            handler = self._get_command_handler(parsed.command)
            if handler is None:
                return self._protocol.format_error_response(
                    f"No handler for command: {parsed.command}"
                )
            
            # Execute handler
            result = await handler(parsed)
            return result
            
        except ProtocolError as e:
            return self._protocol.format_error_response(e.message)
        except Exception as e:
            logger.exception(f"Error processing command: {command_str}")
            return self._protocol.format_error_response(f"Internal error: {str(e)}")
    
    def _get_command_handler(self, command: str) -> Optional[callable]:
        """Get handler function for a command.
        
        Args:
            command: Command name.
            
        Returns:
            Handler coroutine or None.
        """
        handlers = {
            "get_metrics": self._handle_get_metrics,
            "get_state": self._handle_get_state,
            "execute_action": self._handle_execute_action,
            "set_load": self._handle_set_load,
            "health_check": self._handle_health_check,
            "reset": self._handle_reset,
            "get_history": self._handle_get_history,
            "get_supported_actions": self._handle_get_supported_actions,
            "validate_action": self._handle_validate_action,
            "get_action_history": self._handle_get_action_history,
            "shutdown": self._handle_shutdown,
        }
        return handlers.get(command)
    
    async def _handle_get_metrics(self, parsed: ParsedCommand) -> str:
        """Handle get_metrics command."""
        metrics = self._metrics_simulator.generate_metrics()
        metrics_dict = {name: metric.to_dict() for name, metric in metrics.items()}
        return self._protocol.format_ok_response(metrics_dict, "Metrics retrieved successfully")
    
    async def _handle_get_state(self, parsed: ParsedCommand) -> str:
        """Handle get_state command."""
        state = self._state_manager.get_state()
        return self._protocol.format_ok_response(state, "State retrieved successfully")
    
    async def _handle_execute_action(self, parsed: ParsedCommand) -> str:
        """Handle execute_action command."""
        if not parsed.args:
            return self._protocol.format_error_response("Missing action_type argument")
        
        action_type = parsed.args[0].upper()
        parameters = parsed.params
        
        result = self._action_handler.execute_action(action_type, parameters)
        
        if result.success:
            return self._protocol.format_ok_response(
                result.to_dict(),
                f"Action {action_type} executed successfully"
            )
        else:
            return self._protocol.format_error_response(
                result.message,
                result.to_dict()
            )
    
    async def _handle_set_load(self, parsed: ParsedCommand) -> str:
        """Handle set_load command."""
        if not parsed.args:
            return self._protocol.format_error_response("Missing load level argument")
        
        try:
            load_level = float(parsed.args[0])
        except ValueError:
            return self._protocol.format_error_response(
                f"Invalid load level: {parsed.args[0]}. Must be a number between 0.0 and 1.0"
            )
        
        try:
            self._metrics_simulator.apply_load(load_level)
            return self._protocol.format_ok_response(
                {"load_level": load_level},
                f"Load level set to {load_level}"
            )
        except ValueError as e:
            return self._protocol.format_error_response(str(e))
    
    async def _handle_health_check(self, parsed: ParsedCommand) -> str:
        """Handle health_check command."""
        uptime = 0.0
        if self._start_time:
            uptime = (datetime.now() - self._start_time).total_seconds()
        
        health_data = {
            "status": "healthy",
            "uptime_seconds": uptime,
            "connections": self.connection_count,
            "max_connections": self.max_connections,
            "server_time": datetime.now().isoformat()
        }
        return self._protocol.format_ok_response(health_data, "System is healthy")
    
    async def _handle_reset(self, parsed: ParsedCommand) -> str:
        """Handle reset command."""
        self._state_manager.reset_to_baseline()
        self._metrics_simulator.reset_smoothing()
        self._action_handler.clear_history()
        
        return self._protocol.format_ok_response(
            {"reset": True},
            "System reset to baseline state"
        )
    
    async def _handle_get_history(self, parsed: ParsedCommand) -> str:
        """Handle get_history command."""
        limit = parsed.params.get("limit", 100)
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            limit = 100
        
        history = self._state_manager.get_history(limit)
        return self._protocol.format_ok_response(
            {"history": history, "count": len(history)},
            f"Retrieved {len(history)} history entries"
        )
    
    async def _handle_get_supported_actions(self, parsed: ParsedCommand) -> str:
        """Handle get_supported_actions command."""
        actions = self._action_handler.get_supported_actions()
        return self._protocol.format_ok_response(
            {"actions": actions},
            f"Retrieved {len(actions)} supported actions"
        )
    
    async def _handle_validate_action(self, parsed: ParsedCommand) -> str:
        """Handle validate_action command."""
        if not parsed.args:
            return self._protocol.format_error_response("Missing action_type argument")
        
        action_type = parsed.args[0].upper()
        parameters = parsed.params
        
        result = self._action_handler.validate_action(action_type, parameters)
        
        return self._protocol.format_ok_response(
            result.to_dict(),
            "Validation complete"
        )
    
    async def _handle_get_action_history(self, parsed: ParsedCommand) -> str:
        """Handle get_action_history command."""
        limit = parsed.params.get("limit", 100)
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            limit = 100
        
        history = self._action_handler.get_action_history(limit)
        stats = self._action_handler.get_action_stats()
        
        return self._protocol.format_ok_response(
            {"history": history, "stats": stats, "count": len(history)},
            f"Retrieved {len(history)} action history entries"
        )
    
    async def _handle_shutdown(self, parsed: ParsedCommand) -> str:
        """Handle shutdown command.
        
        This command initiates graceful shutdown of the server.
        """
        logger.info("Shutdown command received")
        
        # Send response before shutting down
        response = self._protocol.format_ok_response(
            {"status": "shutting_down"},
            "Server shutting down"
        )
        
        # Schedule shutdown after response is sent
        asyncio.create_task(self.stop())
        
        return response
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information.
        
        Returns:
            Dictionary with server information.
        """
        uptime = 0.0
        if self._start_time:
            uptime = (datetime.now() - self._start_time).total_seconds()
        
        return {
            "host": self.host,
            "port": self.port,
            "running": self._running,
            "uptime_seconds": uptime,
            "connections": self.connection_count,
            "max_connections": self.max_connections,
            "start_time": self._start_time.isoformat() if self._start_time else None
        }
    
    def get_connections_info(self) -> List[Dict[str, Any]]:
        """Get information about active connections.
        
        Returns:
            List of connection information dictionaries.
        """
        return [conn.to_dict() for conn in self._connections]


async def run_server(host: str = "localhost", port: int = 5000,
                    config_path: Optional[str] = None) -> None:
    """Run the mock system server.
    
    Args:
        host: Host address to bind to.
        port: Port number to listen on.
        config_path: Optional path to configuration file.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create state manager with config
    state_manager = StateManager(config_path=config_path)
    server_config = state_manager.config.get("server", {})
    
    # Use config values if not overridden
    host = host or server_config.get("host", "localhost")
    port = port or server_config.get("port", 5000)
    max_connections = server_config.get("max_connections", 100)
    
    # Create and start server
    server = MockSystemServer(
        host=host,
        port=port,
        state_manager=state_manager,
        max_connections=max_connections
    )
    
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(server.stop())
    
    # Register signal handlers (Unix only)
    if sys.platform != "win32":
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)
    
    try:
        await server.start()
        await server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await server.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mock External System Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    asyncio.run(run_server(args.host, args.port, args.config))
