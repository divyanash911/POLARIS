#!/usr/bin/env python3
"""
Stop script for mock external system.

This script gracefully shuts down a running mock external system server.
It connects to the server and sends a shutdown command, or can kill the process.

Usage:
    python stop_mock_system.py [--host HOST] [--port PORT] [--timeout TIMEOUT] [--force]
"""

import argparse
import asyncio
import logging
import sys
import socket
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        
    Returns:
        Configured logger instance.
    """
    log_level_value = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level_value,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    return logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Stop the mock external system server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stop server with default host and port
  python stop_mock_system.py
  
  # Stop server on custom host and port
  python stop_mock_system.py --host 0.0.0.0 --port 8000
  
  # Force kill if graceful shutdown fails
  python stop_mock_system.py --force
  
  # Stop with debug logging
  python stop_mock_system.py --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Server host (default: localhost)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Server port (default: 5000)'
    )
    
    parser.add_argument(
        '--timeout',
        type=float,
        default=5.0,
        help='Timeout for graceful shutdown in seconds (default: 5.0)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force kill if graceful shutdown fails'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


async def send_shutdown_command(host: str, port: int, timeout: float) -> bool:
    """Send shutdown command to the server.
    
    Args:
        host: Server host.
        port: Server port.
        timeout: Timeout in seconds.
        
    Returns:
        True if shutdown command was sent successfully, False otherwise.
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Connecting to server at {host}:{port}")
        
        # Create connection with timeout
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout
        )
        
        logger.info("Connected to server")
        
        # Send shutdown command
        shutdown_cmd = "shutdown\n"
        logger.info("Sending shutdown command")
        writer.write(shutdown_cmd.encode())
        await writer.drain()
        
        # Wait for response
        try:
            response = await asyncio.wait_for(
                reader.read(1024),
                timeout=timeout
            )
            logger.info(f"Server response: {response.decode().strip()}")
        except asyncio.TimeoutError:
            logger.warning("No response from server (timeout)")
        
        # Close connection
        writer.close()
        await writer.wait_closed()
        
        logger.info("Shutdown command sent successfully")
        return True
        
    except asyncio.TimeoutError:
        logger.error(f"Connection timeout (timeout={timeout}s)")
        return False
    except ConnectionRefusedError:
        logger.error(f"Connection refused - server may not be running at {host}:{port}")
        return False
    except Exception as e:
        logger.error(f"Error sending shutdown command: {e}")
        return False


async def main():
    """Main entry point for the shutdown script."""
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    logger.info("Mock system shutdown script")
    
    try:
        # Try graceful shutdown
        logger.info(f"Attempting graceful shutdown of {args.host}:{args.port}")
        success = await send_shutdown_command(args.host, args.port, args.timeout)
        
        if success:
            logger.info("Graceful shutdown completed")
            sys.exit(0)
        
        # If graceful shutdown failed and force flag is set
        if args.force:
            logger.warning("Graceful shutdown failed, attempting force kill")
            logger.info("Note: Force kill requires manual process termination")
            logger.info("Please manually kill the process or use system tools")
            sys.exit(1)
        else:
            logger.error("Graceful shutdown failed")
            logger.info("Use --force flag to attempt force kill")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
