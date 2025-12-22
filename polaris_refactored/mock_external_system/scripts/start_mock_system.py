#!/usr/bin/env python3
"""
Start script for mock external system.

This script starts the mock external system server with configurable options.
It handles argument parsing, configuration loading, logging setup, and server initialization.

Usage:
    python start_mock_system.py [--config CONFIG_FILE] [--host HOST] [--port PORT] [--log-level LOG_LEVEL]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import MockSystemServer, StateManager, validate_and_report
import yaml


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
            logging.FileHandler('mock_system.log')
        ]
    )
    
    return logging.getLogger(__name__)


def load_configuration(config_file: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_file: Path to configuration file.
        
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If configuration file not found.
        yaml.YAMLError: If configuration file is invalid YAML.
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Configuration file is empty: {config_file}")
    
    return config


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Start the mock external system server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default configuration
  python start_mock_system.py
  
  # Start with custom configuration
  python start_mock_system.py --config config/scenarios/high_load.yaml
  
  # Start with custom host and port
  python start_mock_system.py --host 0.0.0.0 --port 8000
  
  # Start with debug logging
  python start_mock_system.py --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='Path to configuration file (default: config/default_config.yaml)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='Server host (overrides config file)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='Server port (overrides config file)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Validate configuration and exit'
    )
    
    return parser.parse_args()


async def main():
    """Main entry point for the mock system server."""
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    logger.info("Starting mock external system")
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_configuration(args.config)
        
        # Override host and port if provided
        if args.host:
            config['server']['host'] = args.host
            logger.info(f"Overriding host to {args.host}")
        
        if args.port:
            config['server']['port'] = args.port
            logger.info(f"Overriding port to {args.port}")
        
        # Validate configuration
        logger.info("Validating configuration")
        config_dir = Path(args.config).parent
        all_valid = validate_and_report(str(config_dir))
        
        if not all_valid:
            logger.error("Configuration validation failed")
            sys.exit(1)
        
        if args.validate_only:
            logger.info("Configuration validation successful (--validate-only)")
            sys.exit(0)
        
        # Initialize state manager
        logger.info("Initializing state manager")
        state_manager = StateManager(config)
        
        # Create and start server
        server_config = config.get('server', {})
        host = server_config.get('host', 'localhost')
        port = server_config.get('port', 5000)
        
        logger.info(f"Creating server on {host}:{port}")
        server = MockSystemServer(host, port, state_manager)
        
        logger.info("Starting server")
        await server.start()
        
        logger.info(f"Mock system server running on {host}:{port}")
        logger.info("Press Ctrl+C to stop")
        
        # Keep server running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Shutting down mock system server")
        try:
            await server.stop()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


if __name__ == '__main__':
    asyncio.run(main())
