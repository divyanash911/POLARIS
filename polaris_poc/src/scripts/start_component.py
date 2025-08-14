#!/usr/bin/env python3
"""
Entry point script for starting POLARIS components.

This script can launch monitor or execution adapters with a specified
managed system plugin.

Examples:
    # Start monitor with SWIM plugin
    python start_component.py monitor --plugin-dir extern

    # Start execution adapter with custom config
    python start_component.py execution --plugin-dir extern --config custom_config.yaml

    # Start with debug logging
    python start_component.py monitor --plugin-dir extern --log-level DEBUG
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from polaris.adapters.monitor import MonitorAdapter
from polaris.adapters.execution import ExecutionAdapter
from polaris.common.logging_setup import setup_logging


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start POLARIS adapter components"
    )
    
    parser.add_argument(
        "component",
        choices=["monitor", "execution"],
        help="Component to start"
    )
    
    parser.add_argument(
        "--plugin-dir",
        required=True,
        help="Directory containing the managed system plugin"
    )
    
    parser.add_argument(
        "--config",
        default="src/config/polaris_config.yaml",
        help="Path to POLARIS framework configuration"
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration and exit"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialize adapter but don't start processing"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.setLevel(getattr(logging, args.log_level))
    
    # Resolve paths
    plugin_dir = Path(args.plugin_dir).resolve()
    config_path = Path(args.config).resolve()
    
    if not plugin_dir.exists():
        logger.error(f"Plugin directory not found: {plugin_dir}")
        sys.exit(1)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Create adapter
    try:
        logger.info(f"Creating {args.component} adapter...")
        
        if args.component == "monitor":
            adapter = MonitorAdapter(
                polaris_config_path=str(config_path),
                plugin_dir=str(plugin_dir),
                logger=logger
            )
        else:  # execution
            adapter = ExecutionAdapter(
                polaris_config_path=str(config_path),
                plugin_dir=str(plugin_dir),
                logger=logger
            )
        
        logger.info(f"✅ {args.component.capitalize()} adapter created successfully")
        logger.info(f"   System: {adapter.plugin_config.get('system_name')}")
        logger.info(f"   Connector: {adapter.connector.__class__.__name__}")
        
        # Validation-only mode
        if args.validate_only:
            logger.info("✅ Configuration validation passed")
            logger.info("🏁 Validation complete - exiting")
            return
        
    except Exception as e:
        logger.error(f"❌ Failed to create adapter: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Setup signal handling
    stop_event = asyncio.Event()
    
    def signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()
    
    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda *_: signal_handler())
    
    # Dry run mode
    if args.dry_run:
        logger.info("🧪 Dry run mode - adapter initialized but not started")
        logger.info("🏁 Dry run complete - exiting")
        return
    
    # Start adapter
    logger.info(f"🚀 Starting {args.component} adapter...")
    
    try:
        async with adapter:
            logger.info(f"✅ {args.component.capitalize()} adapter started successfully")
            logger.info("📡 Adapter is running - press Ctrl+C to stop")
            await stop_event.wait()
    except Exception as e:
        logger.error(f"❌ Adapter error: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    logger.info(f"🛑 {args.component.capitalize()} adapter stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass