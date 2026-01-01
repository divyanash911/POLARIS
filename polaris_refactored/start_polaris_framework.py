#!/usr/bin/env python3
"""
POLARIS Framework - CLI Entry Point

This script acts as the main entry point for the POLARIS framework CLI.
It delegates actual logic to the `framework.cli` package.
"""

import sys
import argparse
import asyncio
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from the script directory
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
except ImportError:
    pass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from framework.cli.manager import PolarisFrameworkManager
from framework.cli.shell import InteractiveShell
from framework.cli.utils import print_banner
from framework.cli.dashboard import ObservabilityDashboard

VERSION = "2.0.0"

DEFAULT_CONFIGS = {
    "swim": "config/swim_system_config.yaml",
    "mock": "config/mock_system_config.yaml",
    "llm": "config/llm_integration_config.yaml",
}

def resolve_config_path(config_arg: str) -> str:
    """Resolve configuration path from shortcut or file path."""
    if config_arg in DEFAULT_CONFIGS:
        path = Path(__file__).parent / DEFAULT_CONFIGS[config_arg]
    else:
        path = Path(config_arg)
    
    return str(path)

async def main():
    parser = argparse.ArgumentParser(description="POLARIS Framework CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the framework")
    start_parser.add_argument("--config", "-c", default="mock", help="Configuration file path or alias (mock, swim, llm)")
    start_parser.add_argument("--log-level", "-l", default="INFO", help="Logging level")
    start_parser.add_argument("--shell", "-s", action="store_true", help="Start interactive shell after initialization")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check system status")
    
    # Shell command
    shell_parser = subparsers.add_parser("shell", help="Start interactive shell (connects to running instance or standalone)")
    
    # Systems command
    systems_parser = subparsers.add_parser("systems", help="Managed systems operations")
    systems_sub = systems_parser.add_subparsers(dest="subcommand", help="Systems subcommand")
    systems_sub.add_parser("list", help="List managed systems")
    
    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Start real-time dashboard")
    dash_parser.add_argument("--config", "-c", default="mock", help="Configuration file path to start with if not running")
    
    args = parser.parse_args()
    
    print_banner(VERSION)
    
    if not args.command:
        parser.print_help()
        return

    # Initialize Manager
    manager = PolarisFrameworkManager()
    
    if args.command == "start":
        config_path = resolve_config_path(args.config)
        print(f"Starting with config: {config_path}")
        
        if await manager.initialize(config_path, args.log_level):
            if await manager.start():
                if args.shell:
                    shell = InteractiveShell(manager)
                    await shell.run()
                    # After shell exits, we might want to keep running or stop? 
                    # Usually explicit quit in shell stops it.
                    # If we just exited shell loop, we might want to wait for shutdown if still running.
                    if manager.state == "running":
                        await manager.run_until_shutdown()
                else:
                    await manager.run_until_shutdown()
    
    elif args.command == "shell":
        # For now, shell starts its own instance mostly for demo purposes 
        # because we don't have a daemon/client architecture yet where shell connects to background process.
        # So we initialize a default or prompt for config?
        # Let's assume standalone shell for now requires starting the framework implicitly or just mocking?
        # Re-using start logic for simplicity of this refactor step.
        print("Note: Shell currently runs in standalone mode (starts its own framework instance).")
        config_path = resolve_config_path("mock") 
        if await manager.initialize(config_path):
             await manager.start()
             shell = InteractiveShell(manager)
             await shell.run()
             await manager.stop()

    elif args.command == "dashboard":
        # Similar to shell, starts standalone instance for now
        config_path = resolve_config_path(args.config)
        if await manager.initialize(config_path):
             await manager.start()
             dash = ObservabilityDashboard(manager)
             await dash.display_live_metrics()
             await manager.stop()
             
    elif args.command == "systems":
        # This would ideally connect to running instance.
        print("Command not fully supported in standalone mode without active framework.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
