#!/usr/bin/env python3
"""
POLARIS Framework - CLI Entry Point

This script acts as the main entry point for the POLARIS framework CLI.
It delegates actual logic to the `framework.cli` package.
"""

import sys
import argparse
import asyncio
import signal
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
from framework.cli.utils import print_banner, print_error, print_success, print_info
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
    
    # Validate config file exists
    if not path.exists():
        print_error(f"Configuration file not found: {path}")
        print_info(f"Available shortcuts: {', '.join(DEFAULT_CONFIGS.keys())}")
        sys.exit(1)
    
    return str(path)


def setup_signal_handlers(manager: PolarisFrameworkManager):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        print_info("\nShutdown signal received, stopping framework...")
        # The manager's run_until_shutdown handles the actual shutdown
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def cmd_start(args, manager: PolarisFrameworkManager) -> int:
    """Handle the 'start' command."""
    config_path = resolve_config_path(args.config)
    print_info(f"Starting with config: {config_path}")
    
    if not await manager.initialize(config_path, args.log_level):
        print_error("Failed to initialize framework")
        return 1
    
    if not await manager.start():
        print_error("Failed to start framework")
        return 1
    
    print_success("Framework started successfully")
    
    if args.shell:
        shell = InteractiveShell(manager)
        await shell.run()
        # After shell exits, check if framework should keep running
        if manager.state.value == "running":
            await manager.run_until_shutdown()
    else:
        await manager.run_until_shutdown()
    
    return 0


async def cmd_shell(args, manager: PolarisFrameworkManager) -> int:
    """Handle the 'shell' command."""
    print_info("Shell currently runs in standalone mode (starts its own framework instance).")
    config_path = resolve_config_path("mock")
    
    if not await manager.initialize(config_path):
        print_error("Failed to initialize framework for shell")
        return 1
    
    if not await manager.start():
        print_error("Failed to start framework for shell")
        return 1
    
    shell = InteractiveShell(manager)
    await shell.run()
    await manager.stop()
    return 0


async def cmd_dashboard(args, manager: PolarisFrameworkManager) -> int:
    """Handle the 'dashboard' command."""
    config_path = resolve_config_path(args.config)
    
    if not await manager.initialize(config_path):
        print_error("Failed to initialize framework for dashboard")
        return 1
    
    if not await manager.start():
        print_error("Failed to start framework for dashboard")
        return 1
    
    dash = ObservabilityDashboard(manager)
    await dash.display_live_metrics()
    await manager.stop()
    return 0


async def cmd_status(args, manager: PolarisFrameworkManager) -> int:
    """Handle the 'status' command."""
    status = manager.get_status()
    print(f"\nFramework Status: {status.state.value.upper()}")
    print(f"Uptime: {status.uptime_seconds:.1f}s")
    print(f"Components: {len(status.components)}")
    print(f"Managed Systems: {len(status.managed_systems)}")
    print(f"Meta Learner: {'Enabled' if status.meta_learner_enabled else 'Disabled'}")
    return 0


async def cmd_systems(args, manager: PolarisFrameworkManager) -> int:
    """Handle the 'systems' command."""
    print_info("Command requires active framework. Use 'start --shell' for interactive mode.")
    return 0


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="POLARIS Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start                    Start with default (mock) config
  %(prog)s start -c swim            Start with SWIM system config
  %(prog)s start -c /path/to/config.yaml  Start with custom config
  %(prog)s start --shell            Start and enter interactive shell
  %(prog)s dashboard                Start real-time metrics dashboard
  %(prog)s shell                    Start standalone interactive shell
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the framework")
    start_parser.add_argument(
        "--config", "-c", default="mock",
        help="Configuration file path or alias (mock, swim, llm)"
    )
    start_parser.add_argument(
        "--log-level", "-l", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    start_parser.add_argument(
        "--shell", "-s", action="store_true",
        help="Start interactive shell after initialization"
    )
    
    # Status command
    subparsers.add_parser("status", help="Check system status")
    
    # Shell command
    subparsers.add_parser("shell", help="Start interactive shell (standalone mode)")
    
    # Systems command
    systems_parser = subparsers.add_parser("systems", help="Managed systems operations")
    systems_sub = systems_parser.add_subparsers(dest="subcommand", help="Systems subcommand")
    systems_sub.add_parser("list", help="List managed systems")
    
    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Start real-time dashboard")
    dash_parser.add_argument(
        "--config", "-c", default="mock",
        help="Configuration file path or alias"
    )
    
    # Version command
    subparsers.add_parser("version", help="Show version information")
    
    args = parser.parse_args()
    
    print_banner(VERSION)
    
    if not args.command:
        parser.print_help()
        return 0
    
    if args.command == "version":
        print(f"POLARIS Framework v{VERSION}")
        return 0
    
    # Initialize Manager
    manager = PolarisFrameworkManager()
    setup_signal_handlers(manager)
    
    # Command dispatch
    commands = {
        "start": cmd_start,
        "shell": cmd_shell,
        "dashboard": cmd_dashboard,
        "status": cmd_status,
        "systems": cmd_systems,
    }
    
    handler = commands.get(args.command)
    if handler:
        return await handler(args, manager)
    else:
        print_error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print_info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print_error(f"Fatal error: {e}")
        sys.exit(1)
