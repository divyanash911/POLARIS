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
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from the script directory
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        print(f"✓ Loaded environment variables from {env_path}")
    else:
        print(f"⚠ No .env file found at {env_path}")
except ImportError:
    print("⚠ python-dotenv not installed, skipping .env file loading")

# Add src to path for imports
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Validate critical environment variables for SWIM system
def validate_environment():
    """Validate critical environment variables."""
    required_vars = {
        "GOOGLE_AI_API_KEY": "Google AI API key for LLM reasoning"
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  {var}: {description}")
    
    if missing_vars:
        print("✗ Missing required environment variables:")
        for var in missing_vars:
            print(var)
        print("\nPlease set these variables in your .env file or environment.")
        return False
    
    print("✓ All required environment variables are set")
    return True

try:
    from framework.cli.manager import PolarisFrameworkManager
    from framework.cli.shell import InteractiveShell
    from framework.cli.utils import print_banner, print_error, print_success, print_info
    from framework.cli.dashboard import ObservabilityDashboard
except ImportError as e:
    print(f"✗ Failed to import framework components: {e}")
    print("Make sure you're in the correct directory and all dependencies are installed.")
    print("Run: python setup_swim_system.py")
    sys.exit(1)

VERSION = "2.0.0"

DEFAULT_CONFIGS = {
    "swim": "config/swim_system_config_optimized.yaml",
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
        
        # Suggest running setup if swim config is missing
        if config_arg == "swim":
            print_info("Run 'python setup_swim_system.py' to set up the SWIM system")
        
        sys.exit(1)
    
    return str(path)


def setup_signal_handlers(manager: PolarisFrameworkManager):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        print_info("\nShutdown signal received, stopping framework...")
        # The manager's run_until_shutdown handles the actual shutdown
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def check_prerequisites(config_name: str) -> bool:
    """Check system prerequisites for the given configuration."""
    if config_name == "swim":
        print_info("Checking SWIM system prerequisites...")
        
        # Check NATS server
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(("localhost", 4222))
            sock.close()
            
            if result != 0:
                print_error("NATS server is not running on localhost:4222")
                print_info("Start NATS server with: nats-server --port 4222")
                print_info("Or use Docker: docker run -p 4222:4222 nats:latest")
                return False
            else:
                print_info("✓ NATS server is accessible")
        except Exception as e:
            print_error(f"Error checking NATS server: {e}")
            return False
        
        # Check environment variables
        if not validate_environment():
            return False
        
        # Check logs directory
        logs_dir = Path(__file__).parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        print_info("✓ Logs directory ready")
    
    return True


async def cmd_start(args, manager: PolarisFrameworkManager) -> int:
    """Handle the 'start' command."""
    config_path = resolve_config_path(args.config)
    
    # Check prerequisites
    if not check_prerequisites(args.config):
        print_error("Prerequisites check failed")
        return 1
    
    print_info(f"Starting POLARIS framework with config: {config_path}")
    
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
        print_info("Framework is running. Press Ctrl+C to stop.")
        await manager.run_until_shutdown()
    
    return 0


async def cmd_shell(args, manager: PolarisFrameworkManager) -> int:
    """Handle the 'shell' command."""
    print_info("Starting interactive shell with mock configuration...")
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
    
    # Check prerequisites
    if not check_prerequisites(args.config):
        print_error("Prerequisites check failed")
        return 1
    
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
    try:
        status = manager.get_status()
        print(f"\nFramework Status: {status.state.value.upper()}")
        print(f"Uptime: {status.uptime_seconds:.1f}s")
        print(f"Components: {len(status.components)}")
        print(f"Managed Systems: {len(status.managed_systems)}")
        print(f"Meta Learner: {'Enabled' if status.meta_learner_enabled else 'Disabled'}")
    except Exception as e:
        print_error(f"Failed to get status: {e}")
        return 1
    return 0


async def cmd_systems(args, manager: PolarisFrameworkManager) -> int:
    """Handle the 'systems' command."""
    print_info("Systems management requires active framework.")
    print_info("Use 'start --shell' for interactive mode or 'dashboard' for monitoring.")
    return 0


def cmd_setup(args) -> int:
    """Handle the 'setup' command."""
    print_info("Running SWIM system setup...")
    
    # Import and run setup script
    setup_script = Path(__file__).parent / "setup_swim_system.py"
    if setup_script.exists():
        import subprocess
        result = subprocess.run([sys.executable, str(setup_script)], cwd=Path(__file__).parent)
        return result.returncode
    else:
        print_error("Setup script not found")
        return 1


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="POLARIS Framework CLI - Proactive Optimization & Learning Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s setup                    Set up SWIM system environment
  %(prog)s start                    Start with default (mock) config
  %(prog)s start -c swim            Start with SWIM system config
  %(prog)s start -c swim --shell    Start SWIM system with interactive shell
  %(prog)s dashboard -c swim        Start SWIM system dashboard
  %(prog)s shell                    Start standalone interactive shell
  %(prog)s status                   Check framework status
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Setup command
    subparsers.add_parser("setup", help="Set up SWIM system environment")
    
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
        print("Proactive Optimization & Learning Architecture for Resilient Intelligent Systems")
        return 0
    
    if args.command == "setup":
        return cmd_setup(args)
    
    # Initialize Manager for other commands
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
        import traceback
        traceback.print_exc()
        sys.exit(1)
