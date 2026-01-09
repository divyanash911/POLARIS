
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Try to import rich
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    from rich import print as rprint
    RICH_AVAILABLE = True
    _console = Console()
except ImportError:
    RICH_AVAILABLE = False
    Console = None  # type: ignore
    Table = None  # type: ignore
    Panel = None  # type: ignore
    box = None  # type: ignore
    rprint = print
    _console = None

def get_script_dir() -> Path:
    """Get the absolute path to the script directory."""
    # This assumes this file is in src/framework/cli/
    # We want to return the project root or the dir containing the entry script
    # Let's say we want the root of the repo which seems to be 3 levels up from here if we are in src/framework/cli
    # But more reliably, we might just want where imports are resolved from.
    # For now, let's return the parent of src/
    return Path(__file__).parent.parent.parent.parent

def format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def create_bar(value: float, max_val: float = 100, width: int = 20) -> str:
    """Create a simple ASCII progress bar."""
    if not isinstance(value, (int, float)) or value < 0:
        return ""
    
    # Normalize value
    if max_val > 0:
        ratio = min(value / max_val, 1.0)
    else:
        ratio = 0
    
    filled = int(width * ratio)
    empty = width - filled
    
    # Color coding based on value - just return string with simple indicators if needed
    # The caller can handle color if using rich
    return f"[{'█' * filled}{'░' * empty}]"

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner(version: str):
    """Print the POLARIS banner."""
    banner = r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██████╗  ██████╗ ██╗      █████╗ ██████╗ ██╗███████╗                        ║
║   ██╔══██╗██╔═══██╗██║     ██╔══██╗██╔══██╗██║██╔════╝                        ║
║   ██████╔╝██║   ██║██║     ███████║██████╔╝██║███████╗                        ║
║   ██╔═══╝ ██║   ██║██║     ██╔══██║██╔══██╗██║╚════██║                        ║
║   ██║     ╚██████╔╝███████╗██║  ██║██║  ██║██║███████║                        ║
║   ╚═╝      ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝                        ║
║                                                                               ║
║   Proactive Optimization & Learning Architecture for Resilient               ║
║   Intelligent Systems                                                         ║
║                                                                               ║
║   Version: {version:<10}                                                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """.format(version=version)
    
    if RICH_AVAILABLE:
        rprint(f"[bold cyan]{banner}[/bold cyan]")
    else:
        print(banner)


def get_console() -> Any:
    """Get the Rich console instance or a fallback."""
    if RICH_AVAILABLE and _console:
        return _console
    # Return a simple fallback that mimics basic console operations
    class FallbackConsole:
        def print(self, *args, **kwargs):
            # Strip rich markup if present
            text = str(args[0]) if args else ""
            print(text)
        def print_json(self, data):
            print(data)
        def clear(self):
            clear_screen()
        def status(self, message):
            return _FallbackStatus(message)
    return FallbackConsole()


class _FallbackStatus:
    """Fallback context manager for console.status()."""
    def __init__(self, message: str):
        self.message = message
    def __enter__(self):
        print(self.message)
        return self
    def __exit__(self, *args):
        pass


def create_header(title: str) -> Any:
    """Create a styled header panel."""
    if RICH_AVAILABLE and Panel:
        return Panel(title, style="bold blue", expand=False)
    return f"\n{'=' * 60}\n{title}\n{'=' * 60}"


def print_warning(message: str) -> None:
    """Print a warning message."""
    if RICH_AVAILABLE:
        rprint(f"[yellow]⚠️  {message}[/yellow]")
    else:
        print(f"WARNING: {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    if RICH_AVAILABLE:
        rprint(f"[bold red]❌ {message}[/bold red]")
    else:
        print(f"ERROR: {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    if RICH_AVAILABLE:
        rprint(f"[bold green]✅ {message}[/bold green]")
    else:
        print(f"SUCCESS: {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    if RICH_AVAILABLE:
        rprint(f"[cyan]ℹ️  {message}[/cyan]")
    else:
        print(f"INFO: {message}")
