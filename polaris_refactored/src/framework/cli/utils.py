
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
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None  # type: ignore
    Table = None  # type: ignore
    Panel = None  # type: ignore
    rprint = print

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

