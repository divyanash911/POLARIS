"""
POLARIS Framework CLI Package

Provides command-line interface components for managing and interacting
with the POLARIS framework.
"""

from .manager import PolarisFrameworkManager, ManagedSystemOperations, FrameworkStatus, FrameworkState
from .shell import InteractiveShell
from .dashboard import ObservabilityDashboard
from .utils import (
    print_banner, format_uptime, create_bar, clear_screen,
    get_console, create_header, print_warning, print_error, print_success,
    RICH_AVAILABLE
)

__all__ = [
    'PolarisFrameworkManager',
    'ManagedSystemOperations',
    'FrameworkStatus',
    'FrameworkState',
    'InteractiveShell',
    'ObservabilityDashboard',
    'print_banner',
    'format_uptime',
    'create_bar',
    'clear_screen',
    'get_console',
    'create_header',
    'print_warning',
    'print_error',
    'print_success',
    'RICH_AVAILABLE'
]
