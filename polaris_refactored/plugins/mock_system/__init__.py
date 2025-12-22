"""
Mock System Plugin for POLARIS Framework.

This plugin provides a connector for the mock external system,
enabling testing of POLARIS components without real infrastructure.
"""

from .connector import MockSystemConnector

__all__ = ["MockSystemConnector"]
