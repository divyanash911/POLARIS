
import asyncio
import time
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

from framework.cli.utils import RICH_AVAILABLE, Console, Table, Panel, create_bar, format_uptime, clear_screen, print_banner, rprint

class ObservabilityDashboard:
    """Real-time observability and monitoring dashboard."""
    
    def __init__(self, manager):
        self.manager = manager
        self.refresh_interval = 5.0  # seconds
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
    
    async def display_live_metrics(self, system_id: Optional[str] = None, duration: int = 60):
        """Display live metrics in the terminal."""
        # Avoid circular imports by importing locally if needed, though manager is passed in
        from framework.cli.manager import ManagedSystemOperations
        ops = ManagedSystemOperations(self.manager)
        
        end_time = time.time() + duration
        
        self._print_dashboard_header(duration)
        
        try:
            while time.time() < end_time:
                # We need to render entirely differently for Rich vs Plain
                if RICH_AVAILABLE:
                    self.console.clear()
                    await self._render_rich_dashboard(system_id, ops)
                else:
                    clear_screen()
                    await self._render_plain_dashboard(system_id, ops)
                
                await asyncio.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\nMetrics display stopped.")

    def _print_dashboard_header(self, duration: int):
        if RICH_AVAILABLE:
            self.console.print(Panel(f"Refresh interval: {self.refresh_interval}s | Duration: {duration}s\nPress Ctrl+C to stop", title="POLARIS Live Metrics Dashboard", style="bold green"))
        else:
            print("\n" + "=" * 70)
            print("POLARIS Live Metrics Dashboard")
            print("=" * 70)
            print(f"Refresh interval: {self.refresh_interval}s | Duration: {duration}s")
            print("Press Ctrl+C to stop\n")

    async def _render_plain_dashboard(self, system_id: Optional[str], ops):
        # Re-use the existing plain text rendering logic mostly
        print_banner("2.0.0") # We might want to pass version or get it dynamically
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 70)
        
        status = self.manager.get_status()
        print(f"Framework State: {status.state.value.upper()}")
        print(f"Uptime: {format_uptime(status.uptime_seconds)}")
        print(f"Active Components: {len(status.components)}")
        print("-" * 70)
        
        systems = await ops.list_systems()
        target_systems = [s for s in systems if s['enabled']]
        
        if system_id:
            target_systems = [s for s in target_systems if s['system_id'] == system_id]
        
        for sys_info in target_systems:
            await self._render_plain_system_metrics(sys_info, ops)
        
        print("\n" + "-" * 70)
        print("Press Ctrl+C to stop")

    async def _render_plain_system_metrics(self, sys_info: Dict[str, Any], ops):
        sid = sys_info['system_id']
        print(f"\n📊 System: {sid} ({sys_info['connector_type']})")
        print("-" * 40)
        
        metrics_data = await ops.collect_metrics(sid)
        
        if "error" in metrics_data:
            print(f"  ⚠️  Error: {metrics_data['error']}")
        else:
            metrics = metrics_data.get("metrics", {})
            for name, data in metrics.items():
                value = data.get('value', 'N/A')
                unit = data.get('unit', '')
                bar = create_bar(value) if isinstance(value, (int, float)) else ""
                print(f"  {name:25} {value:>10} {unit:5} {bar}")

    async def _render_rich_dashboard(self, system_id: Optional[str], ops):
        # Header Table
        status = self.manager.get_status()
        status_table = Table(show_header=False, box=None)
        status_table.add_column("Key", style="cyan")
        status_table.add_column("Value", style="magenta")
        status_table.add_row("Framework State", status.state.value.upper())
        status_table.add_row("Uptime", format_uptime(status.uptime_seconds))
        status_table.add_row("Active Components", str(len(status.components)))
        
        self.console.print(Panel(status_table, title=f"Polaris Status - {datetime.now().strftime('%H:%M:%S')}", border_style="blue"))
        
        # Systems Section
        systems = await ops.list_systems()
        target_systems = [s for s in systems if s['enabled']]
        
        if system_id:
            target_systems = [s for s in target_systems if s['system_id'] == system_id]
            
        for sys_info in target_systems:
            await self._render_rich_system_metrics(sys_info, ops)

    async def _render_rich_system_metrics(self, sys_info: Dict[str, Any], ops):
        sid = sys_info['system_id']
        metrics_data = await ops.collect_metrics(sid)
        
        table = Table(title=f"System: {sid} ({sys_info['connector_type']})", expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        table.add_column("Unit", style="yellow")
        table.add_column("Visual", style="blue")

        if "error" in metrics_data:
            self.console.print(f"[bold red]Error fetching metrics for {sid}: {metrics_data['error']}[/bold red]")
            return

        metrics = metrics_data.get("metrics", {})
        for name, data in metrics.items():
            value = data.get('value', 'N/A')
            unit = data.get('unit', '')
            bar = create_bar(value) if isinstance(value, (int, float)) else ""
            table.add_row(name, str(value), unit, bar)
            
        self.console.print(table)
    
    async def show_adaptation_history(self, limit: int = 20):
        """Show recent adaptation history."""
        if RICH_AVAILABLE:
            self._show_rich_adaptation_history(limit)
        else:
            self._show_plain_adaptation_history(limit)
            
    def _show_plain_adaptation_history(self, limit: int):
        print("\n" + "=" * 70)
        print("POLARIS Adaptation History")
        print("=" * 70)
        self._print_history_items(limit)

    def _show_rich_adaptation_history(self, limit: int):
        self.console.print(Panel("Adaptation History", style="bold blue"))
        self._print_history_items(limit)

    def _print_history_items(self, limit: int):
        if self.manager.framework and hasattr(self.manager.framework, 'event_bus'):
            try:
                event_bus = self.manager.framework.event_bus
                if hasattr(event_bus, 'get_event_history'):
                    history = event_bus.get_event_history(limit=limit)
                    
                    if not history:
                        msg = "No adaptation events recorded yet."
                        if RICH_AVAILABLE: self.console.print(msg) 
                        else: print(msg)
                        return
                    
                    if RICH_AVAILABLE:
                        table = Table(show_header=True, header_style="bold magenta")
                        table.add_column("Timestamp")
                        table.add_column("Type")
                        table.add_column("System")
                        table.add_column("Details")
                    
                    for event in history:
                        ts = event.get('timestamp', 'N/A')
                        etype = event.get('event_type', 'N/A')
                        sys_id = event.get('system_id', 'N/A')
                        details = str(event.get('data', {}))
                        
                        if RICH_AVAILABLE:
                            table.add_row(str(ts), etype, sys_id, details)
                        else:
                            print(f"\n{ts}")
                            print(f"  Type: {etype}")
                            print(f"  System: {sys_id}")
                            print(f"  Details: {details}")
                            
                    if RICH_AVAILABLE:
                        self.console.print(table)
                else:
                    msg = "Event history not available."
                    if RICH_AVAILABLE: self.console.print(msg)
                    else: print(msg)
            except Exception as e:
                msg = f"Error retrieving history: {e}"
                if RICH_AVAILABLE: self.console.print(f"[red]{msg}[/red]")
                else: print(msg)
        else:
            msg = "Framework not running."
            if RICH_AVAILABLE: self.console.print(f"[yellow]{msg}[/yellow]")
            else: print(msg)

