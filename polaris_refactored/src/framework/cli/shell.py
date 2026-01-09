
import asyncio
from datetime import datetime
from typing import Dict, Any, List

from .manager import PolarisFrameworkManager, ManagedSystemOperations
from .dashboard import ObservabilityDashboard
from .utils import (
    RICH_AVAILABLE, rprint, get_console, create_header, 
    print_warning, print_error, print_success, Table, box
)

# Import Table and box for rich tables if available
if RICH_AVAILABLE:
    try:
        from rich.table import Table
        from rich import box
    except ImportError:
        Table = None
        box = None

class InteractiveShell:
    """Interactive shell for POLARIS management."""
    
    COMMANDS = {
        "help": "Show available commands",
        "status": "Show framework status",
        "systems": "List managed systems",
        "metrics <system_id>": "Show metrics for a system",
        "action <system_id> <action_type> [params]": "Execute an action",
        "actions <system_id>": "List supported actions for a system",
        "config": "Show current configuration",
        "dashboard [duration]": "Show live metrics dashboard",
        "history": "Show adaptation history",
        "meta-learner": "Check meta learner status",
        "world-model": "Inspect World Model status",
        "export-kb <filename>": "Export Knowledge Base to file",
        "quit": "Exit the shell"
    }
    
    def __init__(self, manager: PolarisFrameworkManager):
        self.manager = manager
        self.ops = ManagedSystemOperations(manager)
        self.dashboard = ObservabilityDashboard(manager)
        self.running = True
    
    async def run(self):
        """Run the interactive shell."""
        print("\n" + "=" * 70)
        print("POLARIS Interactive Shell")
        print("=" * 70)
        print("Type 'help' for available commands, 'quit' to exit.\n")
        
        while self.running:
            try:
                # Use standard input - complicated to do async input properly in cross-platform way
                # gracefully, so using run_in_executor for input could be better but basic input() works 
                # for simple shells if we don't need background concurrency for the shell itself while waiting.
                # However, since the framework is running in background tasks, we should utilize asyncio.
                cmd_input = await asyncio.get_event_loop().run_in_executor(None, input, "polaris> ")
                cmd_input = cmd_input.strip()
                
                if not cmd_input:
                    continue
                
                await self._process_command(cmd_input)
                
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit.")
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Goodbye!")
    
    async def cmd_systems(self, args: List[str]) -> None:
        """Handle systems command."""
        status = await self.manager.get_system_status()
        
        console = get_console()
        console.print(create_header("Managed Systems Analysis"))
        
        if not status.get("systems"):
            print_warning("No managed systems registered.")
            return
        
        # Use rich table if available, otherwise plain text
        if RICH_AVAILABLE and Table and box:
            table = Table(title="Registered Systems", box=box.ROUNDED)
            table.add_column("System ID", style="cyan", no_wrap=True)
            table.add_column("Status", style="green")
            
            for sys_id in status["systems"]:
                table.add_row(sys_id, "ACTIVE")
                
            console.print(table)
        else:
            print(f"\nRegistered Systems ({len(status['systems'])}):")
            for sys_id in status["systems"]:
                print(f"  🟢 {sys_id} - ACTIVE")
            print("")
        
    async def cmd_world_model(self, args: List[str]) -> None:
        """Handle world-model command."""
        status = await self.manager.get_world_model_status()
        
        console = get_console()
        console.print(create_header("Digital Twin World Model"))
        
        if status.get("status") in ("NOT_AVAILABLE", "ERROR"):
            print_error(f"World Model Error: {status.get('error')}")
            return
            
        import json
        console.print_json(json.dumps(status))

    async def cmd_export_kb(self, args: List[str]) -> None:
        """Handle export-kb command."""
        if not args:
            print_error("Usage: export-kb <filename>")
            return
            
        filename = args[0]
        console = get_console()
        with console.status(f"[bold green]Exporting knowledge base to {filename}..."):
            result = await self.manager.export_knowledge_base(filename)
            
        if result.get("status") == "SUCCESS":
            print_success(f"Knowledge Base exported to: {result['file']}")
            console.print(f"Items exported: {result['counts']}")
        else:
            print_error(f"Export failed: {result.get('error')}")

    async def _process_command(self, cmd_input: str):
        """Process a shell command."""
        parts = cmd_input.split()
        if not parts:
            return
            
        command = parts[0].lower()
        args = parts[1:]
        
        if command == "quit" or command == "exit":
            self.running = False
            return
            
        elif command == "help":
            self._show_help()
            
        elif command == "status":
            self._show_status()
            
        elif command == "systems":
            await self.cmd_systems(args)
            
        elif command == "metrics":
            if not args:
                print("Usage: metrics <system_id>")
            else:
                await self._show_metrics(args[0])
        
        elif command == "dashboard":
            duration = int(args[0]) if args else 60
            system_id = args[1] if len(args) > 1 else None
            await self.dashboard.display_live_metrics(system_id, duration)
            
        elif command == "history":
            limit = int(args[0]) if args else 20
            await self.dashboard.show_adaptation_history(limit)

        elif command == "actions":
            if not args:
                print("Usage: actions <system_id>")
            else:
                await self._list_actions(args[0])

        elif command == "action":
            if len(args) < 2:
                print("Usage: action <system_id> <action_type> [key=value ...]")
            else:
                await self._execute_action(args[0], args[1], args[2:])

        elif command == "world-model":
            await self.cmd_world_model(args)
            
        elif command == "meta-learner":
            await self._show_meta_learner_status()
            
        elif command == "export-kb":
            await self.cmd_export_kb(args)
            
        elif command == "config":
            self._show_config()

        else:
            print(f"Unknown command: {command}")

    def _show_help(self):
        print("\nAvailable Commands:")
        for cmd, desc in self.COMMANDS.items():
            print(f"  {cmd:40} {desc}")
        print("")

    def _show_status(self):
        status = self.manager.get_status()
        print(f"\nStatus: {status.state.value.upper()}")
        print(f"Uptime: {status.uptime_seconds:.1f}s")
        print(f"Active Components: {len(status.components)}")
        print("")

    async def _list_systems(self):
        systems = await self.ops.list_systems()
        print(f"\nManaged Systems ({len(systems)}):")
        for sys in systems:
            status_icon = "🟢" if sys['enabled'] else "⚪"
            print(f"  {status_icon} {sys['system_id']} ({sys['connector_type']})")
        print("")

    async def _show_metrics(self, system_id: str):
        data = await self.ops.collect_metrics(system_id)
        if "error" in data:
            print(f"Error: {data['error']}")
        else:
            print(f"\nMetrics for {system_id}:")
            metrics = data.get("metrics", {})
            for name, mdata in metrics.items():
                print(f"  {name}: {mdata.get('value')} {mdata.get('unit', '')}")
            print("")
            
    async def _list_actions(self, system_id: str):
        actions = await self.ops.get_supported_actions(system_id)
        print(f"\nSupported Actions for {system_id}:")
        if not actions:
            print("  None or system not found")
        for action in actions:
            print(f"  - {action}")
        print("")

    async def _execute_action(self, system_id: str, action_type: str, params_list: list):
        # Parse params like key=value
        params = {}
        for p in params_list:
            if '=' in p:
                k, v = p.split('=', 1)
                # Try to cast to number if possible
                try:
                    if '.' in v:
                        v = float(v)
                    else:
                        v = int(v)
                except ValueError:
                    pass
                params[k] = v
        
        print(f"Executing {action_type} on {system_id} with {params}...")
        result = await self.ops.execute_action(system_id, action_type, params)
        print(f"Result: {result}")

    async def _show_meta_learner_status(self):
        """Show meta learner status."""
        status = self.manager.get_status()
        print("\nMeta Learner Status:")
        if status.meta_learner_enabled:
            print("  Status: ENABLED")
            print("  The meta learner is actively monitoring and optimizing the system.")
        else:
            print("  Status: DISABLED")
            print("  Meta learner is not configured or LLM client is not available.")
        print("")

    def _show_config(self):
        """Show current configuration summary."""
        print("\nCurrent Configuration:")
        if self.manager.config_path:
            print(f"  Config File: {self.manager.config_path}")
        else:
            print("  Config File: Not loaded")
        
        status = self.manager.get_status()
        print(f"  Framework State: {status.state.value}")
        print(f"  Components: {len(status.components)}")
        if status.components:
            for comp in status.components:
                print(f"    - {comp}")
        print(f"  Managed Systems: {len(status.managed_systems)}")
        for sys_id, sys_info in status.managed_systems.items():
            print(f"    - {sys_id}: {sys_info.get('connector_type', 'unknown')}")
        print("")

