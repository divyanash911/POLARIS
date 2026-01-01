# POLARIS CLI Reference

The POLARIS Framework provides a comprehensive Command Line Interface (CLI) for managing the framework, interacting with managed systems, and inspecting the digital twin.

## Main Entry Point

The main entry point is `start_polaris_framework.py`.

```bash
python start_polaris_framework.py [COMMAND] [OPTIONS]
```

## Global Options

- `-h, --help`: Show help message and exit.

## Subcommands

### `start`
Start the POLARIS framework in the background or foreground.

**Usage:**
```bash
python start_polaris_framework.py start [--config <path>] [--shell]
```

**Options:**
- `--config <path>`: Path to configuration file (default: `config/mock_system_config.yaml`).
- `--shell`: Start the interactive shell immediately after framework startup.

### `shell`
Start the interactive management shell. Connects to a running instance if available, or starts a standalone instance.

**Usage:**
```bash
python start_polaris_framework.py shell
```

### `dashboard`
Start the real-time observability dashboard in the terminal.

**Usage:**
```bash
python start_polaris_framework.py dashboard [--system <system_id>]
```

### `status`
Check the status of the framework and managed systems.

**Usage:**
```bash
python start_polaris_framework.py status
```

### `systems`
List and manage connected systems.

**Usage:**
```bash
python start_polaris_framework.py systems
```

---

## Interactive Shell Commands

Once inside the shell (`python start_polaris_framework.py shell`), the following commands are available:

### General
- `help`: Show available commands.
- `status`: Show framework operational status and uptime.
- `config`: Display the current loaded configuration.
- `quit` / `exit`: Exit the shell.
- `clear`: Clear the screen.

### System Management
- `systems`: List all managed systems and their status.
- `metrics <system_id>`: View current telemetry metrics for a specific system.
- `actions <system_id>`: List supported adaptation actions for a system.
- `action <system_id> <type> [key=value ...]`: Execute an action on a system.
  - Example: `action demo scale_out factor=1.5`

### Digital Twin & Knowledge Base
- `world-model`: Inspect the status and structure of the Digital Twin World Model.
- `export-kb <filename>`: Export the entire Knowledge Base (patterns, dependencies, history) to a JSON file.
  - Example: `export-kb ./dumps/kb_backup.json`

### Observability
- `dashboard [duration]`: Launch the live dashboard (press `q` to exit back to shell).
- `history [limit]`: View recent adaptation history events.
- `meta-learner`: Check the status of the Meta-Learning subsystem.
