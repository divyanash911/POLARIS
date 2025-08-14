# POLARIS - Plugin-Driven Adaptation Framework

POLARIS is a modular, extensible framework for monitoring and controlling self-adaptive systems. It uses a plugin-driven architecture that allows easy integration with different managed systems while providing a consistent interface for adaptation logic.

## 🏗️ Architecture Overview

POLARIS follows a clean separation between the **core framework** and **managed system plugins**:

- **Core Framework**: Generic adapters, NATS communication, configuration management, data models
- **Managed System Plugins**: System-specific connectors and configurations
- **Observability Tools**: Real-time monitoring and debugging utilities

```
┌─────────────────────────────────────────────────────────────┐
│                    POLARIS Framework                        │
├─────────────────────────────────────────────────────────────┤
│  Monitor Adapter    │    Execution Adapter    │   Tools     │
│  ┌─────────────────┐│  ┌─────────────────────┐│  ┌─────────┐│
│  │ Metric Collection││  │ Action Execution    ││  │NATS Spy ││
│  │ Telemetry Batch ││  │ Result Publishing   ││  │Debugger ││
│  │ NATS Publishing ││  │ Queue Management    ││  │Validator││
│  └─────────────────┘│  └─────────────────────┘│  └─────────┘│
├─────────────────────────────────────────────────────────────┤
│                    Plugin Interface                         │
├─────────────────────────────────────────────────────────────┤
│  SWIM Plugin        │    Custom Plugin        │   Future    │
│  ┌─────────────────┐│  ┌─────────────────────┐│  ┌─────────┐│
│  │ TCP Connector   ││  │ HTTP Connector      ││  │   ...   ││
│  │ Config Schema   ││  │ Config Schema       ││  │         ││
│  │ Retry Logic     ││  │ Auth Handling       ││  │         ││
│  └─────────────────┘│  └─────────────────────┘│  └─────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NATS Server (included in `bin/`)
- Required Python packages (see `requirements.txt`)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Start NATS server
./bin/nats-server

# Validate configuration
python src/scripts/start_component.py monitor --plugin-dir extern --validate-only

# Start monitor adapter
python src/scripts/start_component.py monitor --plugin-dir extern

# Start execution adapter (in another terminal)
python src/scripts/start_component.py execution --plugin-dir extern
```

### Monitor NATS Messages
```bash
# Monitor all POLARIS messages
python src/scripts/nats_spy.py

# Monitor only telemetry
python src/scripts/nats_spy.py --preset telemetry

# Monitor with full message content
python src/scripts/nats_spy.py --show-data
```

## 📁 Directory Structure

```
polaris_poc/
├── bin/                          # Executables (NATS server)
├── extern/                       # Managed system plugins
│   ├── config.yaml              # SWIM plugin configuration
│   ├── connector.py             # SWIM TCP connector
│   └── README.md                # Plugin development guide
├── src/
│   ├── config/                  # Framework configuration
│   │   ├── managed_system.schema.json
│   │   └── polaris_config.yaml
│   ├── polaris/
│   │   ├── adapters/            # Core adapters
│   │   │   ├── base.py         # Base classes and interfaces
│   │   │   ├── monitor.py      # Generic monitor adapter
│   │   │   └── execution.py    # Generic execution adapter
│   │   ├── common/              # Shared utilities
│   │   │   ├── config.py       # Configuration management
│   │   │   ├── nats_client.py  # NATS communication
│   │   │   └── logging_setup.py
│   │   └── models/              # Data models
│   │       ├── actions.py      # Control actions and results
│   │       └── telemetry.py    # Telemetry events
│   └── scripts/                 # Utility scripts
│       ├── start_component.py  # Main entry point
│       └── nats_spy.py         # NATS message monitor
├── test_adapters.py             # Integration tests
└── requirements.txt
```

## Setup Instructions
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Place your NATS server executable in the `/bin` directory.
5. Clone the SWIM project into the `/extern` directory.
6. Fill in your OpenAI API key in the `.env` file.


## 🔧 Core Components

### Monitor Adapter
- **Purpose**: Collects metrics from managed systems and publishes telemetry
- **Features**: 
  - Plugin-driven metric collection
  - Derived metric calculations
  - Batch and streaming telemetry publishing
  - Configurable collection strategies
  - Error handling and retry logic

### Execution Adapter
- **Purpose**: Executes control actions on managed systems
- **Features**:
  - Action validation and precondition checking
  - Parameter type and range validation
  - Concurrent execution control
  - Result publishing and metrics
  - Queue management with throttling

### Plugin System
- **Purpose**: Encapsulates system-specific logic
- **Features**:
  - Declarative configuration via YAML
  - JSON schema validation
  - Dynamic connector loading
  - Standardized interface
  - Easy extensibility

## 🔌 Plugin Development

### Creating a New Plugin

1. **Create Plugin Directory**
   ```bash
   mkdir my_system_plugin
   cd my_system_plugin
   touch __init__.py
   ```

2. **Define Configuration** (`config.yaml`)
   ```yaml
   system_name: "my_system"
   implementation:
     connector_class: "connector.MySystemConnector"
   connection:
     protocol: "http"
     host: "localhost"
     port: 8080
   monitoring:
     metrics:
       - name: "status"
         command: "GET /health"
         unit: "boolean"
   execution:
     actions:
       - type: "RESTART"
         command: "POST /restart"
   ```

3. **Implement Connector** (`connector.py`)
   ```python
   from polaris.adapters.base import ManagedSystemConnector
   
   class MySystemConnector(ManagedSystemConnector):
       async def connect(self):
           # Implementation here
           pass
       
       async def execute_command(self, command, params=None):
           # Implementation here
           pass
   ```

4. **Test Plugin**
   ```bash
   python src/scripts/start_component.py monitor --plugin-dir my_system_plugin --validate-only
   ```

See `extern/README.md` for detailed plugin development guide.

## 🛠️ Development Tools

### Configuration Validation
```bash
# Validate plugin configuration
python src/scripts/start_component.py monitor --plugin-dir extern --validate-only

# Dry run (initialize but don't start)
python src/scripts/start_component.py monitor --plugin-dir extern --dry-run
```

### NATS Message Monitoring
```bash
# Monitor all messages
python src/scripts/nats_spy.py

# Monitor specific subjects
python src/scripts/nats_spy.py --subjects "polaris.telemetry.>" "polaris.execution.>"

# Show full message content
python src/scripts/nats_spy.py --show-data

# Use presets
python src/scripts/nats_spy.py --preset telemetry
python src/scripts/nats_spy.py --preset execution
python src/scripts/nats_spy.py --preset results
```

### Debug Logging
```bash
# Enable debug logging
python src/scripts/start_component.py monitor --plugin-dir extern --log-level DEBUG
```

## 📊 SWIM Plugin Example

The included SWIM plugin demonstrates a complete implementation:

### Metrics Collected
- Server counts (active, max, total)
- Response times (basic, optional, average)
- Throughput (basic, optional)
- Arrival rate and utilization

### Actions Supported
- `ADD_SERVER` - Add server with capacity checks
- `REMOVE_SERVER` - Remove server with minimum checks
- `SET_DIMMER` - Adjust QoS (0.0-1.0 range)

### Key Features
- TCP socket communication with retry logic
- Parameter validation and precondition checking
- Derived metric calculations
- Comprehensive error handling

## 🔍 Monitoring and Observability

### NATS Subjects
- `polaris.telemetry.events.stream` - Individual telemetry events
- `polaris.telemetry.events.batch` - Batched telemetry events
- `polaris.execution.actions` - Control actions to execute
- `polaris.execution.results` - Action execution results
- `polaris.execution.metrics` - Execution performance metrics

### Message Flow
```
Monitor Adapter → NATS → [Reasoning/Planning] → NATS → Execution Adapter
     ↓                                                        ↓
Telemetry Events                                    Action Results
```

## 🧪 Testing

### Integration Tests
```bash
# Run adapter tests
python test_adapters.py

# Test specific components
python -m pytest tests/ -v
```

### Manual Testing
```bash
# Start components
python src/scripts/start_component.py monitor --plugin-dir extern &
python src/scripts/start_component.py execution --plugin-dir extern &

# Monitor messages
python src/scripts/nats_spy.py --preset all

# Send test action (requires NATS client)
nats pub polaris.execution.actions '{"action_type":"SET_DIMMER","params":{"value":0.8}}'
```

## 🔧 Configuration

### Framework Configuration (`src/config/polaris_config.yaml`)
- NATS connection settings
- Telemetry batching parameters
- Logging configuration
- Default timeouts and retries

### Plugin Configuration (`extern/config.yaml`)
- System identification and metadata
- Connection parameters
- Metric definitions and collection strategies
- Action definitions and validation rules
- Execution constraints

## 🚨 Troubleshooting

### Common Issues

1. **Plugin Not Found**
   - Check plugin directory path
   - Ensure `__init__.py` exists
   - Verify connector class path

2. **Configuration Validation Errors**
   - Use `--validate-only` flag
   - Check against JSON schema
   - Review error messages for specific issues

3. **Connection Failures**
   - Verify managed system is running
   - Check connection parameters
   - Review connector implementation

4. **NATS Connection Issues**
   - Ensure NATS server is running
   - Check NATS URL configuration
   - Verify network connectivity

### Debug Steps
1. Enable debug logging (`--log-level DEBUG`)
2. Use validation mode (`--validate-only`)
3. Try dry run mode (`--dry-run`)
4. Monitor NATS messages (`nats_spy.py`)
5. Check connector health methods
