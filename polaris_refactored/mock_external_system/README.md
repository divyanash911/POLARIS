# Mock External System

A simulated managed system for comprehensive testing of the POLARIS framework.

## Overview

The Mock External System provides a controllable simulation of a real managed system (like a web server or cloud service). It enables complete end-to-end testing of POLARIS components including:

- Monitoring and telemetry collection
- Adaptation decision-making
- Action execution
- Performance measurement

## Project Structure

```
mock_external_system/
├── src/                    # Source code
│   ├── __init__.py
│   ├── server.py           # TCP server implementation
│   ├── metrics_simulator.py # Metrics generation
│   ├── action_handler.py   # Action processing
│   ├── state_manager.py    # State management
│   └── protocol.py         # Communication protocol
├── config/                 # Configuration files
│   ├── default_config.yaml
│   └── scenarios/          # Test scenario configs
├── scripts/                # Startup/shutdown scripts
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
├── setup.py               # Package setup
└── README.md              # This file
```

## Installation

```bash
# Install in development mode
pip install -e ".[dev]"

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

```bash
# Start the mock system
python scripts/start_mock_system.py

# Or with custom configuration
python scripts/start_mock_system.py --config config/scenarios/high_load.yaml
```

## Configuration

See `config/default_config.yaml` for configuration options.

## Testing

```bash
# Run tests
pytest tests/
```

## License

MIT License
