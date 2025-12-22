# Mock System Connector Plugin

A POLARIS connector plugin for the Mock External System, enabling comprehensive testing of the POLARIS framework without requiring real external infrastructure.

## Overview

The Mock System Connector implements the `ManagedSystemConnector` interface to communicate with the Mock External System via TCP. It provides:

- **Metric Collection**: Retrieves simulated system metrics (CPU, memory, response time, etc.)
- **Action Execution**: Executes adaptation actions (scale up/down, QoS adjustments, etc.)
- **Action Validation**: Validates if actions can be executed given current system state
- **State Retrieval**: Gets current system state and health information

## Installation

The plugin is automatically discovered by POLARIS when placed in the `plugins/mock_system` directory.

## Configuration

### Basic Configuration

```yaml
managed_systems:
  mock_system:
    system_id: "mock_system"
    connector_type: "mock_system"
    enabled: true
    
    connection:
      host: "localhost"
      port: 5000
    
    implementation:
      timeout: 10.0
      max_retries: 3
      retry_base_delay: 1.0
      retry_max_delay: 5.0
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `connection.host` | string | localhost | Mock system server host |
| `connection.port` | int | 5000 | Mock system server port |
| `implementation.timeout` | float | 10.0 | Command timeout in seconds |
| `implementation.max_retries` | int | 3 | Maximum retry attempts |
| `implementation.retry_base_delay` | float | 1.0 | Base delay between retries |
| `implementation.retry_max_delay` | float | 5.0 | Maximum retry delay |

## Supported Actions

| Action Type | Description | Parameters |
|-------------|-------------|------------|
| `SCALE_UP` | Increase system capacity | `increment` (optional) |
| `SCALE_DOWN` | Decrease system capacity | `decrement` (optional) |
| `ADJUST_QOS` | Adjust quality of service | `mode` (optional) |
| `RESTART_SERVICE` | Restart the service | None |
| `OPTIMIZE_CONFIG` | Apply optimization | None |
| `ENABLE_CACHING` | Enable caching layer | None |
| `DISABLE_CACHING` | Disable caching layer | None |

## Metrics Provided

| Metric | Unit | Description |
|--------|------|-------------|
| `cpu_usage` | percent | CPU utilization (0-100%) |
| `memory_usage` | MB | Memory usage in megabytes |
| `response_time` | ms | Average response time |
| `throughput` | req/s | Requests per second |
| `error_rate` | percent | Error rate percentage |
| `active_connections` | count | Active connection count |
| `capacity` | count | Current capacity units |

## Protocol

The connector uses a text-based TCP protocol:

**Request Format:**
```
COMMAND [arg1] [arg2] ... [key=value]
```

**Response Format:**
```
STATUS|{"data": ..., "message": ...}
```

Where `STATUS` is either `OK` or `ERROR`.

## Usage Example

```python
from plugins.mock_system.connector import MockSystemConnector

# Create connector with configuration
config = {
    "system_name": "test_mock",
    "connection": {
        "host": "localhost",
        "port": 5000
    }
}

connector = MockSystemConnector(config)

# Connect to mock system
await connector.connect()

# Collect metrics
metrics = await connector.collect_metrics()
print(f"CPU Usage: {metrics['cpu_usage'].value}%")

# Execute an action
from src.domain.models import AdaptationAction
action = AdaptationAction(
    action_id="test-1",
    action_type="SCALE_UP",
    target_system="test_mock",
    parameters={"increment": 2}
)
result = await connector.execute_action(action)
print(f"Action result: {result.status}")

# Disconnect
await connector.disconnect()
```

## Testing

The connector includes comprehensive tests:

```bash
# Run unit tests
pytest polaris_refactored/tests/unit/adapters/test_mock_system_connector.py

# Run integration tests (requires mock system running)
pytest polaris_refactored/tests/integration/test_mock_system_integration.py
```

## Requirements

- Python 3.8+
- Mock External System running on configured host/port
- POLARIS framework dependencies

## Related Documentation

- [Mock External System README](../../mock_external_system/README.md)
- [POLARIS Plugin Architecture](../../doc/08-framework-layer.md)
- [Managed System Connector Interface](../../src/domain/interfaces.py)
