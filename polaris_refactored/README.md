# POLARIS - Proactive Optimization and Learning for Adaptive Runtime Intelligence Systems

POLARIS is a comprehensive self-adaptive systems framework that enables intelligent runtime adaptation through multiple reasoning strategies, including threshold-based reactive control and agentic LLM-powered reasoning.

## Quick Links

- 📚 [SWIM System Setup](README_SWIM_SYSTEM.md) - Running POLARIS with SWIM exemplar
- 🧪 [Testing Guide](TESTING_GUIDE.md) - Comprehensive testing documentation
- ⚡ [Testing Quick Reference](TESTING_QUICKREF.md) - Quick command reference
- 📋 [Test Setup Summary](TEST_SETUP_SUMMARY.md) - Test infrastructure overview

## Quick Start

### Installation

```bash
# Clone the repository
cd polaris_refactored

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/WSL
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Tests

POLARIS includes a comprehensive test suite with multiple easy ways to run tests:

```bash
# Using shell scripts (easiest)
./run_tests.sh unit          # Unit tests
./run_tests.sh integration   # Integration tests
./run_tests.sh all           # All tests

# Using Make (recommended)
make unit                    # Unit tests with coverage
make integration             # Integration tests
make quick                   # Fast unit tests (no coverage)
make all                     # All test suites

# Using Python
python -m tests.run_tests --unit
python -m tests.run_tests --integration
python -m tests.run_tests --all

# Using pytest directly
pytest tests/unit -v
pytest tests/integration --integration -v
```

For detailed testing instructions, see [TESTING_GUIDE.md](TESTING_GUIDE.md).

### Verify Test Setup

```bash
# Run verification script
python verify_tests.py
```

## Features

### Core Framework
- **Layered Architecture**: Framework, Adapters, Digital Twin, Control & Reasoning
- **Event-Driven Design**: Asynchronous event bus for component communication
- **Plugin System**: Extensible managed system connectors
- **Configuration Management**: Hierarchical configuration with hot-reload support
- **Comprehensive Observability**: Structured logging, metrics, and distributed tracing

### Adaptation Strategies
- **Threshold Reactive**: Rule-based adaptation with configurable thresholds
- **Agentic LLM Reasoning**: Intelligent reasoning using Google AI (Gemini)
- **LLM World Model**: Natural language system behavior understanding
- **Hybrid Approaches**: Combine multiple strategies for optimal adaptation

### Testing Infrastructure
- **Unit Tests**: Fast, isolated tests for individual components
- **Integration Tests**: Component interaction testing with test harness
- **Performance Tests**: Throughput, latency, and scalability benchmarks
- **Contract Tests**: Interface compliance validation
- **80%+ Coverage**: Comprehensive test coverage with automated reporting

## Project Structure

```
polaris_refactored/
├── src/                      # Source code
│   ├── framework/           # Core framework layer
│   ├── adapters/            # Monitor and execution adapters
│   ├── digital_twin/        # Digital twin and world model
│   ├── control_reasoning/   # Adaptation strategies
│   ├── infrastructure/      # Infrastructure services
│   └── domain/              # Domain models
│
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   ├── performance/        # Performance tests
│   ├── fixtures/           # Test fixtures
│   └── utils/              # Test utilities
│
├── config/                  # Configuration files
├── plugins/                 # System connector plugins
├── logs/                    # Log files
├── doc/                     # Documentation
│
├── pytest.ini              # Pytest configuration
├── Makefile                # Make targets for testing
├── run_tests.sh            # Test runner script (Linux/WSL)
├── run_tests.bat           # Test runner script (Windows)
├── verify_tests.py         # Test setup verification
│
├── TESTING_GUIDE.md        # Comprehensive testing guide
├── TESTING_QUICKREF.md     # Quick reference card
├── TEST_SETUP_SUMMARY.md   # Test setup overview
└── README_SWIM_SYSTEM.md   # SWIM system setup guide
```

## Architecture

POLARIS follows a layered architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│              (SWIM, Custom Applications)                 │
└─────────────────────────────────────────────────────────┘
                           ▲
                           │
┌─────────────────────────────────────────────────────────┐
│              Control & Reasoning Layer                   │
│  • Threshold Reactive Strategy                          │
│  • Agentic LLM Reasoning Strategy                       │
│  • Adaptation Orchestrator                              │
└─────────────────────────────────────────────────────────┘
                           ▲
                           │
┌─────────────────────────────────────────────────────────┐
│                 Digital Twin Layer                       │
│  • System State Management                              │
│  • LLM World Model                                      │
│  • Behavior Prediction                                  │
└─────────────────────────────────────────────────────────┘
                           ▲
                           │
┌─────────────────────────────────────────────────────────┐
│                   Adapter Layer                          │
│  • Monitor Adapter (Telemetry Collection)               │
│  • Execution Adapter (Action Execution)                 │
└─────────────────────────────────────────────────────────┘
                           ▲
                           │
┌─────────────────────────────────────────────────────────┐
│                  Framework Layer                         │
│  • Event Bus • Configuration • Plugin System            │
│  • Logging • Metrics • Tracing                          │
└─────────────────────────────────────────────────────────┘
```

## Testing

POLARIS includes a comprehensive testing infrastructure:

### Test Types

1. **Unit Tests** - Fast, isolated tests
   - Test individual components in isolation
   - Use dependency injection and mocking
   - Run in milliseconds
   - 80%+ code coverage target

2. **Integration Tests** - Component interaction tests
   - Test component interactions
   - Use integration test harness
   - Validate event flows
   - Test real workflows

3. **Performance Tests** - Performance benchmarks
   - Measure throughput and latency
   - Test scalability
   - Detect performance regressions
   - Generate performance reports

4. **Contract Tests** - Interface compliance
   - Validate plugin interfaces
   - Ensure connector compliance
   - Test API contracts

### Running Tests

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive testing instructions.

Quick commands:
```bash
# Unit tests
make unit
./run_tests.sh unit

# Integration tests
make integration
./run_tests.sh integration

# Performance tests
make performance
./run_tests.sh performance

# All tests
make all
./run_tests.sh all

# Quick tests (no coverage)
make quick
./run_tests.sh quick

# Watch mode
make watch
./run_tests.sh watch
```

## Configuration

POLARIS uses hierarchical YAML configuration with environment variable substitution:

```yaml
# config/swim_system_config.yaml
framework:
  event_bus:
    type: "nats"
    nats_config:
      servers: ["nats://localhost:4222"]

managed_systems:
  swim:
    connection:
      host: "localhost"
      port: 4242

llm:
  provider: "google"
  api_key: "${GOOGLE_AI_API_KEY}"
  model_name: "gemini-1.5-pro"
```

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov pytest-asyncio pytest-watch pytest-xdist

# Or use make
make install

# Verify setup
python verify_tests.py
```

### Running Tests During Development

```bash
# Quick feedback loop
make quick

# Watch mode (auto-rerun on changes)
make watch

# Run specific tests
pytest tests/unit/framework/test_configuration.py -v

# Debug mode
pytest tests/unit -v -s --pdb
```

### Code Quality

```bash
# Run tests with coverage
make unit

# Generate HTML coverage report
make coverage-html
# Open htmlcov/index.html

# Run all tests
make all
```

## CI/CD Integration

POLARIS is designed for easy CI/CD integration:

```yaml
# GitHub Actions example
- name: Run tests
  run: |
    cd polaris_refactored
    make ci-unit
    make ci-integration

# GitLab CI example
test:
  script:
    - cd polaris_refactored
    - make ci-all
```

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for complete CI/CD examples.

## Documentation

- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Comprehensive testing documentation
- **[TESTING_QUICKREF.md](TESTING_QUICKREF.md)** - Quick command reference
- **[TEST_SETUP_SUMMARY.md](TEST_SETUP_SUMMARY.md)** - Test infrastructure overview
- **[README_SWIM_SYSTEM.md](README_SWIM_SYSTEM.md)** - SWIM system setup guide
- **[tests/README.md](tests/README.md)** - Test infrastructure details

## Examples

### Running POLARIS with SWIM

See [README_SWIM_SYSTEM.md](README_SWIM_SYSTEM.md) for detailed instructions.

```bash
# Set up environment
export GOOGLE_AI_API_KEY='your-api-key'
python setup_environment.py

# Run POLARIS with SWIM
python run_swim_system.py
```

### Running Tests

```bash
# Quick verification
./run_tests.sh quick

# Full test suite
./run_tests.sh all

# Specific test category
./run_tests.sh unit
./run_tests.sh integration
./run_tests.sh performance
```

## Troubleshooting

### Test Issues

```bash
# Verify test setup
python verify_tests.py

# Check test discovery
pytest --collect-only tests/unit

# Activate virtual environment
source .venv/bin/activate  # Linux/WSL
.venv\Scripts\activate     # Windows
```

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for detailed troubleshooting.

### System Issues

See [README_SWIM_SYSTEM.md](README_SWIM_SYSTEM.md) for SWIM-specific troubleshooting.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass: `make all`
5. Ensure coverage is maintained: `make coverage`
6. Submit a pull request

## License

[Add your license information here]

## Support

For questions or issues:
1. Check the documentation in the `doc/` directory
2. Review the testing guides
3. Run verification script: `python verify_tests.py`
4. Check the logs in `logs/polaris.log`
5. Open an issue in the repository

## Acknowledgments

POLARIS builds upon research in self-adaptive systems, autonomic computing, and AI-powered system management.
