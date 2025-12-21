# POLARIS Testing Quick Reference

## Quick Commands

### Using Scripts (Easiest)

```bash
# Linux/WSL
./run_tests.sh unit          # Unit tests
./run_tests.sh integration   # Integration tests
./run_tests.sh all           # All tests
./run_tests.sh quick         # Fast unit tests

# Windows
run_tests.bat unit           # Unit tests
run_tests.bat integration    # Integration tests
run_tests.bat all            # All tests
run_tests.bat quick          # Fast unit tests
```

### Using Make (Recommended)

```bash
make unit                    # Unit tests with coverage
make integration             # Integration tests
make performance             # Performance tests
make all                     # All test suites
make quick                   # Fast unit tests (no coverage)
make coverage                # Coverage analysis
make watch                   # Watch mode
make clean                   # Clean test artifacts
```

### Using Python

```bash
python -m tests.run_tests --unit              # Unit tests
python -m tests.run_tests --integration       # Integration tests
python -m tests.run_tests --performance       # Performance tests
python -m tests.run_tests --all               # All tests
python -m tests.run_tests --coverage          # Coverage analysis
python -m tests.run_tests --unit --no-coverage  # Fast unit tests
```

### Using pytest Directly

```bash
pytest tests/unit -v                          # All unit tests
pytest tests/integration --integration -v     # All integration tests
pytest tests/performance --performance -v     # All performance tests
pytest tests/unit/framework -v                # Specific directory
pytest tests/unit -k "configuration" -v       # Pattern matching
pytest tests/unit -x -v                       # Stop on first failure
```

## Common Patterns

### Run Specific Tests

```bash
# Specific file
pytest tests/unit/framework/test_configuration.py -v

# Specific test function
pytest tests/unit/framework/test_configuration.py::TestConfiguration::test_basic_configuration -v

# Pattern matching
pytest tests/unit -k "configuration and not hot_reload" -v

# Multiple files
pytest tests/unit/framework/test_configuration.py tests/unit/framework/test_events.py -v
```

### Coverage

```bash
# Unit tests with coverage
make unit

# HTML coverage report
make coverage-html
# Open htmlcov/index.html

# Coverage with custom threshold
python -m tests.run_tests --unit --threshold 85.0
```

### Debugging

```bash
# Show print statements
pytest tests/unit -v -s

# Drop into debugger on failure
pytest tests/unit -v --pdb

# Verbose output
pytest tests/unit -vv --tb=long

# Show local variables
pytest tests/unit -v -l
```

### Parallel Execution

```bash
# Install pytest-xdist first
pip install pytest-xdist

# Run in parallel
pytest tests/unit -n auto -v
make unit-parallel
```

### Watch Mode

```bash
# Install pytest-watch first
pip install pytest-watch

# Run in watch mode
make watch
python -m tests.run_tests --watch
```

## Test Markers

```bash
# Run only unit tests
pytest -m unit -v

# Run only integration tests
pytest -m integration --integration -v

# Run only performance tests
pytest -m performance --performance -v

# Exclude slow tests
pytest -m "not slow" -v

# Multiple markers
pytest -m "unit or integration" --integration -v
```

## Test Organization

```
tests/
├── unit/                # Fast, isolated tests
├── integration/         # Component interaction tests
├── performance/         # Performance benchmarks
├── e2e/                # End-to-end tests
├── fixtures/           # Test fixtures and mocks
└── utils/              # Test utilities
```

## Troubleshooting

```bash
# Verify test discovery
pytest --collect-only tests/unit

# Check pytest configuration
cat pytest.ini

# Activate virtual environment
source ../.venv/bin/activate  # Linux/WSL
..\\.venv\\Scripts\\activate   # Windows

# Install test dependencies
make install
pip install pytest pytest-cov pytest-asyncio
```

## CI/CD Integration

```yaml
# GitHub Actions example
- name: Run tests
  run: |
    make unit
    make integration
```

## Performance

```bash
# Quick performance check
pytest tests/performance -m "performance and not slow" --performance -v

# Full performance suite
make performance

# Show test durations
pytest tests/unit -v --durations=10
```

## Tips

1. Use `make quick` for rapid feedback during development
2. Use `make watch` to automatically re-run tests on file changes
3. Use `pytest -x` to stop on first failure
4. Use `pytest -k "pattern"` to run specific tests
5. Use `make coverage-html` to see detailed coverage reports
6. Run integration tests before committing
7. Run performance tests periodically to catch regressions

## Help

```bash
# Show available make targets
make help

# Show pytest help
pytest --help

# Show test runner help
python -m tests.run_tests --help
```

For detailed documentation, see [TESTING_GUIDE.md](TESTING_GUIDE.md)
