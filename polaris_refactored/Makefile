# POLARIS Test Makefile
# Provides convenient shortcuts for running tests

.PHONY: help test unit integration performance all coverage quick watch clean install

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "POLARIS Test Runner - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make unit              # Run unit tests"
	@echo "  make integration       # Run integration tests"
	@echo "  make all               # Run all tests"
	@echo "  make quick             # Quick unit test run"

test: unit ## Run unit tests (default)

unit: ## Run unit tests with coverage
	@echo "Running unit tests..."
	python -m tests.run_tests --unit

integration: ## Run integration tests
	@echo "Running integration tests..."
	python -m tests.run_tests --integration

performance: ## Run performance tests
	@echo "Running performance tests..."
	python -m tests.run_tests --performance

all: ## Run all test suites
	@echo "Running all tests..."
	python -m tests.run_tests --all

coverage: ## Run coverage analysis
	@echo "Running coverage analysis..."
	python -m tests.run_tests --coverage

quick: ## Run unit tests without coverage (fast)
	@echo "Running quick unit tests..."
	python -m tests.run_tests --unit --no-coverage

watch: ## Run tests in watch mode
	@echo "Running tests in watch mode..."
	python -m tests.run_tests --watch

clean: ## Clean test artifacts
	@echo "Cleaning test artifacts..."
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.json
	rm -rf coverage_report.txt
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

install: ## Install test dependencies
	@echo "Installing test dependencies..."
	pip install pytest pytest-cov pytest-asyncio pytest-watch pytest-xdist

# Advanced targets
unit-parallel: ## Run unit tests in parallel
	@echo "Running unit tests in parallel..."
	pytest tests/unit -n auto -v

unit-debug: ## Run unit tests with debugging
	@echo "Running unit tests with debugging..."
	pytest tests/unit -v -s --pdb

unit-verbose: ## Run unit tests with verbose output
	@echo "Running unit tests with verbose output..."
	pytest tests/unit -vv --tb=long

integration-verbose: ## Run integration tests with verbose output
	@echo "Running integration tests with verbose output..."
	pytest tests/integration --integration -vv --tb=long

# Coverage targets
coverage-html: ## Generate HTML coverage report
	@echo "Generating HTML coverage report..."
	pytest tests/unit --cov=src --cov-report=html -v
	@echo "Coverage report generated in htmlcov/index.html"

coverage-xml: ## Generate XML coverage report
	@echo "Generating XML coverage report..."
	pytest tests/unit --cov=src --cov-report=xml -v

# Specific test targets
test-framework: ## Run framework tests only
	@echo "Running framework tests..."
	pytest tests/unit/framework -v

test-adapters: ## Run adapter tests only
	@echo "Running adapter tests..."
	pytest tests/unit/adapters -v

test-digital-twin: ## Run digital twin tests only
	@echo "Running digital twin tests..."
	pytest tests/unit/digital_twin -v

test-control: ## Run control reasoning tests only
	@echo "Running control reasoning tests..."
	pytest tests/unit/control_reasoning -v

# CI targets
ci-unit: ## Run unit tests for CI
	@echo "Running unit tests for CI..."
	pytest tests/unit --cov=src --cov-report=xml --cov-report=term -v

ci-integration: ## Run integration tests for CI
	@echo "Running integration tests for CI..."
	pytest tests/integration --integration -v

ci-all: ci-unit ci-integration ## Run all tests for CI
