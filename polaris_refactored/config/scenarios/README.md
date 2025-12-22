# POLARIS Mock System Test Scenario Configurations

This directory contains POLARIS configuration files for different test scenarios used to validate the framework's behavior with the mock external system.

## Available Scenarios

### 1. Normal Operation (`normal_operation_config.yaml`)
**Purpose**: Test stable system operation with no expected adaptations.

**Characteristics**:
- Conservative thresholds to avoid unnecessary adaptations
- Stable baseline metrics
- Minimal monitoring frequency
- Expected adaptations: 0

**Success Criteria**:
- No adaptations triggered
- All metrics remain stable
- No errors in logs
- System maintains baseline performance

### 2. High Load (`high_load_config.yaml`)
**Purpose**: Test scale-up adaptations under high system load.

**Characteristics**:
- Aggressive thresholds to trigger scale-up adaptations
- Frequent monitoring (3-second intervals)
- Load simulation from 30% to 90%
- Expected adaptations: 3-5

**Success Criteria**:
- Scale-up adaptations triggered when CPU > 70%
- Response time optimization triggered when latency > 300ms
- System performance improves after adaptations
- Caching enabled under high connection load

### 3. Resource Constraint (`resource_constraint_config.yaml`)
**Purpose**: Test optimization adaptations under resource constraints.

**Characteristics**:
- Focus on efficiency and optimization
- Scale-up disabled to simulate constraints
- Aggressive optimization thresholds
- Expected adaptations: 4-6

**Success Criteria**:
- Scale-down triggered when CPU < 40%
- Configuration optimization triggered when response time > 200ms
- Caching enabled to improve efficiency
- QoS adjustments made under memory pressure
- No scale-up adaptations (resource constrained)

### 4. Failure Recovery (`failure_recovery_config.yaml`)
**Purpose**: Test detection and recovery from system failures.

**Characteristics**:
- Very frequent monitoring (2-second intervals)
- Aggressive failure detection thresholds
- Multiple failure types simulated
- Expected adaptations: 5-8

**Success Criteria**:
- All simulated failures detected within 30 seconds
- Recovery actions triggered for each failure type
- Service restarts executed for high error rates
- System returns to stable state after each failure
- No cascading failures occur

### 5. Mixed Workload (`mixed_workload_config.yaml`)
**Purpose**: Test handling multiple concurrent adaptation needs.

**Characteristics**:
- All adaptation types enabled
- Balanced thresholds for realistic scenarios
- Multiple concurrent adaptations allowed
- Expected adaptations: 8-12

**Success Criteria**:
- Multiple adaptation types triggered concurrently
- Scale-up and scale-down adaptations both occur
- System handles concurrent adaptations without conflicts
- All adaptation priorities respected
- System recovers to stable state at end

## Usage

### Running a Scenario

1. Start the mock external system:
   ```bash
   cd polaris_refactored/mock_external_system
   python scripts/start_mock_system.py
   ```

2. Run POLARIS with the desired scenario configuration:
   ```bash
   cd polaris_refactored
   python -m src.framework.polaris_framework --config config/scenarios/normal_operation_config.yaml
   ```

### Configuration Structure

Each scenario configuration:
- Extends the base `mock_system_config.yaml`
- Overrides specific settings for the scenario
- Includes scenario metadata with success criteria
- Defines expected behavior and adaptations

### Scenario Metadata

Each configuration includes a `scenario` section with:
- `name`: Scenario identifier
- `description`: Purpose and goals
- `expected_adaptations`: Number of adaptations expected
- `duration_minutes`: Recommended test duration
- `success_criteria`: List of conditions for successful test

### Threshold Configuration

Scenarios use different threshold strategies:
- **Conservative**: Higher thresholds to avoid adaptations (normal operation)
- **Aggressive**: Lower thresholds to trigger adaptations easily (high load, failure recovery)
- **Balanced**: Moderate thresholds for realistic scenarios (mixed workload)
- **Optimization-focused**: Thresholds that favor efficiency over scaling (resource constraint)

## Monitoring and Observability

Each scenario configures appropriate observability settings:
- **Collection intervals**: Adjusted based on scenario needs
- **Logging levels**: DEBUG for failure analysis, INFO for normal operation
- **Tracing**: High sampling rates for detailed scenarios
- **Metrics**: Enhanced collection for performance analysis

## Extending Scenarios

To create new scenarios:
1. Copy an existing scenario configuration
2. Modify thresholds and rules as needed
3. Update scenario metadata
4. Add appropriate success criteria
5. Test and validate the scenario

## Integration with Test Framework

These configurations are designed to work with:
- Mock external system test harness
- POLARIS performance test suite
- Automated test execution scripts
- Test result validation tools