# Execution Adapter Configuration

The Execution Adapter executes adaptation actions on managed systems using a configurable pipeline of stages. This document describes its configuration options.

## Example Configuration

```yaml
# Example configuration for an Execution Adapter
adapter_id: "execution-adapter-1"
adapter_type: "execution"
enabled: true
config:
  pipeline_stages:
    - type: "validation"
    - type: "pre_condition"
      rules:
        - name: "check_system_health"
          condition: "system.health_status == 'HEALTHY'"
    - type: "action_execution"
    - type: "post_verification"
      rules:
        - name: "verify_action_applied"
          condition: "result.status == 'SUCCESS'"
  
  # Timeouts in seconds for each stage
  stage_timeouts:
    validation: 5.0
    pre_condition: 10.0
    action_execution: 30.0
    post_verification: 15.0
  
  # Managed systems configuration
  managed_systems:
    - system_id: "system-a"
      connector_type: "aws_ec2_connector"
      config:
        region: "us-west-2"
        max_retries: 3
    - system_id: "system-b"
      connector_type: "kubernetes_connector"
      config:
        context: "production-cluster"
        namespace: "default"
```

## Configuration Fields

### Top-level Fields

- `adapter_id` (string; required)
  - Unique identifier for this adapter instance.

- `adapter_type` (string; required)
  - Must be `"execution"` for this adapter type.

- `enabled` (boolean; default: `true`)
  - Whether this adapter instance is enabled.

### `config` Object

- `pipeline_stages` (array of objects; required)
  - Defines the sequence of stages in the execution pipeline.
  - Each object must have a `type` field (see "Stage Types" below).
  - Stages are executed in the order they appear.

- `stage_timeouts` (object; optional)
  - Timeout in seconds for each stage type.
  - If a stage exceeds its timeout, the pipeline fails with `TIMEOUT` status.
  - Supported keys:
    - `validation`: Timeout for validation stage (default: 5.0s)
    - `pre_condition`: Timeout for pre-condition checks (default: 10.0s)
    - `action_execution`: Timeout for action execution (default: 30.0s)
    - `post_verification`: Timeout for post-verification (default: 15.0s)

- `managed_systems` (array of objects; required)
  - List of systems this adapter can execute actions on.
  - Each object must have:
    - `system_id` (string): Unique identifier for the system
    - `connector_type` (string): Type of connector to use
    - `config` (object): Connector-specific configuration

## Stage Types

### 1. Validation Stage

Validates the action parameters and checks if the action is supported.

```yaml
- type: "validation"
```

**Behavior:**
- Verifies required action parameters are present
- Checks if the action type is supported by the target system's connector
- Validates parameter types and values

### 2. Pre-Condition Check Stage

Validates system state before executing the action.

```yaml
- type: "pre_condition"
  rules:
    - name: "check_system_health"
      condition: "system.health_status == 'HEALTHY'"
    - name: "check_resource_availability"
      condition: "system.metrics.available_cpu > 0.2"
```

**Configuration:**
- `rules` (array of objects): List of validation rules
  - `name` (string): Descriptive name for the rule
  - `condition` (string): Python expression that must evaluate to `True`
  - `error_message` (string; optional): Custom error message if condition fails

### 3. Action Execution Stage

Executes the action on the target system.

```yaml
- type: "action_execution"
  # Optional: Override default timeout
  timeout_seconds: 45.0
```

**Configuration:**
- `timeout_seconds` (float; optional): Override the default action execution timeout

### 4. Post-Execution Verification Stage

Verifies the action was successful by checking system state.

```yaml
- type: "post_verification"
  rules:
    - name: "verify_desired_state"
      condition: "system.state == 'RUNNING'"
    - name: "verify_no_errors"
      condition: "'error' not in system.logs"
```

**Configuration:**
- `rules` (array of objects): List of verification rules
  - `name` (string): Descriptive name for the rule
  - `condition` (string): Python expression that must evaluate to `True`
  - `error_message` (string; optional): Custom error message if condition fails

## Error Handling

The adapter handles errors at multiple levels:

1. **Stage Timeout**: If any stage exceeds its timeout, the pipeline fails with `TIMEOUT` status.
2. **Validation Failure**: If validation rules fail, the pipeline stops with `INVALID_ACTION` status.
3. **Pre-Condition Failure**: If pre-conditions aren't met, the pipeline stops with `PRECONDITION_FAILED` status.
4. **Execution Error**: If the action fails, the pipeline stops with `FAILED` status.
5. **Verification Failure**: If post-verification fails, the pipeline completes with `PARTIAL` status.

## Monitoring and Metrics

The adapter emits the following metrics:

- `execution_adapter_actions_total`: Counter of actions executed, labeled by status
- `execution_adapter_duration_seconds`: Histogram of execution times
- `execution_adapter_stage_duration_seconds`: Histogram of per-stage execution times

## Logging

Logs are emitted at various levels:

- `DEBUG`: Detailed execution flow
- `INFO`: Major lifecycle events
- `WARNING`: Non-fatal issues
- `ERROR`: Execution failures

## Example: Scaling Action

```yaml
adapter_id: "ec2-scaling-executor"
adapter_type: "execution"
config:
  pipeline_stages:
    - type: "validation"
    - type: "pre_condition"
      rules:
        - name: "check_autoscaling_enabled"
          condition: "system.autoscaling_enabled == true"
    - type: "action_execution"
      timeout_seconds: 60.0
    - type: "post_verification"
      rules:
        - name: "verify_desired_capacity"
          condition: "system.desired_capacity == target_desired_capacity"
  
  stage_timeouts:
    action_execution: 60.0
  
  managed_systems:
    - system_id: "production-asg"
      connector_type: "aws_autoscaling_connector"
      config:
        region: "us-west-2"
        auto_scaling_group_name: "prod-app"
```
