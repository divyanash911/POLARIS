# Meta-Learning Subsystem

The Meta-Learning subsystem is a high-level governance component in the POLARIS framework that enables autonomous self-improvement. It sits above the standard adaptive control loop, analyzing long-term system behavior to optimize the adaptive controller's own configuration.

## Overview

While the **Adaptive Controller** reacts to immediate system changes (seconds/minutes), the **Meta Learner** operates on a longer timescale (hours/days) to answer the question: *"Is the adaptive controller configured correctly for the current environment?"*

Key capabilities:
- **Autonomous Parameter Tuning**: Adjusts thresholds, cooldowns, and strategy weights based on historical performance.
- **Context-Aware Analysis**: Uses LLMs to analyze complex system telemetry and adaptation history.
- **Verification & Safety**: Validates proposed changes before applying them to prevent instability.
- **Governance**: Generates detailed reports explaining *why* changes were made.

## Architecture

The Meta Learner interacts with the following components:
- **LLM Client**: For reasoning and analysis (e.g., Google Gemini).
- **Polaris Knowledge Base**: To retrieve historical adaptation data and metrics.
- **Adaptive Controller**: To read current runtime config and apply updates.
- **World Model**: To assess alignment between predictions and reality.

### Workflow

The meta-learning cycle consists of 7 steps:

1. **Gather Context**: Collects system metrics, recent adaptation events, and current configuration.
2. **Identify Focus Areas**: Uses LLM to identify domains requiring attention (e.g., "Frequent oscillation in scaling actions").
3. **Evaluate Alignment**: Checks if the digital twin's world model aligns with observed reality.
4. **Propose Updates**: Generates specific parameter updates (e.g., "Increase CPU high threshold to 85%").
5. **Validate & Rank**: Reviews proposals for safety and expected impact.
6. **Apply Changes**: Automatically updates the `adaptive_controller_runtime.yaml` file (hot-reloaded by the controller).
7. **Emit Governance Report**: Records the analysis, decisions, and outcomes.

## Configuration

The meta learner is configured via `config/meta_learner_config.yaml`.

```yaml
version: 1.0
component_id: "polaris_meta_learner"

# Operation mode
mode: "hybrid"  # autonomous, advisory, or hybrid

# Context gathering settings
context:
  window_hours: 24
  limit_states: 100

# LLM Configuration
llm:
  provider: "google"
  model: "gemini-1.5-pro"
  temperature: 0.1

# Continuous analysis settings
continuous:
  enabled: true
  analysis_interval_seconds: 3600  # Run every hour
  auto_apply_changes: false        # Require manual approval (or true for autonomous)
```

## Usage

 The Meta Learner runs automatically as part of the POLARIS framework if enabled.

### Checking Status
Use the CLI to check status:
```bash
polaris> meta-learner
Meta Learner Status
==================================================
Status: Active (Background Analysis)
Use 'history' to see if any governance reports have been generated.
```

### Governance Reports
Reports are stored in the knowledge base and can be viewed via the history command.

## Runtime Updates
The Meta Learner modifies `config/adaptive_controller_runtime.yaml`. The Adaptive Controller watches this file and hot-reloads changes immediately without restart.

Example update:
```yaml
thresholds:
  cpu_high: 85.0  # Updated from 80.0 by Meta Learner
  memory_high: 90.0
last_updated: "2024-03-15T10:00:00Z"
updated_by: "meta_learner:polaris_meta_learner"
```
