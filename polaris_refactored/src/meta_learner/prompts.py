"""Prompt Templates for Meta Learner.

Stores prompt templates for LLM-based meta learning operations.
"""

# System prompt for meta learner reasoning
SYSTEM_PROMPT = """You are a Meta Learner for the POLARIS self-adaptive systems framework.
Your role is to analyze system behavior, identify areas for improvement, and propose
parameter adjustments to enhance the adaptation system's effectiveness.

You have access to:
- Historical system states and metrics
- Current threshold and strategy configurations
- Adaptation history and outcomes

Your responses should be:
- Data-driven and evidence-based
- Conservative in proposing changes
- Clear about confidence levels
- Focused on measurable improvements"""


# Template for identifying focus areas
SELECT_FOCUS_AREAS_PROMPT = """Analyze the following system snapshot and identify areas requiring attention.

## System Context
- System ID: {system_id}
- Time Range: {time_range}
- Samples Analyzed: {samples}

## Metric Averages
{metric_averages}

## Current Thresholds
{current_thresholds}

## Recent Adaptation History
{adaptation_history}

## Task
Identify up to 5 focus areas that require attention. For each area, provide:
1. Name of the area (e.g., "CPU Threshold Calibration")
2. Priority (low/medium/high/critical)
3. Reason for attention
4. Related metrics
5. Suggested action

Respond in JSON format:
```json
{{
  "focus_areas": [
    {{
      "name": "string",
      "priority": "low|medium|high|critical",
      "reason": "string",
      "metrics": ["metric1", "metric2"],
      "suggested_action": "string",
      "confidence": 0.0-1.0
    }}
  ]
}}
```"""


# Template for evaluating world model alignment
EVALUATE_ALIGNMENT_PROMPT = """Evaluate whether the digital twin/world model accurately reflects the real system behavior.

## System Context
- System ID: {system_id}
- Time Range: {time_range}

## Predicted vs Actual Metrics
{prediction_comparison}

## Recent Prediction Accuracy
{prediction_accuracy}

## Task
Assess the alignment between the world model and actual system behavior:
1. Is the model aligned? (boolean)
2. Confidence in assessment (0.0-1.0)
3. Is drift detected?
4. Drift severity if detected (0.0-1.0)
5. Recommendations for improvement

Respond in JSON format:
```json
{{
  "is_aligned": true|false,
  "confidence": 0.0-1.0,
  "drift_detected": true|false,
  "drift_severity": 0.0-1.0,
  "recommendations": ["recommendation1", "recommendation2"],
  "metrics_assessed": ["metric1", "metric2"]
}}
```"""


# Template for proposing parameter updates
PROPOSE_UPDATES_PROMPT = """Based on the analysis, propose parameter updates for the adaptive controller.

## Focus Areas Identified
{focus_areas}

## Current Configuration
{current_config}

## Historical Performance
- Total adaptations: {total_adaptations}
- Successful adaptations: {successful_adaptations}
- Failed adaptations: {failed_adaptations}

## Constraints
- Thresholds must be between {threshold_min} and {threshold_max}
- Cooldowns must be between {cooldown_min} and {cooldown_max} seconds
- Changes should be incremental (max 20% change per proposal)

## Task
Propose specific parameter updates to improve system performance.
For each proposal, provide:
1. Parameter path (dot-notation, e.g., "thresholds.cpu_high")
2. Current value
3. Proposed value
4. Rationale for change
5. Expected impact
6. Confidence in proposal

Respond in JSON format:
```json
{{
  "proposals": [
    {{
      "parameter_path": "string",
      "current_value": number|boolean,
      "proposed_value": number|boolean,
      "rationale": "string",
      "expected_impact": "string",
      "confidence": 0.0-1.0
    }}
  ]
}}
```"""


# Template for validating and ranking updates
VALIDATE_UPDATES_PROMPT = """Review and rank the following parameter update proposals.

## Proposals to Review
{proposals}

## Current System State
{current_state}

## Risk Assessment Criteria
- Impact on system stability
- Reversibility of change
- Historical precedent
- Confidence in expected outcome

## Task
For each proposal:
1. Validate if it should proceed (approve/reject)
2. Provide validation reasoning
3. Assign risk score (0.0-1.0, lower is safer)
4. Rank by priority

Respond in JSON format:
```json
{{
  "validated_proposals": [
    {{
      "proposal_id": "string",
      "status": "approved|rejected",
      "validation_reason": "string",
      "risk_score": 0.0-1.0,
      "rank": 1
    }}
  ]
}}
```"""


# Template for governance report
GOVERNANCE_REPORT_PROMPT = """Generate a governance report summarizing the meta learning cycle.

## Cycle Summary
- System ID: {system_id}
- Analysis Time: {analysis_time}
- Focus Areas Identified: {focus_area_count}
- Proposals Generated: {proposals_generated}
- Proposals Applied: {proposals_applied}

## Focus Areas
{focus_areas}

## World Model Assessment
{alignment_assessment}

## Applied Changes
{applied_changes}

## Task
Generate a concise governance report with:
1. Executive summary (2-3 sentences)
2. Key findings (bullet points)
3. Actions taken
4. Recommendations for next cycle

Respond in JSON format:
```json
{{
  "summary": "string",
  "key_findings": ["finding1", "finding2"],
  "actions_taken": ["action1", "action2"],
  "recommendations": ["recommendation1", "recommendation2"],
  "next_review_suggested_hours": number
}}
```"""
