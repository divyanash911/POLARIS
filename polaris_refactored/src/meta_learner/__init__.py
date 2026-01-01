"""Meta-Learner Interface Layer.

This module provides the Meta-Learner component for the POLARIS framework.
The Meta Learner is responsible for:

- Analyzing adaptation patterns and outcomes
- Proposing parameter adjustments for the adaptive controller
- Evaluating world model alignment with actual system behavior
- Generating governance reports on system adaptation

Components:
- BaseMetaLearner: Abstract interface for meta learning implementations
- LLMMetaLearner: Default implementation using LLM APIs
- ContextWindow: Parameters for knowledge gathering
- Various dataclasses for proposals, reports, and assessments
"""

from __future__ import annotations

# Import core models
from .models import (
    ContextWindow,
    FocusArea,
    FocusAreaPriority,
    GovernanceReport,
    MetaLearnerConfigurationError,
    ParameterProposal,
    ProposalStatus,
    WorldModelAlignment,
)

# Import base class
from .base import BaseMetaLearner

# Import LLM implementation
from .llm_meta_learner import LLMMetaLearner


__all__ = [
    # Base class
    "BaseMetaLearner",
    # LLM implementation
    "LLMMetaLearner",
    # Models
    "ContextWindow",
    "FocusArea",
    "FocusAreaPriority",
    "GovernanceReport",
    "MetaLearnerConfigurationError",
    "ParameterProposal",
    "ProposalStatus",
    "WorldModelAlignment",
]
