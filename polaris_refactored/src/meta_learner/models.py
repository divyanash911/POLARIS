"""Meta Learner Models.

Dataclasses and exceptions for the Meta Learner component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class MetaLearnerConfigurationError(RuntimeError):
    """Raised when configuration load/save operations fail."""


class ProposalStatus(str, Enum):
    """Status of a parameter update proposal."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    FAILED = "failed"


class FocusAreaPriority(str, Enum):
    """Priority levels for focus areas."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ContextWindow:
    """Parameters describing the knowledge-gathering window."""

    system_id: str
    time_window_hours: float = 24.0
    focus_metrics: List[str] = field(default_factory=list)
    limit_states: int = 50
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FocusArea:
    """Represents an area requiring meta learner attention."""
    
    name: str
    priority: FocusAreaPriority
    reason: str
    metrics: List[str] = field(default_factory=list)
    suggested_action: Optional[str] = None
    confidence: float = 0.0


@dataclass
class WorldModelAlignment:
    """Assessment of digital twin trustworthiness."""
    
    is_aligned: bool
    confidence: float
    drift_detected: bool = False
    drift_severity: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    metrics_assessed: List[str] = field(default_factory=list)


@dataclass
class ParameterProposal:
    """Proposal for a parameter update."""
    
    proposal_id: str
    parameter_path: str  # Dot-notation path (e.g., "thresholds.cpu_high")
    current_value: Any
    proposed_value: Any
    rationale: str
    confidence: float
    expected_impact: str
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    applied_at: Optional[datetime] = None


@dataclass
class GovernanceReport:
    """Summary report from meta learner analysis."""
    
    report_id: str
    timestamp: datetime
    system_id: str
    focus_areas: List[FocusArea]
    alignment_assessment: Optional[WorldModelAlignment]
    proposals_generated: int
    proposals_applied: int
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
