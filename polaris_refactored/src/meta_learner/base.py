"""Base Meta Learner Interface.

Abstract interface for strategic meta-learning components.
"""

from __future__ import annotations

import abc
import copy
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import yaml

from .models import (
    ContextWindow,
    FocusArea,
    GovernanceReport,
    MetaLearnerConfigurationError,
    ParameterProposal,
    WorldModelAlignment,
)

if TYPE_CHECKING:
    from digital_twin.knowledge_base import PolarisKnowledgeBase
else:
    PolarisKnowledgeBase = Any


class BaseMetaLearner(abc.ABC):
    """Abstract interface for strategic meta-learning components.

    The interface offers configuration lifecycle helpers plus a lightweight
    knowledge-base context collector so derived implementations can focus on
    higher-level reasoning tasks (policy calibration, utility tuning, etc.).
    """

    def __init__(
        self,
        component_id: str,
        config_path: str,
        knowledge_base: Optional[PolarisKnowledgeBase] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.component_id = component_id
        self.config_path = Path(config_path)
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.knowledge_base = knowledge_base
        self._config_cache: Dict[str, Any] = {}

        self._load_config()

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _load_config(self) -> None:
        if not self.config_path.exists():
            raise MetaLearnerConfigurationError(
                f"Meta-Learner config file not found: {self.config_path}"
            )

        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                self._config_cache = yaml.safe_load(f) or {}
        except Exception as exc:
            raise MetaLearnerConfigurationError(
                f"Failed to load meta-learner config: {exc}"
            ) from exc

    def reload_config(self) -> Dict[str, Any]:
        """Reload configuration from disk, returning the new snapshot."""
        self._load_config()
        return copy.deepcopy(self._config_cache)

    def get_config(self) -> Dict[str, Any]:
        """Return an immutable copy of the cached configuration."""
        return copy.deepcopy(self._config_cache)

    def update_config_value(self, path: str, value: Any) -> None:
        """Update a nested config value using dot-notation (e.g. "limits.cpu")."""
        if not path:
            raise ValueError("path must be provided")

        parts = path.split(".")
        node = self._config_cache
        for key in parts[:-1]:
            node = node.setdefault(key, {})
        node[parts[-1]] = value
        self._persist_config()

    def apply_bulk_updates(self, updates: Dict[str, Any]) -> None:
        """Apply multiple dot-notation updates in a single call."""
        for k, v in updates.items():
            self.update_config_value(k, v)

    def _persist_config(self) -> None:
        try:
            with self.config_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(self._config_cache, f, sort_keys=False)
        except Exception as exc:
            raise MetaLearnerConfigurationError(
                f"Failed to persist meta-learner config: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Knowledge base helpers
    # ------------------------------------------------------------------
    async def gather_context_snapshot(
        self,
        window: ContextWindow,
    ) -> Dict[str, Any]:
        """Collect historical context from the knowledge base.

        Returns a normalized payload that downstream reasoning strategies can
        consume. Derived classes can extend/transform the snapshot as needed.
        """

        if not self.knowledge_base:
            raise RuntimeError("Knowledge base dependency is not configured")

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=window.time_window_hours)
        states = await self.knowledge_base.get_historical_states(
            window.system_id, start_time, end_time
        )

        limited_states = states[-window.limit_states :]
        metrics_summary: Dict[str, Dict[str, float]] = {}
        for st in limited_states:
            for metric_name, metric in (getattr(st, "metrics", {}) or {}).items():
                payload = metrics_summary.setdefault(
                    metric_name,
                    {"count": 0, "sum": 0.0},
                )
                try:
                    payload["sum"] += float(getattr(metric, "value", metric))
                    payload["count"] += 1
                except (TypeError, ValueError):
                    continue

        averages = {
            name: summary["sum"] / summary["count"]
            for name, summary in metrics_summary.items()
            if summary["count"]
        }

        return {
            "component_id": self.component_id,
            "system_id": window.system_id,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "samples": len(limited_states),
            "metric_averages": averages,
            "focus_metrics": window.focus_metrics,
            "metadata": window.metadata,
            "raw_states": limited_states,
        }

    # ------------------------------------------------------------------
    # Abstract hooks for the strategic workflow
    # ------------------------------------------------------------------
    @abc.abstractmethod
    async def select_focus_areas(self, snapshot: Dict[str, Any]) -> List[FocusArea]:
        """Identify adaptation domains that require attention."""

    @abc.abstractmethod
    async def evaluate_world_model_alignment(
        self, snapshot: Dict[str, Any]
    ) -> WorldModelAlignment:
        """Assess whether the digital twin remains trustworthy."""

    @abc.abstractmethod
    async def propose_parameter_updates(
        self, snapshot: Dict[str, Any]
    ) -> List[ParameterProposal]:
        """Return structured proposals for downstream controllers."""

    @abc.abstractmethod
    async def validate_and_rank_updates(
        self, proposals: List[ParameterProposal]
    ) -> List[ParameterProposal]:
        """Gate the proposed updates before they reach execution layers."""

    @abc.abstractmethod
    async def emit_governance_report(
        self,
        snapshot: Dict[str, Any],
        final_updates: List[ParameterProposal],
    ) -> GovernanceReport:
        """Produce a summary that can be stored in the knowledge base."""
