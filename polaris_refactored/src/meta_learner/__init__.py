"""Meta-Learner Interface Layer.

This module defines a high-level interface for a future Meta-Learner service
that will run as its own container alongside the Digital Twin. The component is
responsible for:

- Owning its configuration file and exposing safe update helpers
- Gathering long-horizon context from the POLARIS Knowledge Base
- Providing abstract hooks for strategy evaluation, calibration, and parameter
  proposals at the adaptation-governance level

Only the abstract interface and shared utilities are defined here; concrete
implementations can live in separate packages or services.
"""

from __future__ import annotations

import abc
import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import yaml

if TYPE_CHECKING:  # Local import resolves when running inside the POLARIS runtime
    from digital_twin.knowledge_base import PolarisKnowledgeBase
else:  # pragma: no cover - keeps the interface importable elsewhere
    PolarisKnowledgeBase = Any  # type: ignore


@dataclass
class ContextWindow:
    """Parameters describing the knowledge-gathering window."""

    system_id: str
    time_window_hours: float = 24.0
    focus_metrics: List[str] = field(default_factory=list)
    limit_states: int = 50
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetaLearnerConfigurationError(RuntimeError):
    """Raised when configuration load/save operations fail."""


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
        except Exception as exc:  # pragma: no cover - simple IO guard
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
        except Exception as exc:  # pragma: no cover
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
    async def select_focus_areas(self, snapshot: Dict[str, Any]) -> List[str]:
        """Identify adaptation domains that require attention."""

    @abc.abstractmethod
    async def evaluate_world_model_alignment(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Assess whether the digital twin remains trustworthy."""

    @abc.abstractmethod
    async def propose_parameter_updates(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return structured proposals for downstream controllers."""

    @abc.abstractmethod
    async def validate_and_rank_updates(
        self, proposals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Gate the proposed updates before they reach execution layers."""

    @abc.abstractmethod
    async def emit_governance_report(
        self,
        snapshot: Dict[str, Any],
        final_updates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Produce a summary that can be stored in the knowledge base."""
