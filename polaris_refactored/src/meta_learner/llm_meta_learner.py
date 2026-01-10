"""LLM-Based Meta Learner Implementation.

Default implementation of the BaseMetaLearner interface using LLM APIs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseMetaLearner
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
from . import prompts

# Observability imports
try:
    from infrastructure.observability import (
        get_control_logger, get_metrics_collector, get_tracer,
        trace_polaris_method, MetricType
    )
    HAS_OBSERVABILITY = True
except ImportError:
    HAS_OBSERVABILITY = False
    get_control_logger = None
    get_metrics_collector = None
    get_tracer = None
    MetricType = None

if TYPE_CHECKING:
    from digital_twin.knowledge_base import PolarisKnowledgeBase
    from infrastructure.llm.client import LLMClient
    from control_reasoning.adaptive_controller import PolarisAdaptiveController
else:
    PolarisKnowledgeBase = Any
    LLMClient = Any
    PolarisAdaptiveController = Any


# Path to the controller's runtime config file
CONTROLLER_RUNTIME_CONFIG_PATH = (
    Path(__file__).parent.parent.parent / "config" / "adaptive_controller_runtime.yaml"
)


class LLMMetaLearner(BaseMetaLearner):
    """LLM-based meta learner implementation.
    
    Uses an LLM client to perform intelligent analysis of system behavior
    and propose parameter updates for the adaptive controller.
    
    The meta learner can operate in two modes:
    1. Continuous: Runs periodic analysis in the background
    2. On-demand: Triggered via CLI or programmatic calls
    """

    def __init__(
        self,
        component_id: str,
        config_path: str,
        llm_client: Optional[LLMClient] = None,
        knowledge_base: Optional[PolarisKnowledgeBase] = None,
        adaptive_controller: Optional[PolarisAdaptiveController] = None,
        controller_config_path: Optional[Path] = None,
        managed_system_ids: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the LLM Meta Learner.
        
        Args:
            component_id: Unique identifier for this meta learner instance.
            config_path: Path to the meta learner's own configuration file.
            llm_client: LLM client for reasoning (e.g., GoogleClient).
            knowledge_base: POLARIS knowledge base for historical data.
            adaptive_controller: Reference to the adaptive controller.
            controller_config_path: Path to controller runtime config YAML.
            managed_system_ids: List of managed system IDs to analyze.
            logger: Optional logger instance.
        """
        super().__init__(
            component_id=component_id,
            config_path=config_path,
            knowledge_base=knowledge_base,
            logger=logger,
        )
        
        self.llm_client = llm_client
        self.adaptive_controller = adaptive_controller
        self.controller_config_path = controller_config_path or CONTROLLER_RUNTIME_CONFIG_PATH
        self._managed_system_ids = managed_system_ids or []
        
        # Cache for recent analysis results
        self._last_analysis_time: Optional[datetime] = None
        self._last_focus_areas: List[FocusArea] = []
        self._last_alignment: Optional[WorldModelAlignment] = None
        
        # Background task management
        self._is_running = False
        self._analysis_task: Optional[asyncio.Task] = None
        
        # Initialize observability
        self._init_observability()

    def set_managed_systems(self, system_ids: List[str]) -> None:
        """Update the list of managed systems to analyze.
        
        Args:
            system_ids: List of system IDs to analyze in continuous mode.
        """
        self._managed_system_ids = system_ids.copy()
        self.logger.info(f"Updated managed systems for analysis: {system_ids}")

    def get_managed_systems(self) -> List[str]:
        """Get the current list of managed systems being analyzed."""
        return self._managed_system_ids.copy()

    async def start(self) -> None:
        """Start the meta learner."""
        self.logger.info("Starting meta learner", extra={
            "component": self.component_id,
            "managed_systems": self._managed_system_ids
        })
        self._record_metric("meta_learner.start_count", 1)
        self._is_running = True
        
        # Start continuous analysis loop if enabled
        continuous_config = self._config_cache.get("continuous", {})
        if continuous_config.get("enabled", False):
            if not self._managed_system_ids:
                self.logger.warning(
                    "Continuous analysis enabled but no managed systems configured. "
                    "Call set_managed_systems() to configure systems for analysis."
                )
            self._analysis_task = asyncio.create_task(self._run_continuous_analysis_loop())
            interval = continuous_config.get("analysis_interval_seconds", 300)
            self.logger.info(
                "Continuous analysis loop started",
                extra={
                    "interval_seconds": interval, 
                    "auto_apply": continuous_config.get("auto_apply_changes", False),
                    "systems_count": len(self._managed_system_ids)
                }
            )
        
    async def stop(self) -> None:
        """Stop the meta learner."""
        self.logger.info("Stopping meta learner...")
        self._is_running = False
        
        # Stop analysis task
        if self._analysis_task and not self._analysis_task.done():
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
            self.logger.info("Continuous analysis loop stopped")

    async def _run_continuous_analysis_loop(self) -> None:
        """Background task for continuous analysis."""
        continuous_config = self._config_cache.get("continuous", {})
        interval = continuous_config.get("analysis_interval_seconds", 300)
        auto_apply = continuous_config.get("auto_apply_changes", False)
        
        self.logger.info(f"Analysis loop running every {interval}s (auto-apply: {auto_apply})")
        
        while self._is_running:
            try:
                # Wait for interval
                await asyncio.sleep(interval)
                
                # Get systems to analyze
                systems_to_analyze = self._managed_system_ids.copy()
                
                # If no managed systems configured, try to get from knowledge base
                if not systems_to_analyze and self.knowledge_base:
                    try:
                        # Try to get systems that have recent telemetry data
                        # This is a fallback mechanism
                        self.logger.debug("No managed systems configured, attempting to discover from knowledge base")
                    except Exception as e:
                        self.logger.warning(f"Could not discover systems from knowledge base: {e}")
                
                if not systems_to_analyze:
                    self.logger.warning("No managed systems available for analysis, skipping cycle")
                    continue
                
                # Run analysis for each managed system
                for system_id in systems_to_analyze:
                    try:
                        self.logger.info(f"Running analysis cycle for system: {system_id}")
                        await self.run_analysis_cycle(system_id=system_id, apply_changes=auto_apply)
                    except Exception as e:
                        self.logger.error(f"Error analyzing system {system_id}: {e}")
                        # Continue with other systems
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(60)  # Wait a bit before retrying on error

        
    def _init_observability(self) -> None:
        """Initialize observability components."""
        # Use structured logger if available
        if HAS_OBSERVABILITY and get_control_logger:
            try:
                self.logger = get_control_logger(f"meta_learner.{self.component_id}")
            except Exception:
                pass  # Keep existing logger
        
        # Get metrics collector
        self._metrics = None
        if HAS_OBSERVABILITY and get_metrics_collector:
            try:
                self._metrics = get_metrics_collector()
            except Exception:
                pass
        
        # Get tracer
        self._tracer = None
        if HAS_OBSERVABILITY and get_tracer:
            try:
                self._tracer = get_tracer()
            except Exception:
                pass
    
    def _record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        if self._metrics:
            try:
                self._metrics.record_metric(name, value, labels or {})
            except Exception:
                pass  # Metrics are best-effort
    
    def _start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Start a tracing span."""
        if self._tracer:
            return self._tracer.start_span(name, attributes=attributes)
        return None
    
    def _ensure_llm_client(self) -> None:
        """Ensure LLM client is available."""
        if not self.llm_client:
            raise MetaLearnerConfigurationError(
                "LLM client is required for LLMMetaLearner operations"
            )
    
    async def _call_llm(self, prompt: str, system_prompt: str = prompts.SYSTEM_PROMPT) -> str:
        """Make an LLM API call and return the response text."""
        self._ensure_llm_client()
        
        from infrastructure.llm.models import LLMRequest, Message, MessageRole
        
        # Record metric for LLM calls
        self._record_metric("meta_learner.llm_calls", 1)
        start_time = datetime.now(timezone.utc)
        
        # Get model name from LLM client config or meta learner config
        model_name = "gemini-1.5-pro"  # Default
        if hasattr(self.llm_client, 'config') and hasattr(self.llm_client.config, 'model_name'):
            model_name = self.llm_client.config.model_name
        elif "llm" in self._config_cache:
            model_name = self._config_cache["llm"].get("model", model_name)
        
        self.logger.debug(
            "Calling LLM",
            extra={"model": model_name, "prompt_length": len(prompt)}
        )
        
        request = LLMRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(role=MessageRole.USER, content=prompt),
            ],
            model_name=model_name,
            max_tokens=2000,
            temperature=0.3,
        )
        
        try:
            response = await self.llm_client.generate_response(request)
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            self._record_metric("meta_learner.llm_duration_ms", duration_ms)
            self.logger.debug(
                "LLM response received",
                extra={"response_length": len(response.content), "duration_ms": round(duration_ms, 2)}
            )
            
            return response.content
        except Exception as e:
            self._record_metric("meta_learner.llm_errors", 1)
            self.logger.error(
                "LLM call failed",
                extra={"error": str(e), "model": model_name},
                exc_info=True
            )
            raise
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response."""
        # Try to find JSON block in response
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            self.logger.debug(f"Response was: {response}")
            return {}
    
    def _get_current_controller_config(self) -> Dict[str, Any]:
        """Load current controller runtime configuration."""
        try:
            import yaml
            if self.controller_config_path.exists():
                with open(self.controller_config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.warning(f"Could not load controller config: {e}")
        return {}
    
    async def _get_adaptation_history(self, system_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent adaptation history from knowledge base."""
        if not self.knowledge_base:
            return []
        
        try:
            history = await self.knowledge_base.get_adaptation_history(system_id)
            return history[:limit] if history else []
        except Exception as e:
            self.logger.warning(f"Could not get adaptation history: {e}")
            return []

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------
    
    async def select_focus_areas(self, snapshot: Dict[str, Any]) -> List[FocusArea]:
        """Identify adaptation domains that require attention using LLM."""
        self._ensure_llm_client()
        
        system_id = snapshot.get("system_id", "unknown")
        
        # Get adaptation history
        adaptation_history = await self._get_adaptation_history(system_id)
        
        # Format prompt
        prompt = prompts.SELECT_FOCUS_AREAS_PROMPT.format(
            system_id=system_id,
            time_range=snapshot.get("time_range", {}),
            samples=snapshot.get("samples", 0),
            metric_averages=json.dumps(snapshot.get("metric_averages", {}), indent=2),
            current_thresholds=json.dumps(self._get_current_controller_config().get("thresholds", {}), indent=2),
            adaptation_history=json.dumps(adaptation_history[:5], indent=2, default=str),
        )
        
        # Call LLM
        response = await self._call_llm(prompt)
        parsed = self._parse_json_response(response)
        
        # Convert to FocusArea objects
        focus_areas = []
        for area_data in parsed.get("focus_areas", []):
            try:
                priority_str = area_data.get("priority", "medium").lower()
                priority = FocusAreaPriority(priority_str) if priority_str in [p.value for p in FocusAreaPriority] else FocusAreaPriority.MEDIUM
                
                focus_areas.append(FocusArea(
                    name=area_data.get("name", "Unknown"),
                    priority=priority,
                    reason=area_data.get("reason", ""),
                    metrics=area_data.get("metrics", []),
                    suggested_action=area_data.get("suggested_action"),
                    confidence=float(area_data.get("confidence", 0.5)),
                ))
            except Exception as e:
                self.logger.warning(f"Failed to parse focus area: {e}")
        
        self._last_focus_areas = focus_areas
        return focus_areas
    
    async def evaluate_world_model_alignment(
        self, snapshot: Dict[str, Any]
    ) -> WorldModelAlignment:
        """Assess digital twin trustworthiness using LLM."""
        self._ensure_llm_client()
        
        # For now, create a basic alignment assessment
        # In full implementation, would compare predictions vs actuals
        prompt = prompts.EVALUATE_ALIGNMENT_PROMPT.format(
            system_id=snapshot.get("system_id", "unknown"),
            time_range=snapshot.get("time_range", {}),
            prediction_comparison="No prediction data available",
            prediction_accuracy="N/A",
        )
        
        response = await self._call_llm(prompt)
        parsed = self._parse_json_response(response)
        
        alignment = WorldModelAlignment(
            is_aligned=parsed.get("is_aligned", True),
            confidence=float(parsed.get("confidence", 0.5)),
            drift_detected=parsed.get("drift_detected", False),
            drift_severity=float(parsed.get("drift_severity", 0.0)),
            recommendations=parsed.get("recommendations", []),
            metrics_assessed=parsed.get("metrics_assessed", []),
        )
        
        self._last_alignment = alignment
        return alignment
    
    async def propose_parameter_updates(
        self, snapshot: Dict[str, Any]
    ) -> List[ParameterProposal]:
        """Generate parameter update proposals using LLM."""
        self._ensure_llm_client()
        
        system_id = snapshot.get("system_id", "unknown")
        adaptation_history = await self._get_adaptation_history(system_id)
        
        # Calculate success/failure counts
        successful = sum(1 for a in adaptation_history if a.get("success", False))
        failed = len(adaptation_history) - successful
        
        # Get constraints from meta learner config
        constraints = self._config_cache.get("meta_learner", {}).get("constraints", {})
        threshold_constraints = constraints.get("thresholds", {})
        cooldown_constraints = constraints.get("cooldowns", {})
        
        prompt = prompts.PROPOSE_UPDATES_PROMPT.format(
            focus_areas=json.dumps([{
                "name": fa.name,
                "priority": fa.priority.value,
                "reason": fa.reason,
            } for fa in self._last_focus_areas], indent=2),
            current_config=json.dumps(self._get_current_controller_config(), indent=2),
            total_adaptations=len(adaptation_history),
            successful_adaptations=successful,
            failed_adaptations=failed,
            threshold_min=threshold_constraints.get("min", 0.0),
            threshold_max=threshold_constraints.get("max", 100.0),
            cooldown_min=cooldown_constraints.get("min", 10.0),
            cooldown_max=cooldown_constraints.get("max", 3600.0),
        )
        
        response = await self._call_llm(prompt)
        parsed = self._parse_json_response(response)
        
        proposals = []
        for prop_data in parsed.get("proposals", []):
            try:
                proposals.append(ParameterProposal(
                    proposal_id=str(uuid.uuid4())[:8],
                    parameter_path=prop_data.get("parameter_path", ""),
                    current_value=prop_data.get("current_value"),
                    proposed_value=prop_data.get("proposed_value"),
                    rationale=prop_data.get("rationale", ""),
                    confidence=float(prop_data.get("confidence", 0.5)),
                    expected_impact=prop_data.get("expected_impact", ""),
                ))
            except Exception as e:
                self.logger.warning(f"Failed to parse proposal: {e}")
        
        return proposals
    
    async def validate_and_rank_updates(
        self, proposals: List[ParameterProposal]
    ) -> List[ParameterProposal]:
        """Validate and rank proposals using LLM."""
        if not proposals:
            return []
        
        self._ensure_llm_client()
        
        prompt = prompts.VALIDATE_UPDATES_PROMPT.format(
            proposals=json.dumps([{
                "proposal_id": p.proposal_id,
                "parameter_path": p.parameter_path,
                "current_value": p.current_value,
                "proposed_value": p.proposed_value,
                "rationale": p.rationale,
                "confidence": p.confidence,
            } for p in proposals], indent=2),
            current_state="System operational",
        )
        
        response = await self._call_llm(prompt)
        parsed = self._parse_json_response(response)
        
        # Update proposal statuses based on validation
        proposal_map = {p.proposal_id: p for p in proposals}
        validated = []
        
        for val_data in parsed.get("validated_proposals", []):
            proposal_id = val_data.get("proposal_id")
            if proposal_id in proposal_map:
                proposal = proposal_map[proposal_id]
                status_str = val_data.get("status", "pending")
                if status_str == "approved":
                    proposal.status = ProposalStatus.APPROVED
                elif status_str == "rejected":
                    proposal.status = ProposalStatus.REJECTED
                validated.append(proposal)
        
        # Sort by rank
        return sorted(validated, key=lambda p: p.confidence, reverse=True)
    
    async def emit_governance_report(
        self,
        snapshot: Dict[str, Any],
        final_updates: List[ParameterProposal],
    ) -> GovernanceReport:
        """Generate governance report using LLM."""
        self._ensure_llm_client()
        
        applied_changes = [p for p in final_updates if p.status == ProposalStatus.APPLIED]
        
        prompt = prompts.GOVERNANCE_REPORT_PROMPT.format(
            system_id=snapshot.get("system_id", "unknown"),
            analysis_time=datetime.now(timezone.utc).isoformat(),
            focus_area_count=len(self._last_focus_areas),
            proposals_generated=len(final_updates),
            proposals_applied=len(applied_changes),
            focus_areas=json.dumps([{
                "name": fa.name,
                "priority": fa.priority.value,
                "reason": fa.reason,
            } for fa in self._last_focus_areas], indent=2),
            alignment_assessment=json.dumps({
                "is_aligned": self._last_alignment.is_aligned if self._last_alignment else True,
                "confidence": self._last_alignment.confidence if self._last_alignment else 0.0,
                "drift_detected": self._last_alignment.drift_detected if self._last_alignment else False,
            }, indent=2),
            applied_changes=json.dumps([{
                "parameter": p.parameter_path,
                "old_value": p.current_value,
                "new_value": p.proposed_value,
            } for p in applied_changes], indent=2),
        )
        
        response = await self._call_llm(prompt)
        parsed = self._parse_json_response(response)
        
        report = GovernanceReport(
            report_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc),
            system_id=snapshot.get("system_id", "unknown"),
            focus_areas=self._last_focus_areas,
            alignment_assessment=self._last_alignment,
            proposals_generated=len(final_updates),
            proposals_applied=len(applied_changes),
            recommendations=parsed.get("recommendations", []),
            metadata={
                "summary": parsed.get("summary", ""),
                "key_findings": parsed.get("key_findings", []),
                "actions_taken": parsed.get("actions_taken", []),
                "next_review_hours": parsed.get("next_review_suggested_hours", 24),
            },
        )
        
        self._last_analysis_time = datetime.now(timezone.utc)
        return report
    
    # ------------------------------------------------------------------
    # Controller configuration update methods
    # ------------------------------------------------------------------
    
    async def apply_approved_proposals(
        self, proposals: List[ParameterProposal]
    ) -> List[ParameterProposal]:
        """Apply approved proposals to the controller configuration.
        
        Updates the controller configuration using the new evolvable threshold system.
        """
        approved = [p for p in proposals if p.status == ProposalStatus.APPROVED]
        if not approved:
            return []
        
        try:
            applied = []
            
            # Apply proposals directly to the adaptive controller if available
            if self.adaptive_controller:
                for proposal in approved:
                    try:
                        # Extract threshold name from parameter path
                        # e.g., "thresholds.cpu_high" -> "cpu_high"
                        parts = proposal.parameter_path.split(".")
                        if len(parts) >= 2 and parts[0] == "thresholds":
                            threshold_name = parts[1]
                            
                            # Use the controller's update_threshold method for evolvable thresholds
                            success = self.adaptive_controller.update_threshold(
                                threshold_name=threshold_name,
                                new_value=float(proposal.proposed_value),
                                updated_by=f"meta_learner:{self.component_id}",
                                reason=proposal.rationale[:100],  # Truncate for storage
                                confidence=proposal.confidence,
                                performance_impact=None  # Will be updated later based on monitoring
                            )
                            
                            if success:
                                proposal.status = ProposalStatus.APPLIED
                                proposal.applied_at = datetime.now(timezone.utc)
                                applied.append(proposal)
                                
                                self.logger.info(
                                    f"Applied evolvable threshold proposal {proposal.proposal_id}: "
                                    f"{threshold_name} = {proposal.proposed_value} "
                                    f"(confidence: {proposal.confidence:.2f})"
                                )
                            else:
                                proposal.status = ProposalStatus.FAILED
                                self.logger.warning(
                                    f"Failed to apply threshold proposal {proposal.proposal_id}: "
                                    f"Validation failed for {threshold_name} = {proposal.proposed_value}"
                                )
                        else:
                            # Handle non-threshold parameters using legacy method
                            success = await self._apply_legacy_proposal(proposal)
                            if success:
                                proposal.status = ProposalStatus.APPLIED
                                proposal.applied_at = datetime.now(timezone.utc)
                                applied.append(proposal)
                            else:
                                proposal.status = ProposalStatus.FAILED
                                
                    except Exception as e:
                        self.logger.error(f"Failed to apply proposal {proposal.proposal_id}: {e}")
                        proposal.status = ProposalStatus.FAILED
            else:
                # Fallback to legacy YAML file method if no controller reference
                applied = await self._apply_legacy_proposals(approved)
            
            if applied:
                self.logger.info(f"Applied {len(applied)} proposals to controller configuration")
                
                # Record metrics
                self._record_metric("meta_learner.proposals_applied_success", len(applied))
                for proposal in applied:
                    if proposal.parameter_path.startswith("thresholds."):
                        self._record_metric("meta_learner.threshold_evolutions", 1)
            
            return applied
            
        except Exception as e:
            self.logger.error(f"Failed to apply proposals: {e}")
            for p in approved:
                p.status = ProposalStatus.FAILED
            return []
    
    async def _apply_legacy_proposal(self, proposal: ParameterProposal) -> bool:
        """Apply a single non-threshold proposal using legacy method."""
        try:
            # Load current config
            config = self._get_current_controller_config()
            
            # Navigate to the nested key and update
            parts = proposal.parameter_path.split(".")
            node = config
            for key in parts[:-1]:
                node = node.setdefault(key, {})
            node[parts[-1]] = proposal.proposed_value
            
            # Update metadata
            config["last_updated"] = datetime.now(timezone.utc).isoformat()
            config["updated_by"] = f"meta_learner:{self.component_id}"
            
            # Write back to file
            import yaml
            with open(self.controller_config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, sort_keys=False)
            
            # Notify controller to reload if available
            if self.adaptive_controller:
                self.adaptive_controller.reload_config()
            
            self.logger.info(
                f"Applied legacy proposal: {proposal.parameter_path} = {proposal.proposed_value}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply legacy proposal: {e}")
            return False
    
    async def _apply_legacy_proposals(self, proposals: List[ParameterProposal]) -> List[ParameterProposal]:
        """Apply proposals using legacy YAML file method."""
        import yaml
        
        try:
            # Load current config
            config = self._get_current_controller_config()
            
            # Apply each approved proposal
            applied = []
            for proposal in proposals:
                try:
                    # Navigate to the nested key and update
                    parts = proposal.parameter_path.split(".")
                    node = config
                    for key in parts[:-1]:
                        node = node.setdefault(key, {})
                    node[parts[-1]] = proposal.proposed_value
                    
                    proposal.status = ProposalStatus.APPLIED
                    proposal.applied_at = datetime.now(timezone.utc)
                    applied.append(proposal)
                    
                    self.logger.info(
                        f"Applied legacy proposal {proposal.proposal_id}: "
                        f"{proposal.parameter_path} = {proposal.proposed_value}"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to apply proposal {proposal.proposal_id}: {e}")
                    proposal.status = ProposalStatus.FAILED
            
            # Update metadata
            config["last_updated"] = datetime.now(timezone.utc).isoformat()
            config["updated_by"] = f"meta_learner:{self.component_id}"
            
            # Write back to file
            with open(self.controller_config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, sort_keys=False)
            
            return applied
            
        except Exception as e:
            self.logger.error(f"Failed to apply legacy proposals: {e}")
            for p in proposals:
                p.status = ProposalStatus.FAILED
            return []
    
    # ------------------------------------------------------------------
    # Full analysis cycle
    # ------------------------------------------------------------------
    
    async def run_analysis_cycle(
        self, system_id: str, apply_changes: bool = False
    ) -> GovernanceReport:
        """Run a complete meta learning analysis cycle.
        
        Args:
            system_id: Target system to analyze.
            apply_changes: If True, apply approved proposals to config.
            
        Returns:
            GovernanceReport with analysis results.
        """
        cycle_start = datetime.now(timezone.utc)
        self._record_metric("meta_learner.analysis_cycles", 1)
        self.logger.info(
            "Starting meta learning cycle",
            extra={"system_id": system_id, "apply_changes": apply_changes}
        )
        
        try:
            # 1. Gather context
            window = ContextWindow(
                system_id=system_id,
                time_window_hours=self._config_cache.get("context_window_hours", 24.0),
                limit_states=self._config_cache.get("limit_states", 50),
            )
            snapshot = await self.gather_context_snapshot(window)
            
            # 2. Identify focus areas
            focus_areas = await self.select_focus_areas(snapshot)
            self._record_metric("meta_learner.focus_areas_identified", len(focus_areas))
            self.logger.info(
                "Identified focus areas",
                extra={"count": len(focus_areas), "areas": [fa.name for fa in focus_areas]}
            )
            
            # 3. Evaluate world model
            alignment = await self.evaluate_world_model_alignment(snapshot)
            self.logger.info(
                "World model alignment assessed",
                extra={"aligned": alignment.is_aligned, "confidence": alignment.confidence, "drift": alignment.drift_detected}
            )
            
            # 4. Generate proposals
            proposals = await self.propose_parameter_updates(snapshot)
            self._record_metric("meta_learner.proposals_generated", len(proposals))
            self.logger.info(
                "Generated parameter proposals",
                extra={"count": len(proposals)}
            )
            
            # 5. Validate and rank
            validated = await self.validate_and_rank_updates(proposals)
            approved_count = sum(1 for p in validated if p.status == ProposalStatus.APPROVED)
            self._record_metric("meta_learner.proposals_approved", approved_count)
            self.logger.info(
                "Validated proposals",
                extra={"total": len(validated), "approved": approved_count}
            )
            
            # 6. Apply if requested
            if apply_changes and approved_count > 0:
                applied = await self.apply_approved_proposals(validated)
                self._record_metric("meta_learner.proposals_applied", len(applied))
                self.logger.info(
                    "Applied approved proposals",
                    extra={"applied_count": len(applied)}
                )
            
            # 7. Generate report
            report = await self.emit_governance_report(snapshot, validated)
            
            cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
            self._record_metric("meta_learner.cycle_duration_seconds", cycle_duration)
            self.logger.info(
                "Meta learning cycle complete",
                extra={
                    "report_id": report.report_id,
                    "duration_seconds": round(cycle_duration, 2),
                    "proposals_applied": report.proposals_applied
                }
            )
            return report
            
        except Exception as e:
            self._record_metric("meta_learner.cycle_errors", 1)
            self.logger.error(
                "Meta learning cycle failed",
                extra={"system_id": system_id, "error": str(e)},
                exc_info=True
            )
            raise
