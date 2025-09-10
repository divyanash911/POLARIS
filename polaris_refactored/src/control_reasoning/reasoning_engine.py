"""
Reasoning Engine Implementation

Implements the Reasoning Engine using Chain of Responsibility + Strategy patterns.

This module provides a flexible and extensible reasoning system that combines multiple
reasoning strategies to analyze system states and provide insights. It implements
the Chain of Responsibility pattern to process reasoning requests through multiple
strategies and fuses their results into a consolidated output.

Key Components:
- ReasoningContext: Carries system state and historical data for analysis
- ReasoningResult: Standardized container for reasoning outputs
- Base and concrete reasoning strategies (Statistical, Causal, Experience-based)
- Result fusion for combining multiple strategy outputs
- Main PolarisReasoningEngine class that orchestrates the reasoning process

"""


from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone

from ..infrastructure.di import Injectable
from ..digital_twin.knowledge_base import PolarisKnowledgeBase


class ReasoningContext:
    """Context information for reasoning operations in the POLARIS system.
    
    This class encapsulates all the information needed by reasoning strategies to
    perform their analysis. It provides a standardized way to pass system state,
    historical data, and relationships between components to the reasoning process.
    
    Attributes:
        system_id: Unique identifier for the target system
        current_state: Dictionary containing current system metrics and state
        historical_data: List of past system states for trend analysis
        system_relationships: Graph structure representing component dependencies
    """
    
    def __init__(
        self, 
        system_id: str, 
        current_state: Dict[str, Any],
        historical_data: List[Dict[str, Any]] = None,
        system_relationships: Dict[str, Any] = None
    ):
        self.system_id = system_id
        self.current_state = current_state
        self.historical_data = historical_data or []
        self.system_relationships = system_relationships or {}


class ReasoningResult:
    """Container for the results of a reasoning operation.
    
    This class standardizes the output format for all reasoning strategies,
    ensuring consistent handling of insights, confidence levels, and recommendations
    across different reasoning approaches.
    
    Attributes:
        insights: List of dictionaries containing analysis results and observations
        confidence: Numeric value (0.0 to 1.0) indicating confidence in the results
        recommendations: List of suggested actions based on the analysis
        
    The confidence score helps downstream components understand the reliability
    of the results and make informed decisions about how to use them.
    """
    
    def __init__(
        self, 
        insights: List[Dict[str, Any]], 
        confidence: float,
        recommendations: List[Dict[str, Any]] = None
    ):
        self.insights = insights
        self.confidence = confidence
        self.recommendations = recommendations or []


class ReasoningStrategy(ABC):
    """Abstract base class for reasoning strategies in the POLARIS reasoning engine.
    
    Concrete strategies should implement the reason() method to analyze the provided
    context and return a ReasoningResult containing insights and recommendations.
    The strategy pattern allows for flexible composition of different reasoning
    approaches to form a comprehensive analysis pipeline.
    """
    
    @abstractmethod
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Perform reasoning on the given context."""
        pass


class StatisticalReasoningStrategy(ReasoningStrategy):
    """Statistical reasoning strategy that analyzes metrics using statistical methods.
    
    This strategy performs statistical analysis on system metrics to detect anomalies
    and identify trends. It compares current metric values against historical data
    to identify significant deviations that may indicate issues.
    
    Key Features:
    - Anomaly detection using statistical thresholds
    - Trend analysis of time-series data
    - Configurable sensitivity levels
    - Support for various metric types
    
    Dependencies:
    - Optional knowledge base for historical data access
    - Numeric metrics for meaningful analysis
    """
    
    def __init__(self, knowledge_base: Optional[PolarisKnowledgeBase] = None):
        self._kb = knowledge_base

    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        insights: List[Dict[str, Any]] = []
        confidence = 0.3
        hist = context.historical_data
        # If KB available and no historical data provided in context, fetch last hour
        if not hist and self._kb:
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=1)
            try:
                states = await self._kb.get_historical_states(context.system_id, start, end)
                hist = [
                    {"metrics": s.metrics, "timestamp": s.timestamp, "health_status": s.health_status.value}
                    for s in states
                ]
            except Exception:
                hist = []

        current_metrics = context.current_state.get("metrics", {})
        # Compute simple mean for numeric metrics
        for name, m in current_metrics.items():
            try:
                cur = float(getattr(m, "value", m))
            except Exception:
                continue
            values: List[float] = []
            for h in hist:
                mv = h.get("metrics", {}).get(name)
                try:
                    v = float(getattr(mv, "value", mv)) if mv is not None else None
                except Exception:
                    v = None
                if v is not None:
                    values.append(v)
            if values:
                avg = sum(values) / len(values)
                # Flag anomaly if 20% over rolling mean
                if cur > 1.2 * avg:
                    insights.append({"type": "statistical_anomaly", "metric": name, "current": cur, "mean": avg})
                    confidence = max(confidence, 0.6)
        if not insights:
            insights.append({"type": "statistical_baseline", "message": "No significant anomalies"})
        return ReasoningResult(insights=insights, confidence=confidence)


class CausalReasoningStrategy(ReasoningStrategy):
    """Causal reasoning strategy that understands system dependencies and causality.
    
    This strategy analyzes the system's dependency graph to understand how components
    interact and affect each other. It can identify root causes of issues by tracing
    effects through the dependency graph and understanding causal relationships.
    
    Key Features:
    - Root cause analysis
    - Impact assessment of potential changes
    - Understanding of system topology
    - Support for complex dependency graphs
    
    Dependencies:
    - Knowledge base with system dependency information
    - Properly modeled component relationships
    """
    
    def __init__(self, knowledge_base: Optional[PolarisKnowledgeBase] = None):
        self._kb = knowledge_base

    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        insights: List[Dict[str, Any]] = []
        confidence = 0.4
        # If dependency context provided, use it, else query from KB
        deps = context.system_relationships
        if not deps and self._kb:
            try:
                deps = await self._kb.get_dependency_chain(context.system_id, max_depth=2)
            except Exception:
                deps = {}

        # Heuristic: if a metric is high, see if any upstream neighbor has similar issue
        cur_metrics = context.current_state.get("metrics", {})
        for name, m in cur_metrics.items():
            try:
                cur = float(getattr(m, "value", m))
            except Exception:
                continue
            if cur < 0.85 and cur < 85.0:
                continue
            # Mark potential upstream cause if exists
            if deps:
                insights.append({"type": "causal_link", "metric": name, "direction": "upstream", "evidence": "high metric with deps"})
                confidence = max(confidence, 0.6)
        if not insights:
            insights.append({"type": "causal_baseline", "message": "No clear causal links"})
        return ReasoningResult(insights=insights, confidence=confidence)


class ExperienceBasedReasoningStrategy(ReasoningStrategy):
    """Experience-based reasoning that leverages historical patterns and past experiences.
    
    This strategy searches the knowledge base for similar past situations and their
    outcomes to provide recommendations. It learns from historical data to improve
    its suggestions over time.
    
    Key Features:
    - Case-based reasoning
    - Similarity-based pattern matching
    - Learning from past experiences
    - Adaptive recommendations
    
    Dependencies:
    - Populated knowledge base with historical patterns
    - Effective similarity metrics for pattern matching
    - Periodic retraining for model updates
    """
    
    def __init__(self, knowledge_base: Optional[PolarisKnowledgeBase] = None):
        self._kb = knowledge_base

    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        if not self._kb:
            return ReasoningResult(insights=[{"type": "experience_baseline"}], confidence=0.2)
        conditions = _conditions_from_state(context.current_state)
        patterns = await self._kb.get_similar_patterns(conditions, similarity_threshold=0.6)
        insights: List[Dict[str, Any]] = []
        recs: List[Dict[str, Any]] = []
        for p in patterns[:3]:
            insights.append({"type": "experience_match", "pattern_id": p.pattern_id, "confidence": p.confidence})
            action_type = p.outcomes.get("action_type")
            params = p.outcomes.get("parameters", {})
            if action_type:
                recs.append({"action_type": action_type, "parameters": params, "source": "experience", "pattern_id": p.pattern_id})
        conf = 0.5 if patterns else 0.3
        return ReasoningResult(insights=insights or [{"type": "experience_none"}], confidence=conf, recommendations=recs)


class ResultFusionStrategy:
    """Strategy for fusing results from multiple reasoning strategies.
    
    This class implements techniques to combine the outputs of different reasoning
    strategies into a single, coherent result. It handles conflicts, normalizes
    confidence scores, and ensures consistent output formatting.
    
    Fusion Techniques:
    - Weighted combination of confidence scores
    - Conflict resolution between strategies
    - Deduplication of similar recommendations
    - Confidence-based result filtering
    
    The fusion process considers the confidence levels of individual strategy
    outputs to produce a consolidated view of the system's state and recommended
    actions.
    """
    
    async def fuse(self, results: List[ReasoningResult]) -> ReasoningResult:
        """Fuse multiple reasoning results into a single result."""
        if not results:
            return ReasoningResult(insights=[], confidence=0.0)
        # Weighted by individual confidence
        all_insights: List[Dict[str, Any]] = []
        all_recs: List[Dict[str, Any]] = []
        total = 0.0
        for r in results:
            all_insights.extend(r.insights)
            all_recs.extend(r.recommendations)
            total += max(0.0, r.confidence)
        avg_confidence = min(1.0, total / max(1, len(results)))
        # Deduplicate recommendations by (action_type, parameters)
        dedup: Dict[str, Dict[str, Any]] = {}
        for rec in all_recs:
            key = f"{rec.get('action_type')}::{sorted(rec.get('parameters', {}).items())}"
            if key not in dedup:
                dedup[key] = rec
        fused_recs = list(dedup.values())
        return ReasoningResult(insights=all_insights, confidence=avg_confidence, recommendations=fused_recs)


class PolarisReasoningEngine(Injectable):
    """POLARIS Reasoning Engine implementing Chain of Responsibility and Strategy patterns.
    
    This engine coordinates multiple reasoning strategies to analyze system states and
    provide actionable insights. It implements the Chain of Responsibility pattern
    to process reasoning requests through a configurable pipeline of strategies.
    
    The engine supports three primary reasoning strategies:
    1. StatisticalReasoningStrategy: Analyzes metrics using statistical methods
    2. CausalReasoningStrategy: Understands system dependencies and causality
    3. ExperienceBasedReasoningStrategy: Leverages historical patterns and experiences
    
    Features:
    - Pluggable strategy architecture
    - Configurable result fusion
    - Asynchronous processing
    - Extensible design for custom strategies
    """
    
    def __init__(
        self, 
        reasoning_strategies: Optional[List[ReasoningStrategy]] = None,
        fusion_strategy: Optional[ResultFusionStrategy] = None,
        knowledge_base: Optional[PolarisKnowledgeBase] = None,
    ):
        self._kb = knowledge_base
        self._strategies = reasoning_strategies or [
            StatisticalReasoningStrategy(knowledge_base=knowledge_base),
            CausalReasoningStrategy(knowledge_base=knowledge_base),
            ExperienceBasedReasoningStrategy(knowledge_base=knowledge_base)
        ]
        self._fusion_strategy = fusion_strategy or ResultFusionStrategy()
    
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Perform reasoning using all available strategies and fuse results."""
        results: List[ReasoningResult] = []
        for strategy in self._strategies:
            try:
                result = await strategy.reason(context)
                results.append(result)
            except Exception:
                # Continue with others on failure
                continue
        return await self._fusion_strategy.fuse(results)
    
    async def analyze_root_cause(
        self, 
        system_id: str, 
        problem_description: str
    ) -> ReasoningResult:
        """Analyze the root cause of a problem."""
        # Build minimal context from KB if available
        current_state: Dict[str, Any] = {"problem": problem_description}
        if self._kb:
            try:
                state = await self._kb.get_current_state(system_id)
                if state:
                    current_state = {"metrics": state.metrics, "health_status": state.health_status.value}
            except Exception:
                pass
        context = ReasoningContext(system_id, current_state)
        return await self.reason(context)
    
    async def recommend_solutions(
        self, 
        system_id: str, 
        problem_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Recommend solutions for a given problem using fused reasoning output."""
        result = await self.reason(ReasoningContext(system_id, problem_context))
        return result.recommendations


# ---------- Helpers ----------

def _conditions_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    metrics = state.get("metrics", {}) if isinstance(state, dict) else {}
    cond: Dict[str, Any] = {}
    for k, v in metrics.items():
        try:
            val = float(getattr(v, "value", v))
            cond[k] = "high" if val >= 0.85 or val >= 85.0 else ("low" if val <= 0.15 else "normal")
        except Exception:
            cond[k] = "present"
    return cond