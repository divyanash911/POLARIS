#!/usr/bin/env python3
"""
Standalone POLARIS Stabilization Simulation

A self-contained simulation demonstrating POLARIS stabilizing an unstable system.
This version runs without external processes - everything is in-memory.

Components demonstrated:
- Mock System Simulator (generates unstable metrics, responds to actions)
- Statistical World Model (tracks metric trends)
- Rule-Based Reasoner (threshold evaluation)
- Adaptive Controller (action selection with cooldowns)
- Meta-Learner Interface (threshold evolution)

Usage:
    python scripts/run_standalone_stabilization.py [--duration SECONDS]
"""

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


class ActionType(Enum):
    """Supported adaptation actions."""
    SCALE_UP = "SCALE_UP"
    SCALE_DOWN = "SCALE_DOWN"
    RESTART_SERVICE = "RESTART_SERVICE"
    OPTIMIZE_CONFIG = "OPTIMIZE_CONFIG"
    ENABLE_CACHING = "ENABLE_CACHING"
    ADJUST_QOS = "ADJUST_QOS"


@dataclass
class SystemMetrics:
    """Current system metrics."""
    cpu_usage: float = 50.0
    memory_usage: float = 2048.0
    response_time: float = 100.0
    throughput: float = 50.0
    error_rate: float = 1.0
    capacity: int = 5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "response_time": self.response_time,
            "throughput": self.throughput,
            "error_rate": self.error_rate,
            "capacity": float(self.capacity)
        }


@dataclass
class AdaptationAction:
    """An adaptation action to execute."""
    action_id: str
    action_type: ActionType
    reason: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ThresholdRule:
    """A threshold-based rule for triggering actions."""
    rule_id: str
    metric: str
    operator: str  # "gt", "lt", "gte", "lte"
    threshold: float
    action_type: ActionType
    priority: int
    cooldown_seconds: float = 15.0
    
    def evaluate(self, metrics: SystemMetrics) -> bool:
        """Check if rule is triggered."""
        value = getattr(metrics, self.metric, None)
        if value is None:
            return False
        
        if self.operator == "gt":
            return value > self.threshold
        elif self.operator == "lt":
            return value < self.threshold
        elif self.operator == "gte":
            return value >= self.threshold
        elif self.operator == "lte":
            return value <= self.threshold
        return False


class MockSystemSimulator:
    """Simulates an external system with realistic metric behavior."""
    
    def __init__(self, initial_state: str = "unstable"):
        self.metrics = SystemMetrics()
        self.noise_factor = 0.1
        self.action_effects: Dict[ActionType, Dict[str, float]] = {
            ActionType.SCALE_UP: {
                "cpu_usage": -15.0,
                "memory_usage": -300.0,
                "response_time": -80.0,
                "throughput": 10.0,
                "error_rate": -2.0,
                "capacity": 2
            },
            ActionType.SCALE_DOWN: {
                "cpu_usage": 10.0,
                "memory_usage": 200.0,
                "response_time": 50.0,
                "throughput": -5.0,
                "capacity": -1
            },
            ActionType.RESTART_SERVICE: {
                "error_rate": -8.0,
                "response_time": -100.0,
                "cpu_usage": -10.0
            },
            ActionType.OPTIMIZE_CONFIG: {
                "response_time": -120.0,
                "throughput": 8.0,
                "cpu_usage": -5.0
            },
            ActionType.ENABLE_CACHING: {
                "response_time": -60.0,
                "throughput": 12.0,
                "cpu_usage": 3.0,
                "memory_usage": 200.0
            },
            ActionType.ADJUST_QOS: {
                "memory_usage": -400.0,
                "cpu_usage": -8.0,
                "throughput": -3.0
            }
        }
        
        if initial_state == "unstable":
            self._set_unstable_state()
    
    def _set_unstable_state(self):
        """Set system to unstable initial state."""
        self.metrics.cpu_usage = 92.0
        self.metrics.memory_usage = 4200.0
        self.metrics.response_time = 650.0
        self.metrics.throughput = 15.0
        self.metrics.error_rate = 12.0
        self.metrics.capacity = 3
    
    def get_metrics(self) -> SystemMetrics:
        """Get current metrics with noise."""
        # Add realistic noise
        self.metrics.cpu_usage += random.gauss(0, self.noise_factor * 5)
        self.metrics.memory_usage += random.gauss(0, self.noise_factor * 100)
        self.metrics.response_time += random.gauss(0, self.noise_factor * 30)
        self.metrics.throughput += random.gauss(0, self.noise_factor * 3)
        self.metrics.error_rate += random.gauss(0, self.noise_factor * 1)
        
        # Clamp values to realistic ranges
        self.metrics.cpu_usage = max(5, min(99, self.metrics.cpu_usage))
        self.metrics.memory_usage = max(500, min(8000, self.metrics.memory_usage))
        self.metrics.response_time = max(20, min(2000, self.metrics.response_time))
        self.metrics.throughput = max(1, min(200, self.metrics.throughput))
        self.metrics.error_rate = max(0, min(50, self.metrics.error_rate))
        self.metrics.capacity = max(1, min(20, self.metrics.capacity))
        
        self.metrics.timestamp = datetime.now(timezone.utc)
        return self.metrics
    
    def apply_action(self, action: AdaptationAction) -> bool:
        """Apply an adaptation action to the system."""
        effects = self.action_effects.get(action.action_type, {})
        
        for metric, delta in effects.items():
            current = getattr(self.metrics, metric, None)
            if current is not None:
                if metric == "capacity":
                    setattr(self.metrics, metric, int(current + delta))
                else:
                    setattr(self.metrics, metric, current + delta)
        
        return True


class StatisticalWorldModel:
    """Simple statistical world model for tracking trends."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: List[SystemMetrics] = []
    
    def update(self, metrics: SystemMetrics):
        """Update model with new metrics."""
        self.history.append(metrics)
        if len(self.history) > self.window_size:
            self.history.pop(0)
    
    def get_trend(self, metric: str) -> str:
        """Get trend direction for a metric."""
        if len(self.history) < 3:
            return "unknown"
        
        values = [getattr(m, metric, 0) for m in self.history[-5:]]
        if len(values) < 2:
            return "stable"
        
        avg_first = sum(values[:len(values)//2]) / (len(values)//2)
        avg_second = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        diff = avg_second - avg_first
        if abs(diff) < 2:
            return "stable"
        return "increasing" if diff > 0 else "decreasing"
    
    def predict_next(self, metric: str) -> float:
        """Simple prediction based on moving average."""
        if not self.history:
            return 0
        
        values = [getattr(m, metric, 0) for m in self.history[-3:]]
        return sum(values) / len(values)


class RuleBasedReasoner:
    """Rule-based reasoning engine."""
    
    def __init__(self):
        self.rules: List[ThresholdRule] = [
            ThresholdRule("cpu_critical", "cpu_usage", "gt", 85.0, ActionType.SCALE_UP, 5, 10.0),
            ThresholdRule("error_critical", "error_rate", "gt", 10.0, ActionType.RESTART_SERVICE, 5, 25.0),
            ThresholdRule("latency_high", "response_time", "gt", 500.0, ActionType.OPTIMIZE_CONFIG, 4, 15.0),
            ThresholdRule("throughput_low", "throughput", "lt", 25.0, ActionType.ENABLE_CACHING, 3, 20.0),
            ThresholdRule("memory_high", "memory_usage", "gt", 4000.0, ActionType.ADJUST_QOS, 3, 15.0),
            ThresholdRule("cpu_high", "cpu_usage", "gt", 70.0, ActionType.SCALE_UP, 2, 20.0),
        ]
    
    def evaluate(self, metrics: SystemMetrics) -> List[ThresholdRule]:
        """Evaluate all rules and return triggered ones sorted by priority."""
        triggered = [rule for rule in self.rules if rule.evaluate(metrics)]
        return sorted(triggered, key=lambda r: r.priority, reverse=True)


class AdaptiveController:
    """Adaptive controller with cooldown management."""
    
    def __init__(self, reasoner: RuleBasedReasoner):
        self.reasoner = reasoner
        self.last_action_time: Dict[ActionType, float] = {}
        self.action_count: Dict[ActionType, int] = {}
    
    def select_action(self, metrics: SystemMetrics) -> Optional[AdaptationAction]:
        """Select the best action based on current metrics."""
        triggered_rules = self.reasoner.evaluate(metrics)
        
        current_time = time.time()
        
        for rule in triggered_rules:
            last_time = self.last_action_time.get(rule.action_type, 0)
            
            if current_time - last_time >= rule.cooldown_seconds:
                action = AdaptationAction(
                    action_id=f"action_{int(current_time)}",
                    action_type=rule.action_type,
                    reason=rule.rule_id
                )
                return action
        
        return None
    
    def record_action(self, action: AdaptationAction):
        """Record that an action was taken."""
        self.last_action_time[action.action_type] = time.time()
        self.action_count[action.action_type] = self.action_count.get(action.action_type, 0) + 1


class MetaLearnerInterface:
    """Interface for meta-learning (threshold evolution)."""
    
    def __init__(self, reasoner: RuleBasedReasoner):
        self.reasoner = reasoner
        self.performance_history: List[Dict[str, Any]] = []
    
    def record_performance(self, metrics: SystemMetrics, action_taken: Optional[AdaptationAction]):
        """Record system performance for learning."""
        self.performance_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics.to_dict(),
            "action": action_taken.action_type.value if action_taken else None
        })
    
    def suggest_threshold_update(self) -> Optional[Dict[str, Any]]:
        """Suggest threshold updates based on performance (simplified)."""
        if len(self.performance_history) < 10:
            return None
        
        # Simple heuristic: if we're taking too many actions, relax thresholds
        recent = self.performance_history[-10:]
        action_count = sum(1 for p in recent if p["action"] is not None)
        
        if action_count > 7:
            return {
                "suggestion": "Consider relaxing thresholds - too many actions",
                "confidence": 0.6
            }
        elif action_count < 2:
            return {
                "suggestion": "Thresholds may be too relaxed",
                "confidence": 0.5
            }
        
        return None


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging."""
    log_level_value = getattr(logging, log_level.upper(), logging.INFO)
    
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=log_level_value,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / "standalone_simulation.log")
        ]
    )
    
    return logging.getLogger("polaris_sim")


def format_metrics(metrics: SystemMetrics, show_status: bool = True) -> str:
    """Format metrics for display."""
    lines = []
    
    thresholds = {
        "cpu_usage": (70, 85),
        "memory_usage": (3500, 4000),
        "response_time": (300, 500),
        "throughput": (30, 20),  # inverted
        "error_rate": (5, 10),
    }
    
    for name, value in [
        ("cpu_usage", metrics.cpu_usage),
        ("memory_usage", metrics.memory_usage),
        ("response_time", metrics.response_time),
        ("throughput", metrics.throughput),
        ("error_rate", metrics.error_rate),
        ("capacity", float(metrics.capacity))
    ]:
        if show_status and name in thresholds:
            warn, crit = thresholds[name]
            if name in ["throughput"]:
                status = "🔴" if value < crit else "🟡" if value < warn else "🟢"
            else:
                status = "🔴" if value > crit else "🟡" if value > warn else "🟢"
        else:
            status = "⚪"
        
        if name == "memory_usage":
            formatted = f"{value:.0f} MB"
        elif name == "response_time":
            formatted = f"{value:.0f} ms"
        elif name == "throughput":
            formatted = f"{value:.1f} req/s"
        elif name in ["cpu_usage", "error_rate"]:
            formatted = f"{value:.1f}%"
        else:
            formatted = f"{value:.0f}"
        
        lines.append(f"  {name:18s}: {formatted:12s} {status}")
    
    return "\n".join(lines)


def is_system_stable(metrics: SystemMetrics) -> bool:
    """Check if system is in stable state (all metrics in good range)."""
    # System is stable when all metrics are in the "green" zone
    return (
        metrics.cpu_usage < 70 and       # Green zone
        metrics.memory_usage < 3700 and  # Green zone
        metrics.response_time < 450 and  # Green zone
        metrics.throughput > 30 and      # Green zone
        metrics.error_rate < 5           # Green zone
    )


async def run_simulation(duration: int, logger: logging.Logger):
    """Run the stabilization simulation."""
    
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + " POLARIS Standalone Stabilization Simulation ".center(58) + "║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("")
    
    # Initialize components
    logger.info("Initializing POLARIS components...")
    mock_system = MockSystemSimulator(initial_state="unstable")
    world_model = StatisticalWorldModel()
    reasoner = RuleBasedReasoner()
    controller = AdaptiveController(reasoner)
    meta_learner = MetaLearnerInterface(reasoner)
    
    logger.info("✓ Mock System Simulator (unstable initial state)")
    logger.info("✓ Statistical World Model")
    logger.info("✓ Rule-Based Reasoner (6 threshold rules)")
    logger.info("✓ Adaptive Controller (with cooldowns)")
    logger.info("✓ Meta-Learner Interface")
    logger.info("")
    
    # Show initial state
    logger.info("=" * 60)
    logger.info("INITIAL SYSTEM STATE (UNSTABLE)")
    logger.info("=" * 60)
    initial_metrics = mock_system.get_metrics()
    logger.info(format_metrics(initial_metrics))
    logger.info("")
    
    # Simulation loop
    logger.info("=" * 60)
    logger.info("STARTING MAPE-K STABILIZATION LOOP")
    logger.info("=" * 60)
    logger.info("")
    
    start_time = time.time()
    iteration = 0
    actions_taken = []
    metrics_history = []
    
    while (time.time() - start_time) < duration:
        iteration += 1
        elapsed = int(time.time() - start_time)
        
        logger.info(f"─── Cycle {iteration} [{elapsed}s / {duration}s] ───")
        
        # MONITOR: Collect metrics
        metrics = mock_system.get_metrics()
        metrics_history.append(metrics.to_dict())
        
        # Update world model
        world_model.update(metrics)
        
        logger.info(format_metrics(metrics))
        
        # Show trends
        cpu_trend = world_model.get_trend("cpu_usage")
        error_trend = world_model.get_trend("error_rate")
        logger.info(f"  Trends: CPU {cpu_trend}, Errors {error_trend}")
        
        # ANALYZE & PLAN: Select action
        action = controller.select_action(metrics)
        
        if action:
            # EXECUTE: Apply action
            logger.info(f"  🔧 ACTION: {action.action_type.value} (reason: {action.reason})")
            
            success = mock_system.apply_action(action)
            if success:
                controller.record_action(action)
                actions_taken.append({
                    "iteration": iteration,
                    "action": action.action_type.value,
                    "reason": action.reason
                })
                logger.info(f"  ✓ Action executed successfully")
            else:
                logger.info(f"  ✗ Action failed")
        else:
            triggered = reasoner.evaluate(metrics)
            if triggered:
                logger.info(f"  ⏳ Rules triggered but on cooldown: {[r.rule_id for r in triggered]}")
            else:
                logger.info(f"  ✓ No action needed - within thresholds")
        
        # Record for meta-learner
        meta_learner.record_performance(metrics, action)
        
        # Check for meta-learner suggestions
        suggestion = meta_learner.suggest_threshold_update()
        if suggestion:
            logger.info(f"  💡 Meta-Learner: {suggestion['suggestion']}")
        
        # Check if stable
        if is_system_stable(metrics):
            logger.info("")
            logger.info("🎉 SYSTEM STABILIZED!")
            break
        
        logger.info("")
        await asyncio.sleep(3)
    
    # Final report
    logger.info("")
    logger.info("=" * 60)
    logger.info("SIMULATION COMPLETE")
    logger.info("=" * 60)
    
    final_metrics = mock_system.get_metrics()
    logger.info(f"Total iterations: {iteration}")
    logger.info(f"Actions taken: {len(actions_taken)}")
    
    if actions_taken:
        action_counts = {}
        for a in actions_taken:
            action_counts[a["action"]] = action_counts.get(a["action"], 0) + 1
        logger.info("Action breakdown:")
        for action, count in sorted(action_counts.items()):
            logger.info(f"  - {action}: {count}")
    
    logger.info("")
    logger.info("Final metrics:")
    logger.info(format_metrics(final_metrics))
    
    stable = is_system_stable(final_metrics)
    logger.info("")
    if stable:
        logger.info("✓ System is STABLE")
    else:
        logger.info("⚠ System not fully stabilized (may need more time)")
    
    # Save report
    report_path = PROJECT_ROOT / "logs" / "standalone_simulation_report.json"
    report = {
        "simulation_time": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "iterations": iteration,
        "actions_taken": actions_taken,
        "final_stable": stable,
        "initial_metrics": metrics_history[0] if metrics_history else {},
        "final_metrics": final_metrics.to_dict()
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved to: {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="POLARIS Standalone Stabilization Simulation")
    parser.add_argument('--duration', type=int, default=90, help='Simulation duration (seconds)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING'])
    args = parser.parse_args()
    
    logger = setup_logging(args.log_level)
    
    try:
        asyncio.run(run_simulation(args.duration, logger))
    except KeyboardInterrupt:
        logger.info("\nSimulation interrupted by user")
        sys.exit(0)


if __name__ == '__main__':
    main()
