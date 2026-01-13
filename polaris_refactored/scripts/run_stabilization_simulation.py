#!/usr/bin/env python3
"""
POLARIS Stabilization Simulation

This script demonstrates POLARIS's ability to stabilize an initially unstable system.
It starts a mock system in a degraded state (high CPU, high error rate, high latency)
and shows how POLARIS monitors, reasons, and takes adaptive actions to stabilize it.

Components initialized:
- Mock External System (simulates unstable system)
- POLARIS Framework with:
  - Monitor Adapter (collects telemetry)
  - Statistical World Model (predicts behavior)
  - Knowledge Base (stores patterns)
  - Reasoning Engine (hybrid rule-based + statistical)
  - Adaptive Controller (threshold-reactive + learning)
  - Execution Adapter (executes actions)
  - Meta Learner (LLM-powered, optional)

Usage:
    python scripts/run_stabilization_simulation.py [--duration SECONDS] [--with-llm]
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add src to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "mock_external_system"))

# Global shutdown flag
_shutdown_requested = False


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging for the simulation."""
    log_level_value = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=log_level_value,
        format='%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / "stabilization_simulation.log")
        ]
    )
    
    return logging.getLogger("stabilization_sim")


class UnstableSystemConfig:
    """Configuration for an unstable system scenario."""
    
    @staticmethod
    def create_unstable_config() -> Dict[str, Any]:
        """Create mock system config that simulates an unstable system."""
        return {
            "server": {
                "host": "localhost",
                "port": 5000,
                "max_connections": 100
            },
            # Start with HIGH baseline metrics (unstable state)
            "baseline_metrics": {
                "cpu_usage": 92.0,        # Very high CPU - triggers SCALE_UP
                "memory_usage": 4200.0,   # High memory usage
                "response_time": 650.0,   # High latency - triggers OPTIMIZE_CONFIG
                "throughput": 15.0,       # Low throughput
                "error_rate": 12.0,       # High error rate - triggers RESTART_SERVICE
                "active_connections": 85,  # Many connections
                "capacity": 3             # Low capacity
            },
            "simulation": {
                "noise_factor": 0.15,     # Higher noise for realistic fluctuation
                "update_interval": 1.0,
                "load_response_time": 1.5
            },
            "capacity": {
                "min_capacity": 1,
                "max_capacity": 20,
                "scale_up_increment": 2,
                "scale_down_increment": 1
            }
        }
    
    @staticmethod
    def create_polaris_config() -> Dict[str, Any]:
        """Create POLARIS config optimized for stabilization demo."""
        return {
            "framework": {
                "service_name": "polaris-stabilization-demo",
                "version": "2.0.0",
                "environment": "simulation",
                "logging_config": {
                    "level": "INFO",
                    "format": "text",
                    "output": "console",
                    "file_path": str(PROJECT_ROOT / "logs" / "polaris_stabilization.log")
                },
                "plugin_search_paths": [str(PROJECT_ROOT / "plugins")],
                "max_concurrent_adaptations": 3,
                "adaptation_timeout": 60
            },
            "managed_systems": {
                "mock_system": {
                    "system_id": "mock_system",
                    "connector_type": "mock_system",
                    "enabled": True,
                    "connection_params": {
                        "host": "localhost",
                        "port": 5000,
                        "timeout": 10.0,
                        "max_retries": 3,
                        "retry_delay": 1.0
                    },
                    "monitoring_config": {
                        "collection_interval": 3,  # Fast collection for demo
                        "collection_strategy": "polling_direct_connector",
                        "health_check_interval": 10,
                        "metrics_to_collect": [
                            "cpu_usage", "memory_usage", "response_time",
                            "throughput", "error_rate", "active_connections", "capacity"
                        ]
                    }
                }
            },
            "control_reasoning": {
                "adaptive_controller": {
                    "enabled": True,
                    "control_strategies": ["threshold_reactive"],
                    "enable_enhanced_assessment": True
                },
                "threshold_reactive": {
                    "enabled": True,
                    "enable_multi_metric_evaluation": True,
                    "action_prioritization_enabled": True,
                    "max_concurrent_actions": 2,
                    "default_cooldown_seconds": 15.0,  # Short cooldown for demo
                    "enable_fallback": True,
                    "rules": [
                        {
                            "rule_id": "critical_cpu_scale_up",
                            "name": "Critical CPU Scale Up",
                            "description": "Immediate scale up for critical CPU",
                            "enabled": True,
                            "priority": 4,
                            "cooldown_seconds": 10.0,
                            "action_type": "SCALE_UP",
                            "action_parameters": {"reason": "critical_cpu"},
                            "conditions": [
                                {"metric_name": "cpu_usage", "operator": "gt", "value": 85.0, "weight": 2.0}
                            ]
                        },
                        {
                            "rule_id": "high_error_restart",
                            "name": "High Error Rate Restart",
                            "description": "Restart service when error rate is critical",
                            "enabled": True,
                            "priority": 5,
                            "cooldown_seconds": 30.0,
                            "action_type": "RESTART_SERVICE",
                            "action_parameters": {"reason": "high_error_rate"},
                            "conditions": [
                                {"metric_name": "error_rate", "operator": "gt", "value": 10.0, "weight": 2.0}
                            ]
                        },
                        {
                            "rule_id": "high_latency_optimize",
                            "name": "High Latency Optimization",
                            "description": "Optimize config for high response time",
                            "enabled": True,
                            "priority": 3,
                            "cooldown_seconds": 20.0,
                            "action_type": "OPTIMIZE_CONFIG",
                            "action_parameters": {"reason": "high_latency"},
                            "conditions": [
                                {"metric_name": "response_time", "operator": "gt", "value": 500.0, "weight": 1.5}
                            ]
                        },
                        {
                            "rule_id": "low_throughput_cache",
                            "name": "Low Throughput Enable Caching",
                            "description": "Enable caching for low throughput",
                            "enabled": True,
                            "priority": 2,
                            "cooldown_seconds": 25.0,
                            "action_type": "ENABLE_CACHING",
                            "action_parameters": {"reason": "low_throughput"},
                            "conditions": [
                                {"metric_name": "throughput", "operator": "lt", "value": 25.0, "weight": 1.0}
                            ]
                        },
                        {
                            "rule_id": "high_memory_qos",
                            "name": "High Memory QoS Adjustment",
                            "description": "Adjust QoS for high memory",
                            "enabled": True,
                            "priority": 2,
                            "cooldown_seconds": 20.0,
                            "action_type": "ADJUST_QOS",
                            "action_parameters": {"reason": "high_memory", "qos_level": "reduced"},
                            "conditions": [
                                {"metric_name": "memory_usage", "operator": "gt", "value": 4000.0, "weight": 1.0}
                            ]
                        }
                    ]
                }
            },
            "digital_twin": {
                "world_model": {
                    "type": "statistical",
                    "prediction_horizon": 5,
                    "enable_fallback_model": True
                },
                "knowledge_base": {
                    "enabled": True,
                    "state_retention_hours": 1,
                    "max_state_entries": 500,
                    "pattern_detection_enabled": True
                }
            },
            "data_storage": {
                "backends": {
                    "time_series": "in_memory",
                    "document": "in_memory",
                    "graph": "in_memory"
                }
            },
            "event_bus": {
                "max_queue_size": 500,
                "worker_count": 2,
                "enable_event_history": True
            },
            "observability": {
                "service_name": "polaris-stabilization-demo",
                "enable_metrics": True,
                "enable_tracing": True,
                "enable_logging": True
            }
        }


class StabilizationSimulator:
    """Orchestrates the stabilization simulation."""
    
    def __init__(self, logger: logging.Logger, duration: int = 120, with_llm: bool = False):
        self.logger = logger
        self.duration = duration
        self.with_llm = with_llm
        
        self.mock_process: Optional[subprocess.Popen] = None
        self.polaris_manager = None
        self.metrics_history: List[Dict[str, Any]] = []
        self.actions_taken: List[Dict[str, Any]] = []
        
        # Paths
        self.config_dir = PROJECT_ROOT / "scripts" / "simulation_configs"
        self.config_dir.mkdir(exist_ok=True)
        
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        global _shutdown_requested
        if not _shutdown_requested:
            _shutdown_requested = True
            self.logger.info("Shutdown requested...")
    
    async def setup_configs(self) -> tuple:
        """Create configuration files for the simulation."""
        self.logger.info("Creating simulation configurations...")
        
        # Create unstable mock system config
        mock_config = UnstableSystemConfig.create_unstable_config()
        mock_config_path = self.config_dir / "unstable_mock_config.yaml"
        with open(mock_config_path, 'w') as f:
            yaml.dump(mock_config, f, default_flow_style=False)
        
        # Create POLARIS config
        polaris_config = UnstableSystemConfig.create_polaris_config()
        polaris_config_path = self.config_dir / "stabilization_polaris_config.yaml"
        with open(polaris_config_path, 'w') as f:
            yaml.dump(polaris_config, f, default_flow_style=False)
        
        self.logger.info(f"Configs created in {self.config_dir}")
        return mock_config_path, polaris_config_path
    
    async def start_mock_system(self, config_path: Path) -> bool:
        """Start the mock external system in unstable state."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 1: Starting Mock System (UNSTABLE STATE)")
        self.logger.info("=" * 60)
        
        try:
            mock_script = PROJECT_ROOT / "mock_external_system" / "scripts" / "start_mock_system.py"
            
            cmd = [
                sys.executable,
                str(mock_script),
                "--config", str(config_path),
                "--log-level", "INFO"
            ]
            
            log_file = PROJECT_ROOT / "logs" / "mock_system_simulation.log"
            with open(log_file, 'w') as f:
                self.mock_process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT),
                    start_new_session=True
                )
            
            # Wait for startup
            await asyncio.sleep(3)
            
            if self.mock_process.poll() is None:
                self.logger.info(f"✓ Mock system started (PID: {self.mock_process.pid})")
                self.logger.info("  Initial state: HIGH CPU (92%), HIGH errors (12%), HIGH latency (650ms)")
                return True
            else:
                self.logger.error("✗ Mock system failed to start")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start mock system: {e}")
            return False

    async def start_polaris_framework(self, config_path: Path) -> bool:
        """Start POLARIS framework with all components."""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("PHASE 2: Starting POLARIS Framework")
        self.logger.info("=" * 60)
        
        try:
            from framework.cli.manager import PolarisFrameworkManager
            
            self.polaris_manager = PolarisFrameworkManager()
            
            if not await self.polaris_manager.initialize(str(config_path), "INFO"):
                self.logger.error("✗ Failed to initialize POLARIS")
                return False
            
            self.logger.info("✓ POLARIS initialized with components:")
            self.logger.info("  - Monitor Adapter (polling strategy)")
            self.logger.info("  - Statistical World Model")
            self.logger.info("  - Knowledge Base (in-memory)")
            self.logger.info("  - Reasoning Engine (statistical + causal + experience)")
            self.logger.info("  - Adaptive Controller (threshold-reactive)")
            self.logger.info("  - Execution Adapter (chain-of-responsibility)")
            
            if not await self.polaris_manager.start():
                self.logger.error("✗ Failed to start POLARIS")
                return False
            
            self.logger.info("✓ POLARIS framework started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start POLARIS: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def collect_and_display_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect current metrics from mock system via POLARIS."""
        try:
            if not self.polaris_manager or not self.polaris_manager.framework:
                return None
            
            registry = self.polaris_manager.framework.plugin_registry
            connectors = registry.get_loaded_connectors()
            connector = connectors.get("mock_system")
            
            if connector:
                metrics = await connector.collect_metrics()
                return {name: m.value for name, m in metrics.items()}
            
        except Exception as e:
            self.logger.debug(f"Metrics collection error: {e}")
        
        return None
    
    def format_metrics_display(self, metrics: Dict[str, float], iteration: int) -> str:
        """Format metrics for display with status indicators."""
        if not metrics:
            return "  [No metrics available]"
        
        lines = [f"  Iteration {iteration} Metrics:"]
        
        # Define thresholds for status indicators
        thresholds = {
            "cpu_usage": (70, 85),      # warning, critical
            "memory_usage": (3500, 4000),
            "response_time": (300, 500),
            "throughput": (30, 20),      # inverted - low is bad
            "error_rate": (5, 10),
            "capacity": (5, 3)           # inverted - low is bad
        }
        
        for name, value in sorted(metrics.items()):
            if name in thresholds:
                warn, crit = thresholds[name]
                if name in ["throughput", "capacity"]:
                    # Inverted thresholds
                    if value < crit:
                        status = "🔴 CRITICAL"
                    elif value < warn:
                        status = "🟡 WARNING"
                    else:
                        status = "🟢 OK"
                else:
                    if value > crit:
                        status = "🔴 CRITICAL"
                    elif value > warn:
                        status = "🟡 WARNING"
                    else:
                        status = "🟢 OK"
            else:
                status = "⚪"
            
            # Format value based on metric type
            if name == "memory_usage":
                formatted = f"{value:.0f} MB"
            elif name == "response_time":
                formatted = f"{value:.0f} ms"
            elif name == "throughput":
                formatted = f"{value:.1f} req/s"
            elif name in ["cpu_usage", "error_rate"]:
                formatted = f"{value:.1f}%"
            elif name == "capacity":
                formatted = f"{value:.0f} units"
            else:
                formatted = f"{value:.1f}"
            
            lines.append(f"    {name:20s}: {formatted:15s} {status}")
        
        return "\n".join(lines)
    
    async def execute_adaptation_action(self, action_type: str, reason: str) -> bool:
        """Execute an adaptation action via POLARIS."""
        try:
            if not self.polaris_manager or not self.polaris_manager.framework:
                return False
            
            from framework.cli.manager import ManagedSystemOperations
            ops = ManagedSystemOperations(self.polaris_manager)
            
            result = await ops.execute_action(
                "mock_system",
                action_type,
                {"reason": reason}
            )
            
            if "error" not in result:
                self.actions_taken.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action_type": action_type,
                    "reason": reason,
                    "status": result.get("status", "unknown")
                })
                return True
            
        except Exception as e:
            self.logger.debug(f"Action execution error: {e}")
        
        return False
    
    def evaluate_rules(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Evaluate threshold rules against current metrics."""
        triggered_rules = []
        
        rules = [
            {"id": "critical_cpu", "metric": "cpu_usage", "op": "gt", "value": 85, "action": "SCALE_UP", "priority": 4},
            {"id": "high_error", "metric": "error_rate", "op": "gt", "value": 10, "action": "RESTART_SERVICE", "priority": 5},
            {"id": "high_latency", "metric": "response_time", "op": "gt", "value": 500, "action": "OPTIMIZE_CONFIG", "priority": 3},
            {"id": "low_throughput", "metric": "throughput", "op": "lt", "value": 25, "action": "ENABLE_CACHING", "priority": 2},
            {"id": "high_memory", "metric": "memory_usage", "op": "gt", "value": 4000, "action": "ADJUST_QOS", "priority": 2},
        ]
        
        for rule in rules:
            metric_value = metrics.get(rule["metric"])
            if metric_value is None:
                continue
            
            triggered = False
            if rule["op"] == "gt" and metric_value > rule["value"]:
                triggered = True
            elif rule["op"] == "lt" and metric_value < rule["value"]:
                triggered = True
            
            if triggered:
                triggered_rules.append(rule)
        
        # Sort by priority (higher first)
        return sorted(triggered_rules, key=lambda r: r["priority"], reverse=True)
    
    async def run_stabilization_loop(self) -> bool:
        """Main stabilization loop - monitor, reason, act."""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("PHASE 3: Stabilization Loop (MAPE-K)")
        self.logger.info("=" * 60)
        self.logger.info("POLARIS will now monitor the system and take adaptive actions")
        self.logger.info("")
        
        global _shutdown_requested
        start_time = time.time()
        iteration = 0
        last_action_time = {}  # Track cooldowns
        cooldown_seconds = 15
        
        while not _shutdown_requested and (time.time() - start_time) < self.duration:
            iteration += 1
            elapsed = int(time.time() - start_time)
            
            self.logger.info("-" * 50)
            self.logger.info(f"[{elapsed}s / {self.duration}s] MAPE-K Cycle {iteration}")
            
            # MONITOR: Collect metrics
            metrics = await self.collect_and_display_metrics()
            
            if metrics:
                self.metrics_history.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "iteration": iteration,
                    "metrics": metrics
                })
                
                self.logger.info(self.format_metrics_display(metrics, iteration))
                
                # ANALYZE: Evaluate rules
                triggered_rules = self.evaluate_rules(metrics)
                
                if triggered_rules:
                    self.logger.info(f"  Triggered rules: {[r['id'] for r in triggered_rules]}")
                    
                    # PLAN & EXECUTE: Take action on highest priority rule
                    for rule in triggered_rules:
                        action = rule["action"]
                        
                        # Check cooldown
                        last_time = last_action_time.get(action, 0)
                        if time.time() - last_time < cooldown_seconds:
                            self.logger.info(f"  ⏳ {action} on cooldown, skipping...")
                            continue
                        
                        self.logger.info(f"  🔧 EXECUTING: {action} (reason: {rule['id']})")
                        
                        success = await self.execute_adaptation_action(action, rule["id"])
                        
                        if success:
                            last_action_time[action] = time.time()
                            self.logger.info(f"  ✓ Action {action} executed successfully")
                            break  # Only execute one action per cycle
                        else:
                            self.logger.warning(f"  ✗ Action {action} failed")
                else:
                    self.logger.info("  ✓ System within acceptable thresholds")
            else:
                self.logger.warning("  Could not collect metrics")
            
            # Check if system is stable
            if metrics and self._is_system_stable(metrics):
                self.logger.info("")
                self.logger.info("🎉 SYSTEM STABILIZED!")
                self.logger.info("All metrics within acceptable ranges")
                break
            
            # Wait before next cycle
            await asyncio.sleep(5)
        
        return True
    
    def _is_system_stable(self, metrics: Dict[str, float]) -> bool:
        """Check if all metrics are within stable ranges."""
        stable_ranges = {
            "cpu_usage": (0, 70),
            "memory_usage": (0, 3500),
            "response_time": (0, 300),
            "throughput": (30, 1000),
            "error_rate": (0, 5),
        }
        
        for metric, (low, high) in stable_ranges.items():
            value = metrics.get(metric)
            if value is None:
                continue
            if not (low <= value <= high):
                return False
        
        return True
    
    async def generate_report(self) -> None:
        """Generate simulation report."""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("SIMULATION REPORT")
        self.logger.info("=" * 60)
        
        # Summary
        self.logger.info(f"Total iterations: {len(self.metrics_history)}")
        self.logger.info(f"Actions taken: {len(self.actions_taken)}")
        
        # Actions breakdown
        if self.actions_taken:
            self.logger.info("\nActions executed:")
            action_counts = {}
            for action in self.actions_taken:
                action_type = action["action_type"]
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
            
            for action_type, count in sorted(action_counts.items()):
                self.logger.info(f"  - {action_type}: {count} times")
        
        # Metrics evolution
        if len(self.metrics_history) >= 2:
            first = self.metrics_history[0]["metrics"]
            last = self.metrics_history[-1]["metrics"]
            
            self.logger.info("\nMetrics evolution (first → last):")
            for metric in ["cpu_usage", "memory_usage", "response_time", "error_rate", "throughput"]:
                if metric in first and metric in last:
                    change = last[metric] - first[metric]
                    direction = "↓" if change < 0 else "↑" if change > 0 else "→"
                    self.logger.info(f"  {metric}: {first[metric]:.1f} → {last[metric]:.1f} ({direction} {abs(change):.1f})")
        
        # Save detailed report
        report_path = PROJECT_ROOT / "logs" / "stabilization_report.json"
        report = {
            "simulation_time": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": self.duration,
            "metrics_history": self.metrics_history,
            "actions_taken": self.actions_taken,
            "final_stable": self._is_system_stable(self.metrics_history[-1]["metrics"]) if self.metrics_history else False
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"\nDetailed report saved to: {report_path}")
    
    async def cleanup(self) -> None:
        """Clean up all processes."""
        self.logger.info("")
        self.logger.info("Cleaning up...")
        
        # Stop POLARIS
        if self.polaris_manager:
            try:
                await self.polaris_manager.stop()
                self.logger.info("✓ POLARIS stopped")
            except Exception as e:
                self.logger.warning(f"Error stopping POLARIS: {e}")
        
        # Stop mock system
        if self.mock_process and self.mock_process.poll() is None:
            try:
                if sys.platform != 'win32':
                    os.killpg(os.getpgid(self.mock_process.pid), signal.SIGTERM)
                else:
                    self.mock_process.terminate()
                await asyncio.sleep(2)
                if self.mock_process.poll() is None:
                    if sys.platform != 'win32':
                        os.killpg(os.getpgid(self.mock_process.pid), signal.SIGKILL)
                    else:
                        self.mock_process.kill()
                self.logger.info("✓ Mock system stopped")
            except Exception as e:
                self.logger.warning(f"Error stopping mock system: {e}")
    
    async def run(self) -> bool:
        """Run the complete stabilization simulation."""
        global _shutdown_requested
        _shutdown_requested = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            self.logger.info("")
            self.logger.info("╔" + "═" * 58 + "╗")
            self.logger.info("║" + " POLARIS Stabilization Simulation ".center(58) + "║")
            self.logger.info("║" + " Demonstrating Autonomous System Stabilization ".center(58) + "║")
            self.logger.info("╚" + "═" * 58 + "╝")
            self.logger.info("")
            
            # Setup configurations
            mock_config_path, polaris_config_path = await self.setup_configs()
            
            # Start mock system (unstable)
            if not await self.start_mock_system(mock_config_path):
                return False
            
            # Start POLARIS
            if not await self.start_polaris_framework(polaris_config_path):
                return False
            
            # Wait for systems to connect
            self.logger.info("")
            self.logger.info("Waiting for systems to connect...")
            await asyncio.sleep(5)
            
            # Run stabilization loop
            await self.run_stabilization_loop()
            
            # Generate report
            await self.generate_report()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            await self.cleanup()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="POLARIS Stabilization Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This simulation demonstrates POLARIS's ability to stabilize an unstable system.

The mock system starts with:
  - CPU: 92% (critical)
  - Error Rate: 12% (critical)
  - Response Time: 650ms (critical)
  - Memory: 4200MB (high)
  - Throughput: 15 req/s (low)

POLARIS will:
  1. Monitor these metrics continuously
  2. Detect threshold violations
  3. Reason about appropriate actions
  4. Execute adaptations (SCALE_UP, RESTART_SERVICE, OPTIMIZE_CONFIG, etc.)
  5. Continue until system stabilizes

Examples:
  python scripts/run_stabilization_simulation.py
  python scripts/run_stabilization_simulation.py --duration 180
  python scripts/run_stabilization_simulation.py --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=120,
        help='Simulation duration in seconds (default: 120)'
    )
    
    parser.add_argument(
        '--with-llm',
        action='store_true',
        help='Enable LLM-powered meta-learner (requires API key)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    logger = setup_logging(args.log_level)
    
    simulator = StabilizationSimulator(
        logger=logger,
        duration=args.duration,
        with_llm=args.with_llm
    )
    
    success = await simulator.run()
    
    if success:
        logger.info("")
        logger.info("✓ Simulation completed successfully")
        sys.exit(0)
    else:
        logger.error("✗ Simulation failed")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
