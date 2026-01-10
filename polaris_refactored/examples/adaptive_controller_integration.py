"""
Adaptive Controller Integration Example

Demonstrates how to integrate the adaptive controller with the Polaris
configuration system and enable threshold evolution via the meta learner.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

from framework.configuration.core import PolarisConfiguration
from framework.configuration.adaptive_controller_source import AdaptiveControllerConfigurationSource
from control_reasoning.adaptive_controller import PolarisAdaptiveController
from meta_learner.llm_meta_learner import LLMMetaLearner


async def setup_adaptive_controller_with_polaris_config():
    """Set up adaptive controller with Polaris configuration system."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 1. Create adaptive controller configuration source
    config_path = Path(__file__).parent.parent / "config" / "adaptive_controller_runtime.yaml"
    
    def on_threshold_evolution(threshold_name: str, old_value: float, new_value: float):
        """Callback for threshold evolution events."""
        logger.info(f"Threshold evolved: {threshold_name} {old_value} -> {new_value}")
    
    adaptive_config_source = AdaptiveControllerConfigurationSource(
        file_path=config_path,
        priority=100,
        enable_evolution=True,
        evolution_callbacks=[on_threshold_evolution]
    )
    
    # 2. Create Polaris configuration with adaptive controller source
    polaris_config = PolarisConfiguration(
        sources=[adaptive_config_source],
        enable_hot_reload=True
    )
    
    # 3. Create adaptive controller with Polaris configuration
    adaptive_controller = PolarisAdaptiveController(
        polaris_config=polaris_config,
        config_key="adaptive_controller",
        enable_config_watch=True,
        enable_enhanced_assessment=True
    )
    
    # 4. Create meta learner with controller reference
    meta_learner = LLMMetaLearner(
        component_id="main_meta_learner",
        config_path=str(Path(__file__).parent.parent / "config" / "meta_learner_config.yaml"),
        adaptive_controller=adaptive_controller,
        managed_system_ids=["web_service", "database", "cache"]
    )
    
    # 5. Add configuration change callback to controller
    def on_config_change(config):
        logger.info(f"Controller configuration updated: version {config.version}")
    
    adaptive_controller.add_config_callback(on_config_change)
    
    logger.info("Adaptive controller integration setup complete")
    
    return adaptive_controller, meta_learner, polaris_config


async def demonstrate_threshold_evolution():
    """Demonstrate threshold evolution capabilities."""
    
    logger = logging.getLogger(__name__)
    
    # Set up components
    controller, meta_learner, config = await setup_adaptive_controller_with_polaris_config()
    
    # Start meta learner
    await meta_learner.start()
    
    logger.info("=== Initial Configuration ===")
    initial_config = controller.get_config()
    logger.info(f"CPU High Threshold: {initial_config.get_threshold_value('cpu_high')}")
    logger.info(f"Memory High Threshold: {initial_config.get_threshold_value('memory_high')}")
    logger.info(f"Response Time Warning: {initial_config.get_threshold_value('response_time_warning_ms')}")
    
    # Simulate meta learner updating thresholds
    logger.info("\n=== Simulating Meta Learner Threshold Updates ===")
    
    # Update CPU threshold
    success = controller.update_threshold(
        threshold_name="cpu_high",
        new_value=75.0,
        updated_by="meta_learner_demo",
        reason="optimization_based_on_analysis",
        confidence=0.85,
        performance_impact=0.1  # Expected 10% improvement
    )
    
    if success:
        logger.info("✓ CPU high threshold updated successfully")
    else:
        logger.error("✗ Failed to update CPU high threshold")
    
    # Update memory threshold
    success = controller.update_threshold(
        threshold_name="memory_high",
        new_value=82.0,
        updated_by="meta_learner_demo",
        reason="prevent_memory_pressure",
        confidence=0.92
    )
    
    if success:
        logger.info("✓ Memory high threshold updated successfully")
    else:
        logger.error("✗ Failed to update memory high threshold")
    
    # Try invalid update (should fail)
    success = controller.update_threshold(
        threshold_name="cpu_high",
        new_value=150.0,  # Invalid - exceeds max_value
        updated_by="meta_learner_demo",
        reason="test_validation"
    )
    
    if not success:
        logger.info("✓ Invalid threshold update correctly rejected")
    else:
        logger.error("✗ Invalid threshold update was incorrectly accepted")
    
    # Show updated configuration
    logger.info("\n=== Updated Configuration ===")
    updated_config = controller.get_config()
    logger.info(f"CPU High Threshold: {updated_config.get_threshold_value('cpu_high')}")
    logger.info(f"Memory High Threshold: {updated_config.get_threshold_value('memory_high')}")
    
    # Show evolution metadata
    logger.info("\n=== Evolution Metadata ===")
    cpu_threshold = updated_config.cpu_high
    logger.info(f"CPU High Evolution Count: {cpu_threshold.metadata.evolution_count}")
    logger.info(f"CPU High Last Updated: {cpu_threshold.metadata.last_updated}")
    logger.info(f"CPU High Updated By: {cpu_threshold.metadata.updated_by}")
    logger.info(f"CPU High Confidence: {cpu_threshold.metadata.confidence_score}")
    logger.info(f"CPU High Previous Value: {cpu_threshold.metadata.previous_value}")
    
    # Demonstrate configuration reload
    logger.info("\n=== Testing Configuration Reload ===")
    reload_success = controller.reload_config()
    if reload_success:
        logger.info("✓ Configuration reloaded successfully")
    else:
        logger.info("ℹ No configuration changes detected")
    
    # Stop meta learner
    await meta_learner.stop()
    
    logger.info("\n=== Demonstration Complete ===")


async def demonstrate_meta_learner_analysis():
    """Demonstrate meta learner analysis and threshold evolution."""
    
    logger = logging.getLogger(__name__)
    
    # Set up components
    controller, meta_learner, config = await setup_adaptive_controller_with_polaris_config()
    
    # Mock LLM client for demonstration
    class MockLLMClient:
        async def generate_response(self, request):
            # Mock response that suggests threshold adjustments
            class MockResponse:
                def __init__(self):
                    self.content = '''
                    {
                        "proposals": [
                            {
                                "parameter_path": "thresholds.cpu_high",
                                "current_value": 80.0,
                                "proposed_value": 75.0,
                                "rationale": "System shows stable performance at lower CPU thresholds",
                                "confidence": 0.85,
                                "expected_impact": "Reduced resource usage while maintaining performance"
                            },
                            {
                                "parameter_path": "thresholds.response_time_warning_ms",
                                "current_value": 200.0,
                                "proposed_value": 180.0,
                                "rationale": "Tighter response time control for better user experience",
                                "confidence": 0.78,
                                "expected_impact": "Earlier detection of performance issues"
                            }
                        ]
                    }
                    '''
            return MockResponse()
    
    # Set mock LLM client
    meta_learner.llm_client = MockLLMClient()
    
    # Start meta learner
    await meta_learner.start()
    
    logger.info("=== Running Meta Learner Analysis ===")
    
    # Run analysis cycle for a system
    try:
        report = await meta_learner.run_analysis_cycle(
            system_id="web_service",
            apply_changes=True  # Apply approved proposals
        )
        
        logger.info(f"Analysis complete. Report ID: {report.report_id}")
        logger.info(f"Focus areas identified: {len(report.focus_areas)}")
        logger.info(f"Proposals generated: {report.proposals_generated}")
        logger.info(f"Proposals applied: {report.proposals_applied}")
        
        # Show updated thresholds
        updated_config = controller.get_config()
        logger.info(f"Updated CPU High: {updated_config.get_threshold_value('cpu_high')}")
        logger.info(f"Updated Response Time Warning: {updated_config.get_threshold_value('response_time_warning_ms')}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
    
    # Stop meta learner
    await meta_learner.stop()
    
    logger.info("=== Meta Learner Analysis Complete ===")


if __name__ == "__main__":
    # Run demonstrations
    print("Running Adaptive Controller Integration Demonstrations")
    print("=" * 60)
    
    # Demonstrate threshold evolution
    asyncio.run(demonstrate_threshold_evolution())
    
    print("\n" + "=" * 60)
    
    # Demonstrate meta learner analysis (requires mock setup)
    # asyncio.run(demonstrate_meta_learner_analysis())
    
    print("\nAll demonstrations complete!")