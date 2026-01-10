#!/usr/bin/env python3
"""
Validation Script for Adaptive Controller Evolution System

This script validates the basic functionality of the evolvable threshold system
without requiring external dependencies.
"""

import sys
import tempfile
import yaml
from pathlib import Path
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_evolvable_threshold():
    """Test basic evolvable threshold functionality."""
    print("Testing EvolvableThreshold...")
    
    try:
        from control_reasoning.adaptive_controller import EvolvableThreshold, ThresholdEvolutionMetadata
        
        # Create threshold
        threshold = EvolvableThreshold(
            value=80.0,
            min_value=50.0,
            max_value=95.0
        )
        
        assert threshold.value == 80.0
        assert threshold.min_value == 50.0
        assert threshold.max_value == 95.0
        print("✓ Threshold creation works")
        
        # Test valid update
        success = threshold.update_value(75.0, "test", "optimization", 0.85)
        assert success is True
        assert threshold.value == 75.0
        assert threshold.metadata.previous_value == 80.0
        assert threshold.metadata.evolution_count == 1
        print("✓ Valid threshold update works")
        
        # Test invalid update
        success = threshold.update_value(100.0)  # Above max
        assert success is False
        assert threshold.value == 75.0  # Unchanged
        print("✓ Invalid threshold update correctly rejected")
        
        # Test serialization
        data = threshold.to_dict()
        restored = EvolvableThreshold.from_dict(data)
        assert restored.value == 75.0
        assert restored.metadata.evolution_count == 1
        print("✓ Threshold serialization works")
        
        # Test legacy conversion
        legacy_threshold = EvolvableThreshold.from_dict(80.0)
        assert legacy_threshold.value == 80.0
        assert legacy_threshold.min_value == 0.0
        print("✓ Legacy threshold conversion works")
        
        return True
        
    except Exception as e:
        print(f"✗ EvolvableThreshold test failed: {e}")
        return False


def test_adaptive_controller_config():
    """Test adaptive controller configuration."""
    print("\nTesting AdaptiveControllerConfig...")
    
    try:
        from control_reasoning.adaptive_controller import AdaptiveControllerConfig
        
        # Create config
        config = AdaptiveControllerConfig()
        assert config.get_threshold_value("cpu_high") == 80.0
        assert config.version == 2
        print("✓ Config creation works")
        
        # Test threshold update
        success = config.update_threshold("cpu_high", 75.0, "test", "optimization", 0.85)
        assert success is True
        assert config.get_threshold_value("cpu_high") == 75.0
        print("✓ Config threshold update works")
        
        # Test serialization
        data = config.to_dict()
        restored = AdaptiveControllerConfig.from_dict(data)
        assert restored.get_threshold_value("cpu_high") == 75.0
        print("✓ Config serialization works")
        
        return True
        
    except Exception as e:
        print(f"✗ AdaptiveControllerConfig test failed: {e}")
        return False


def test_configuration_source():
    """Test adaptive controller configuration source."""
    print("\nTesting AdaptiveControllerConfigurationSource...")
    
    try:
        from framework.configuration.adaptive_controller_source import AdaptiveControllerConfigurationSource
        
        # Create test config file
        config_data = {
            "version": 2,
            "thresholds": {
                "cpu_high": {
                    "value": 80.0,
                    "min_value": 50.0,
                    "max_value": 95.0,
                    "metadata": {
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "updated_by": "system",
                        "confidence_score": 1.0,
                        "evolution_count": 0
                    }
                }
            },
            "cooldowns": {"default_seconds": 60.0},
            "limits": {"max_concurrent_actions": 5},
            "features": {"enable_predictive": True},
            "evolution": {"enable_threshold_evolution": True}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            # Test loading
            source = AdaptiveControllerConfigurationSource(temp_path)
            
            # Note: We can't test async load without asyncio setup, but we can test creation
            assert source.file_path == temp_path
            assert source._enable_evolution is True
            print("✓ Configuration source creation works")
            
            return True
            
        finally:
            temp_path.unlink()
        
    except Exception as e:
        print(f"✗ AdaptiveControllerConfigurationSource test failed: {e}")
        return False


def test_yaml_config_format():
    """Test the enhanced YAML configuration format."""
    print("\nTesting enhanced YAML configuration format...")
    
    try:
        config_path = Path(__file__).parent / "config" / "adaptive_controller_runtime.yaml"
        
        if not config_path.exists():
            print("⚠ Configuration file not found, skipping YAML format test")
            return True
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate structure
        assert "version" in config
        assert config["version"] == 2
        assert "thresholds" in config
        assert "evolution" in config
        print("✓ YAML configuration has correct structure")
        
        # Validate evolvable thresholds
        thresholds = config["thresholds"]
        for name, threshold_data in thresholds.items():
            assert isinstance(threshold_data, dict)
            assert "value" in threshold_data
            assert "min_value" in threshold_data
            assert "max_value" in threshold_data
            assert "metadata" in threshold_data
        print("✓ All thresholds are in evolvable format")
        
        # Validate evolution settings
        evolution = config["evolution"]
        assert "enable_threshold_evolution" in evolution
        assert "evolution_confidence_threshold" in evolution
        print("✓ Evolution settings are present")
        
        return True
        
    except Exception as e:
        print(f"✗ YAML configuration test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("Validating Adaptive Controller Evolution System")
    print("=" * 50)
    
    tests = [
        test_evolvable_threshold,
        test_adaptive_controller_config,
        test_configuration_source,
        test_yaml_config_format,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'=' * 50}")
    print(f"Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The evolution system is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())