"""
Tests for Adaptive Controller Threshold Evolution

Tests the integration between the adaptive controller, Polaris configuration
system, and meta learner for threshold evolution capabilities.
"""

import asyncio
import pytest
import tempfile
import yaml
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock

from framework.configuration.core import PolarisConfiguration
from framework.configuration.adaptive_controller_source import AdaptiveControllerConfigurationSource
from control_reasoning.adaptive_controller import (
    PolarisAdaptiveController, 
    AdaptiveControllerConfig,
    EvolvableThreshold,
    ThresholdEvolutionMetadata
)
from meta_learner.llm_meta_learner import LLMMetaLearner
from meta_learner.models import ParameterProposal, ProposalStatus


class TestEvolvableThreshold:
    """Test evolvable threshold functionality."""
    
    def test_threshold_creation(self):
        """Test creating an evolvable threshold."""
        threshold = EvolvableThreshold(
            value=80.0,
            min_value=50.0,
            max_value=95.0
        )
        
        assert threshold.value == 80.0
        assert threshold.min_value == 50.0
        assert threshold.max_value == 95.0
        assert threshold.metadata.evolution_count == 0
    
    def test_threshold_update_valid(self):
        """Test valid threshold update."""
        threshold = EvolvableThreshold(
            value=80.0,
            min_value=50.0,
            max_value=95.0
        )
        
        success = threshold.update_value(
            new_value=75.0,
            updated_by="test",
            reason="optimization",
            confidence=0.85
        )
        
        assert success is True
        assert threshold.value == 75.0
        assert threshold.metadata.previous_value == 80.0
        assert threshold.metadata.updated_by == "test"
        assert threshold.metadata.confidence_score == 0.85
        assert threshold.metadata.evolution_count == 1
    
    def test_threshold_update_invalid(self):
        """Test invalid threshold update (out of bounds)."""
        threshold = EvolvableThreshold(
            value=80.0,
            min_value=50.0,
            max_value=95.0
        )
        
        # Test below minimum
        success = threshold.update_value(new_value=40.0)
        assert success is False
        assert threshold.value == 80.0  # Unchanged
        
        # Test above maximum
        success = threshold.update_value(new_value=100.0)
        assert success is False
        assert threshold.value == 80.0  # Unchanged
    
    def test_threshold_serialization(self):
        """Test threshold serialization to/from dict."""
        threshold = EvolvableThreshold(
            value=80.0,
            min_value=50.0,
            max_value=95.0
        )
        threshold.update_value(75.0, "test", "optimization", 0.85)
        
        # Serialize
        data = threshold.to_dict()
        assert data["value"] == 75.0
        assert data["min_value"] == 50.0
        assert data["max_value"] == 95.0
        assert data["metadata"]["previous_value"] == 80.0
        
        # Deserialize
        restored = EvolvableThreshold.from_dict(data)
        assert restored.value == 75.0
        assert restored.metadata.previous_value == 80.0
        assert restored.metadata.evolution_count == 1
    
    def test_threshold_legacy_conversion(self):
        """Test conversion from legacy numeric format."""
        # Test legacy numeric value
        threshold = EvolvableThreshold.from_dict(80.0)
        assert threshold.value == 80.0
        assert threshold.min_value == 0.0
        assert threshold.max_value == 100.0


class TestAdaptiveControllerConfig:
    """Test adaptive controller configuration."""
    
    def test_config_creation(self):
        """Test creating adaptive controller config."""
        config = AdaptiveControllerConfig()
        
        assert config.get_threshold_value("cpu_high") == 80.0
        assert config.get_threshold_value("memory_high") == 85.0
        assert config.enable_threshold_evolution is True
        assert config.version == 2
    
    def test_config_threshold_update(self):
        """Test updating threshold via config."""
        config = AdaptiveControllerConfig()
        
        success = config.update_threshold(
            "cpu_high", 
            75.0, 
            "test", 
            "optimization", 
            0.85
        )
        
        assert success is True
        assert config.get_threshold_value("cpu_high") == 75.0
        assert config.cpu_high.metadata.updated_by == "test"
        assert config.cpu_high.metadata.confidence_score == 0.85
    
    def test_config_serialization(self):
        """Test config serialization to/from dict."""
        config = AdaptiveControllerConfig()
        config.update_threshold("cpu_high", 75.0, "test", "optimization")
        
        # Serialize
        data = config.to_dict()
        assert data["thresholds"]["cpu_high"]["value"] == 75.0
        assert data["version"] == 2
        
        # Deserialize
        restored = AdaptiveControllerConfig.from_dict(data)
        assert restored.get_threshold_value("cpu_high") == 75.0
        assert restored.cpu_high.metadata.updated_by == "test"


class TestAdaptiveControllerConfigurationSource:
    """Test adaptive controller configuration source."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file."""
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
        
        yield temp_path
        
        # Cleanup
        temp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_load_config(self, temp_config_file):
        """Test loading configuration from file."""
        source = AdaptiveControllerConfigurationSource(temp_config_file)
        config = await source.load()
        
        assert config["version"] == 2
        assert config["thresholds"]["cpu_high"]["value"] == 80.0
        assert "metadata" in config["thresholds"]["cpu_high"]
    
    @pytest.mark.asyncio
    async def test_legacy_conversion(self):
        """Test conversion of legacy config format."""
        # Create legacy format config
        legacy_config = {
            "version": 1,
            "thresholds": {
                "cpu_high": 80.0,  # Legacy numeric format
                "memory_high": 85.0
            },
            "cooldowns": {"default_seconds": 60.0},
            "limits": {"max_concurrent_actions": 5},
            "features": {"enable_predictive": True}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(legacy_config, f)
            temp_path = Path(f.name)
        
        try:
            source = AdaptiveControllerConfigurationSource(temp_path)
            config = await source.load()
            
            # Should be converted to evolvable format
            cpu_threshold = config["thresholds"]["cpu_high"]
            assert isinstance(cpu_threshold, dict)
            assert cpu_threshold["value"] == 80.0
            assert "min_value" in cpu_threshold
            assert "max_value" in cpu_threshold
            assert "metadata" in cpu_threshold
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_update_threshold(self, temp_config_file):
        """Test updating threshold via source."""
        source = AdaptiveControllerConfigurationSource(temp_config_file)
        
        success = await source.update_threshold(
            "cpu_high", 
            75.0, 
            "test", 
            "optimization", 
            0.85
        )
        
        assert success is True
        
        # Reload and verify
        config = await source.load()
        assert config["thresholds"]["cpu_high"]["value"] == 75.0
        assert config["thresholds"]["cpu_high"]["metadata"]["updated_by"] == "test"


class TestPolarisAdaptiveController:
    """Test Polaris adaptive controller integration."""
    
    @pytest.fixture
    def mock_polaris_config(self):
        """Create mock Polaris configuration."""
        config = Mock(spec=PolarisConfiguration)
        config.get.return_value = {
            "thresholds": {
                "cpu_high": {"value": 80.0, "min_value": 50.0, "max_value": 95.0, "metadata": {}},
                "memory_high": {"value": 85.0, "min_value": 60.0, "max_value": 95.0, "metadata": {}}
            },
            "cooldowns": {"default_seconds": 60.0},
            "limits": {"max_concurrent_actions": 5},
            "features": {"enable_predictive": True},
            "evolution": {"enable_threshold_evolution": True},
            "version": 2
        }
        config.set = Mock()
        return config
    
    def test_controller_creation_with_polaris_config(self, mock_polaris_config):
        """Test creating controller with Polaris configuration."""
        controller = PolarisAdaptiveController(
            polaris_config=mock_polaris_config,
            config_key="adaptive_controller"
        )
        
        assert controller._polaris_config == mock_polaris_config
        assert controller._config_key == "adaptive_controller"
        
        # Should load config from Polaris
        config = controller.get_config()
        assert config.get_threshold_value("cpu_high") == 80.0
    
    def test_threshold_update_integration(self, mock_polaris_config):
        """Test threshold update through controller."""
        controller = PolarisAdaptiveController(
            polaris_config=mock_polaris_config,
            config_key="adaptive_controller"
        )
        
        success = controller.update_threshold(
            "cpu_high", 
            75.0, 
            "test", 
            "optimization", 
            0.85
        )
        
        assert success is True
        assert controller.get_threshold_value("cpu_high") == 75.0
        
        # Should save to Polaris config
        mock_polaris_config.set.assert_called_once()
    
    def test_legacy_compatibility(self, mock_polaris_config):
        """Test legacy RuntimeConfig compatibility."""
        controller = PolarisAdaptiveController(
            polaris_config=mock_polaris_config,
            config_key="adaptive_controller"
        )
        
        # Legacy method should still work
        runtime_config = controller.get_runtime_config()
        assert runtime_config.cpu_high == 80.0
        assert runtime_config.memory_high == 85.0


class TestMetaLearnerIntegration:
    """Test meta learner integration with evolvable thresholds."""
    
    @pytest.fixture
    def mock_controller(self):
        """Create mock adaptive controller."""
        controller = Mock(spec=PolarisAdaptiveController)
        controller.update_threshold = Mock(return_value=True)
        controller.reload_config = Mock(return_value=True)
        return controller
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = Mock()
        response = Mock()
        response.content = '{"proposals": []}'
        client.generate_response = AsyncMock(return_value=response)
        return client
    
    @pytest.mark.asyncio
    async def test_meta_learner_threshold_update(self, mock_controller, mock_llm_client):
        """Test meta learner updating thresholds through controller."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"mode": "on_demand"}, f)
            config_path = f.name
        
        try:
            meta_learner = LLMMetaLearner(
                component_id="test",
                config_path=config_path,
                llm_client=mock_llm_client,
                adaptive_controller=mock_controller
            )
            
            # Create test proposals
            proposals = [
                ParameterProposal(
                    proposal_id="test1",
                    parameter_path="thresholds.cpu_high",
                    current_value=80.0,
                    proposed_value=75.0,
                    rationale="optimization",
                    confidence=0.85,
                    expected_impact="improved performance",
                    status=ProposalStatus.APPROVED
                )
            ]
            
            # Apply proposals
            applied = await meta_learner.apply_approved_proposals(proposals)
            
            assert len(applied) == 1
            assert applied[0].status == ProposalStatus.APPLIED
            
            # Should call controller's update_threshold method
            mock_controller.update_threshold.assert_called_once_with(
                threshold_name="cpu_high",
                new_value=75.0,
                updated_by="meta_learner:test",
                reason="optimization",
                confidence=0.85,
                performance_impact=None
            )
            
        finally:
            Path(config_path).unlink()
    
    @pytest.mark.asyncio
    async def test_meta_learner_legacy_fallback(self, mock_llm_client):
        """Test meta learner fallback to legacy method when no controller."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"mode": "on_demand"}, f)
            config_path = f.name
        
        # Create temporary controller config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "thresholds": {"cpu_high": 80.0},
                "cooldowns": {"default_seconds": 60.0}
            }, f)
            controller_config_path = Path(f.name)
        
        try:
            meta_learner = LLMMetaLearner(
                component_id="test",
                config_path=config_path,
                llm_client=mock_llm_client,
                controller_config_path=controller_config_path,
                adaptive_controller=None  # No controller reference
            )
            
            # Create test proposals
            proposals = [
                ParameterProposal(
                    proposal_id="test1",
                    parameter_path="cooldowns.default_seconds",
                    current_value=60.0,
                    proposed_value=45.0,
                    rationale="faster response",
                    confidence=0.8,
                    expected_impact="reduced latency",
                    status=ProposalStatus.APPROVED
                )
            ]
            
            # Apply proposals (should use legacy method)
            applied = await meta_learner.apply_approved_proposals(proposals)
            
            assert len(applied) == 1
            assert applied[0].status == ProposalStatus.APPLIED
            
            # Verify file was updated
            with open(controller_config_path, 'r') as f:
                updated_config = yaml.safe_load(f)
            assert updated_config["cooldowns"]["default_seconds"] == 45.0
            
        finally:
            Path(config_path).unlink()
            controller_config_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])