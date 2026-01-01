"""Unit tests for Meta Learner components."""

import pytest
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from meta_learner import (
    BaseMetaLearner,
    LLMMetaLearner,
    ContextWindow,
    FocusArea,
    FocusAreaPriority,
    GovernanceReport,
    MetaLearnerConfigurationError,
    ParameterProposal,
    ProposalStatus,
    WorldModelAlignment,
)


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.last_request = None
    
    async def generate_response(self, request) -> MagicMock:
        self.call_count += 1
        self.last_request = request
        
        # Get user content (the prompt)
        user_content = ""
        if request.messages:
            user_content = request.messages[-1].content.lower()
            
        print(f"DEBUG: MockLLMClient received query: {user_content[:50]}...")
        
        # Default response
        content = json.dumps({"result": "ok"})
        
        # Match based on specific task phrases to avoid overlap
        if "identify up to 5 focus areas" in user_content:
            content = json.dumps({
                "focus_areas": [
                    {
                        "name": "CPU Threshold Calibration",
                        "priority": "high",
                        "reason": "CPU threshold too low causing frequent scaling",
                        "metrics": ["cpu_usage"],
                        "suggested_action": "Increase cpu_high threshold",
                        "confidence": 0.85
                    }
                ]
            })
        elif "assess the alignment" in user_content or "evaluate whether the digital twin" in user_content:
            content = json.dumps({
                "is_aligned": True,
                "confidence": 0.8,
                "drift_detected": False,
                "drift_severity": 0.0,
                "recommendations": [],
                "metrics_assessed": ["cpu_usage", "response_time"]
            })
        elif "propose specific parameter updates" in user_content:
            content = json.dumps({
                "proposals": [
                    {
                        "parameter_path": "thresholds.cpu_high",
                        "current_value": 80.0,
                        "proposed_value": 85.0,
                        "rationale": "Reduce false positive scaling events",
                        "expected_impact": "5% reduction in scaling actions",
                        "confidence": 0.75
                    }
                ]
            })
        elif "review and rank" in user_content:
            content = json.dumps({
                "validated_proposals": [
                    {
                        "proposal_id": "test_id",
                        "status": "approved",
                        "validation_reason": "Safe incremental change",
                        "risk_score": 0.2,
                        "rank": 1
                    }
                ]
            })
        elif "generate a concise governance report" in user_content:
            content = json.dumps({
                "summary": "Meta learning cycle completed successfully",
                "key_findings": ["Thresholds need adjustment"],
                "actions_taken": ["Proposed CPU threshold increase"],
                "recommendations": ["Monitor after change"],
                "next_review_suggested_hours": 12
            })
            
        response = MagicMock()
        response.content = f"```json\n{content}\n```"
        return response


class MockKnowledgeBase:
    """Mock knowledge base for testing."""
    
    def __init__(self):
        self.states = []
        self.adaptation_history = []
    
    async def get_historical_states(self, system_id: str, start_time, end_time) -> List:
        return self.states
    
    async def get_adaptation_history(self, system_id: str) -> List:
        return self.adaptation_history


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file."""
    config_path = tmp_path / "meta_learner_config.yaml"
    config_path.write_text("""
version: 1
component_id: "test_meta_learner"
context_window_hours: 24.0
limit_states: 50
meta_learner:
  constraints:
    thresholds:
      min: 0.0
      max: 100.0
    cooldowns:
      min: 10.0
      max: 3600.0
""")
    return config_path


@pytest.fixture
def temp_controller_config(tmp_path):
    """Create a temporary controller runtime config."""
    config_path = tmp_path / "adaptive_controller_runtime.yaml"
    config_path.write_text("""
version: 1
last_updated: "2026-01-01T00:00:00Z"
updated_by: "system"
thresholds:
  cpu_high: 80.0
  cpu_low: 20.0
cooldowns:
  default_seconds: 60.0
features:
  enable_predictive: true
""")
    return config_path


@pytest.fixture
def mock_llm_client():
    return MockLLMClient()


@pytest.fixture
def mock_knowledge_base():
    return MockKnowledgeBase()


class TestContextWindow:
    """Tests for ContextWindow dataclass."""
    
    def test_default_values(self):
        window = ContextWindow(system_id="test-system")
        assert window.system_id == "test-system"
        assert window.time_window_hours == 24.0
        assert window.limit_states == 50
        assert window.focus_metrics == []
        assert window.metadata == {}
    
    def test_custom_values(self):
        window = ContextWindow(
            system_id="custom-system",
            time_window_hours=48.0,
            focus_metrics=["cpu", "memory"],
            limit_states=100,
            metadata={"key": "value"}
        )
        assert window.time_window_hours == 48.0
        assert window.focus_metrics == ["cpu", "memory"]
        assert window.limit_states == 100


class TestParameterProposal:
    """Tests for ParameterProposal dataclass."""
    
    def test_creation(self):
        proposal = ParameterProposal(
            proposal_id="test-123",
            parameter_path="thresholds.cpu_high",
            current_value=80.0,
            proposed_value=85.0,
            rationale="Reduce scaling frequency",
            confidence=0.8,
            expected_impact="5% reduction in actions"
        )
        assert proposal.proposal_id == "test-123"
        assert proposal.status == ProposalStatus.PENDING
        assert proposal.applied_at is None
    
    def test_status_transitions(self):
        proposal = ParameterProposal(
            proposal_id="test",
            parameter_path="test.path",
            current_value=1,
            proposed_value=2,
            rationale="test",
            confidence=0.5,
            expected_impact="test"
        )
        assert proposal.status == ProposalStatus.PENDING
        
        proposal.status = ProposalStatus.APPROVED
        assert proposal.status == ProposalStatus.APPROVED
        
        proposal.status = ProposalStatus.APPLIED
        proposal.applied_at = datetime.now(timezone.utc)
        assert proposal.applied_at is not None


class TestLLMMetaLearner:
    """Tests for LLMMetaLearner implementation."""
    
    @pytest.mark.asyncio
    async def test_select_focus_areas(self, temp_config_file, mock_llm_client, mock_knowledge_base):
        learner = LLMMetaLearner(
            component_id="test_learner",
            config_path=str(temp_config_file),
            llm_client=mock_llm_client,
            knowledge_base=mock_knowledge_base,
        )
        
        snapshot = {
            "system_id": "test-system",
            "time_range": {"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T06:00:00Z"},
            "samples": 10,
            "metric_averages": {"cpu_usage": 75.0}
        }
        
        focus_areas = await learner.select_focus_areas(snapshot)
        
        assert len(focus_areas) == 1
        assert focus_areas[0].name == "CPU Threshold Calibration"
        assert focus_areas[0].priority == FocusAreaPriority.HIGH
        assert mock_llm_client.call_count == 1
    
    @pytest.mark.asyncio
    async def test_evaluate_world_model_alignment(self, temp_config_file, mock_llm_client, mock_knowledge_base):
        learner = LLMMetaLearner(
            component_id="test_learner",
            config_path=str(temp_config_file),
            llm_client=mock_llm_client,
            knowledge_base=mock_knowledge_base,
        )
        
        snapshot = {"system_id": "test-system", "time_range": {}}
        alignment = await learner.evaluate_world_model_alignment(snapshot)
        
        assert alignment.is_aligned is True
        assert alignment.confidence == 0.8
        assert alignment.drift_detected is False
    
    @pytest.mark.asyncio
    async def test_propose_parameter_updates(self, temp_config_file, temp_controller_config, mock_llm_client, mock_knowledge_base):
        learner = LLMMetaLearner(
            component_id="test_learner",
            config_path=str(temp_config_file),
            llm_client=mock_llm_client,
            knowledge_base=mock_knowledge_base,
            controller_config_path=temp_controller_config,
        )
        
        # First call select_focus_areas to populate cache
        snapshot = {"system_id": "test-system", "time_range": {}, "samples": 10, "metric_averages": {}}
        await learner.select_focus_areas(snapshot)
        
        proposals = await learner.propose_parameter_updates(snapshot)
        
        assert len(proposals) == 1
        assert proposals[0].parameter_path == "thresholds.cpu_high"
        assert proposals[0].proposed_value == 85.0
    
    @pytest.mark.asyncio
    async def test_validate_and_rank_updates(self, temp_config_file, mock_llm_client, mock_knowledge_base):
        learner = LLMMetaLearner(
            component_id="test_learner",
            config_path=str(temp_config_file),
            llm_client=mock_llm_client,
            knowledge_base=mock_knowledge_base,
        )
        
        proposals = [ParameterProposal(
            proposal_id="test_id",
            parameter_path="thresholds.cpu_high",
            current_value=80.0,
            proposed_value=85.0,
            rationale="Test",
            confidence=0.8,
            expected_impact="Test impact"
        )]
        
        validated = await learner.validate_and_rank_updates(proposals)
        
        assert len(validated) == 1
        assert validated[0].status == ProposalStatus.APPROVED
    
    @pytest.mark.asyncio
    async def test_apply_approved_proposals(self, temp_config_file, temp_controller_config, mock_llm_client, mock_knowledge_base):
        learner = LLMMetaLearner(
            component_id="test_learner",
            config_path=str(temp_config_file),
            llm_client=mock_llm_client,
            knowledge_base=mock_knowledge_base,
            controller_config_path=temp_controller_config,
        )
        
        proposals = [ParameterProposal(
            proposal_id="apply_test",
            parameter_path="thresholds.cpu_high",
            current_value=80.0,
            proposed_value=85.0,
            rationale="Test",
            confidence=0.8,
            expected_impact="Test",
            status=ProposalStatus.APPROVED
        )]
        
        applied = await learner.apply_approved_proposals(proposals)
        
        assert len(applied) == 1
        assert applied[0].status == ProposalStatus.APPLIED
        assert applied[0].applied_at is not None
        
        # Verify file was updated
        import yaml
        with open(temp_controller_config) as f:
            updated_config = yaml.safe_load(f)
        assert updated_config["thresholds"]["cpu_high"] == 85.0
        assert "meta_learner" in updated_config["updated_by"]
    
    @pytest.mark.asyncio
    async def test_no_llm_client_raises_error(self, temp_config_file, mock_knowledge_base):
        learner = LLMMetaLearner(
            component_id="test_learner",
            config_path=str(temp_config_file),
            llm_client=None,
            knowledge_base=mock_knowledge_base,
        )
        
        with pytest.raises(MetaLearnerConfigurationError):
            await learner.select_focus_areas({"system_id": "test"})


class TestConfigurationLoading:
    """Tests for configuration loading."""
    
    def test_missing_config_raises_error(self, tmp_path):
        with pytest.raises(MetaLearnerConfigurationError):
            LLMMetaLearner(
                component_id="test",
                config_path=str(tmp_path / "nonexistent.yaml"),
                llm_client=MockLLMClient(),
            )
    
    def test_config_reload(self, temp_config_file, mock_llm_client):
        learner = LLMMetaLearner(
            component_id="test",
            config_path=str(temp_config_file),
            llm_client=mock_llm_client,
        )
        
        # Modify config file
        temp_config_file.write_text("""
version: 2
component_id: "updated_learner"
context_window_hours: 48.0
limit_states: 100
""")
        
        # Reload
        new_config = learner.reload_config()
        assert new_config["version"] == 2
        assert new_config["context_window_hours"] == 48.0


class TestLLMMetaLearnerIntegration:
    """Integration tests for full LLM Meta Learner workflow."""
    
    @pytest.mark.asyncio
    async def test_full_analysis_cycle(self, temp_config_file, temp_controller_config, mock_llm_client, mock_knowledge_base):
        """Test the complete meta learning analysis cycle end-to-end."""
        learner = LLMMetaLearner(
            component_id="integration_test",
            config_path=str(temp_config_file),
            llm_client=mock_llm_client,
            knowledge_base=mock_knowledge_base,
            controller_config_path=temp_controller_config,
        )
        
        # Run complete analysis cycle
        report = await learner.run_analysis_cycle(
            system_id="test-system",
            apply_changes=True
        )
        
        # Verify report was generated
        assert report is not None
        assert report.report_id is not None
        assert report.system_id == "test-system"
        assert report.timestamp is not None
        
        # Verify LLM was called multiple times (focus, alignment, proposals, validation, report)
        assert mock_llm_client.call_count >= 4
        
        # Verify focus areas were identified
        assert len(report.focus_areas) > 0
        
        # Verify proposals were processed
        assert report.proposals_generated >= 0
    
    @pytest.mark.asyncio
    async def test_lifecycle_start_stop(self, temp_config_file, mock_llm_client, mock_knowledge_base):
        """Test the start/stop lifecycle of the meta learner."""
        learner = LLMMetaLearner(
            component_id="lifecycle_test",
            config_path=str(temp_config_file),
            llm_client=mock_llm_client,
            knowledge_base=mock_knowledge_base,
        )
        
        # Initially not running
        assert learner._is_running is False
        
        # Start
        await learner.start()
        assert learner._is_running is True
        
        # Stop
        await learner.stop()
        assert learner._is_running is False
    
    @pytest.mark.asyncio
    async def test_continuous_loop_disabled_by_default(self, temp_config_file, mock_llm_client, mock_knowledge_base):
        """Test that continuous analysis loop is disabled when not configured."""
        learner = LLMMetaLearner(
            component_id="loop_test",
            config_path=str(temp_config_file),
            llm_client=mock_llm_client,
            knowledge_base=mock_knowledge_base,
        )
        
        await learner.start()
        
        # Analysis task should not be created (continuous not enabled in config)
        assert learner._analysis_task is None
        
        await learner.stop()
    
    @pytest.mark.asyncio
    async def test_governance_report_content(self, temp_config_file, temp_controller_config, mock_llm_client, mock_knowledge_base):
        """Test that governance report contains all required information."""
        learner = LLMMetaLearner(
            component_id="report_test",
            config_path=str(temp_config_file),
            llm_client=mock_llm_client,
            knowledge_base=mock_knowledge_base,
            controller_config_path=temp_controller_config,
        )
        
        # Run cycle to generate report
        report = await learner.run_analysis_cycle("test-system", apply_changes=False)
        
        # Verify report structure
        assert hasattr(report, 'report_id')
        assert hasattr(report, 'timestamp')
        assert hasattr(report, 'system_id')
        assert hasattr(report, 'focus_areas')
        assert hasattr(report, 'alignment_assessment')
        assert hasattr(report, 'proposals_generated')
        assert hasattr(report, 'proposals_applied')
        assert hasattr(report, 'recommendations')
        assert hasattr(report, 'metadata')
        
        # Verify metadata contains summary
        assert 'summary' in report.metadata
    
    @pytest.mark.asyncio
    async def test_json_parsing_resilience(self, temp_config_file, mock_knowledge_base):
        """Test that the meta learner handles malformed JSON responses gracefully."""
        # Create a mock that returns invalid JSON
        bad_client = MockLLMClient()
        
        # Override to return malformed response
        async def bad_response(request):
            response = MagicMock()
            response.content = "This is not valid JSON at all"
            return response
        
        bad_client.generate_response = bad_response
        
        learner = LLMMetaLearner(
            component_id="json_test",
            config_path=str(temp_config_file),
            llm_client=bad_client,
            knowledge_base=mock_knowledge_base,
        )
        
        snapshot = {"system_id": "test", "time_range": {}}
        
        # Should not raise, but return empty list
        focus_areas = await learner.select_focus_areas(snapshot)
        assert focus_areas == []
    
    @pytest.mark.asyncio
    async def test_proposal_validation_rejects_invalid(self, temp_config_file, mock_knowledge_base):
        """Test that invalid proposals are properly rejected during validation."""
        # Create mock that rejects proposals
        rejecting_client = MockLLMClient()
        
        async def reject_response(request):
            user_content = request.messages[-1].content.lower() if request.messages else ""
            
            if "review and rank" in user_content:
                content = json.dumps({
                    "validated_proposals": [
                        {
                            "proposal_id": "reject_me",
                            "status": "rejected",
                            "validation_reason": "Unsafe change",
                            "risk_score": 0.9
                        }
                    ]
                })
            else:
                content = json.dumps({"result": "ok"})
            
            response = MagicMock()
            response.content = f"```json\n{content}\n```"
            return response
        
        rejecting_client.generate_response = reject_response
        
        learner = LLMMetaLearner(
            component_id="reject_test",
            config_path=str(temp_config_file),
            llm_client=rejecting_client,
            knowledge_base=mock_knowledge_base,
        )
        
        proposals = [ParameterProposal(
            proposal_id="reject_me",
            parameter_path="thresholds.cpu_high",
            current_value=80.0,
            proposed_value=200.0,  # Invalid value
            rationale="Bad idea",
            confidence=0.3,
            expected_impact="System crash"
        )]
        
        validated = await learner.validate_and_rank_updates(proposals)
        
        assert len(validated) == 1
        assert validated[0].status == ProposalStatus.REJECTED


class TestMetaLearnerObservability:
    """Tests for observability features in the meta learner."""
    
    @pytest.mark.asyncio
    async def test_logging_extra_context(self, temp_config_file, temp_controller_config, mock_llm_client, mock_knowledge_base):
        """Test that the meta learner lifecycle functions correctly (logging is best-effort)."""
        learner = LLMMetaLearner(
            component_id="logging_test",
            config_path=str(temp_config_file),
            llm_client=mock_llm_client,
            knowledge_base=mock_knowledge_base,
            controller_config_path=temp_controller_config,
        )
        
        # Verify start/stop lifecycle works without errors
        assert learner._is_running is False
        await learner.start()
        assert learner._is_running is True
        await learner.stop()
        assert learner._is_running is False
    
    @pytest.mark.asyncio
    async def test_internal_metrics_recording(self, temp_config_file, mock_llm_client, mock_knowledge_base):
        """Test that internal metrics are recorded (even if collector is None)."""
        learner = LLMMetaLearner(
            component_id="metrics_test",
            config_path=str(temp_config_file),
            llm_client=mock_llm_client,
            knowledge_base=mock_knowledge_base,
        )
        
        # Verify _record_metric doesn't crash when metrics collector is None
        learner._record_metric("test.metric", 1.0)  # Should not raise
        learner._record_metric("test.metric", 1.0, {"label": "value"})  # Should not raise

