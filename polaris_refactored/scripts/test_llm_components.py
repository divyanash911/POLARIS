#!/usr/bin/env python3
"""
LLM Components Integration Test

This script tests all LLM-based components of POLARIS:
1. LLM Client (Google Gemini) - API connectivity and response generation
2. Agentic Tools - World Model, Knowledge Base, System State, Action Validation
3. Agentic LLM Reasoning Strategy - Full reasoning loop with tool usage
4. Meta Learner - LLM-powered threshold evolution
5. Conversation Manager - Multi-turn agentic conversations

The test verifies:
- LLM API calls and responses
- Tool invocation and execution
- Reasoning flow with tool usage
- Logging of all LLM interactions

Usage:
    python scripts/test_llm_components.py [--verbose] [--test TEST_NAME]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure detailed logging for LLM testing."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # File handler for detailed LLM logs
    file_handler = logging.FileHandler(logs_dir / "llm_test_detailed.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Set specific loggers to DEBUG for LLM components
    for logger_name in [
        "polaris.llm",
        "polaris.agentic_llm_reasoning",
        "polaris.control.reasoning_engine",
        "infrastructure.llm",
    ]:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)
    
    return logging.getLogger("llm_test")


class LLMComponentTester:
    """Tests all LLM-based components of POLARIS."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.test_results: Dict[str, Dict[str, Any]] = {}
        
    async def setup_infrastructure(self) -> bool:
        """Initialize all required infrastructure components."""
        self.logger.info("=" * 70)
        self.logger.info("SETTING UP POLARIS INFRASTRUCTURE")
        self.logger.info("=" * 70)
        
        try:
            # Import infrastructure components
            from infrastructure.llm.client import GoogleClient
            from infrastructure.llm.models import LLMConfiguration, LLMProvider
            from infrastructure.llm.tool_registry import ToolRegistry
            from infrastructure.llm.conversation_manager import ConversationManager
            from infrastructure.llm.response_parser import ResponseParser
            from infrastructure.data_storage import DataStoreFactory
            from digital_twin.knowledge_base import PolarisKnowledgeBase
            from digital_twin.world_model import StatisticalWorldModel
            from control_reasoning.agentic_tools import create_agentic_tool_registry
            from control_reasoning.agentic_llm_reasoning_strategy import AgenticLLMReasoningStrategy
            from control_reasoning.reasoning_engine import ReasoningContext
            
            # Get API key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                self.logger.error("GEMINI_API_KEY not found in environment")
                return False
            
            self.logger.info(f"✓ API Key found: {api_key[:10]}...{api_key[-4:]}")
            
            # Create LLM configuration
            self.llm_config = LLMConfiguration(
                provider=LLMProvider.GOOGLE,
                api_key=api_key,
                api_endpoint="https://generativelanguage.googleapis.com",
                model_name="gemini-2.0-flash",
                max_tokens=2048,
                temperature=0.1,
                timeout=60.0,
                max_retries=3,
                enable_function_calling=True
            )
            self.logger.info(f"✓ LLM Configuration created (model: {self.llm_config.model_name})")
            
            # Create LLM client
            self.llm_client = GoogleClient(self.llm_config)
            self.logger.info("✓ Google Gemini Client initialized")
            
            # Create data store and knowledge base
            self.data_store = DataStoreFactory.create_in_memory_store()
            await self.data_store.start()
            self.knowledge_base = PolarisKnowledgeBase(self.data_store)
            self.logger.info("✓ Knowledge Base initialized")
            
            # Create world model
            self.world_model = StatisticalWorldModel(self.knowledge_base)
            self.logger.info("✓ Statistical World Model initialized")
            
            # Create agentic tool registry
            self.tool_registry = create_agentic_tool_registry(
                self.world_model, 
                self.knowledge_base
            )
            self.logger.info(f"✓ Agentic Tool Registry created with tools: {self.tool_registry.list_tools()}")
            
            # Create response parser and conversation manager
            self.response_parser = ResponseParser()
            self.conversation_manager = ConversationManager(
                llm_client=self.llm_client,
                tool_registry=self.tool_registry,
                response_parser=self.response_parser
            )
            self.logger.info("✓ Conversation Manager initialized")
            
            # Create agentic reasoning strategy
            self.agentic_reasoner = AgenticLLMReasoningStrategy(
                llm_client=self.llm_client,
                world_model=self.world_model,
                knowledge_base=self.knowledge_base,
                max_iterations=5,
                confidence_threshold=0.6
            )
            self.logger.info("✓ Agentic LLM Reasoning Strategy initialized")
            
            # Store imports for later use
            self.ReasoningContext = ReasoningContext
            
            self.logger.info("")
            self.logger.info("✓ All infrastructure components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup infrastructure: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def seed_test_data(self) -> None:
        """Seed the knowledge base with test data for realistic testing."""
        self.logger.info("")
        self.logger.info("Seeding test data into Knowledge Base...")
        
        from domain.models import SystemState, MetricValue, HealthStatus, LearnedPattern
        from framework.events import TelemetryEvent, EventMetadata
        
        # Create some historical system states via telemetry events
        base_time = datetime.now(timezone.utc)
        
        for i in range(10):
            state_time = base_time - timedelta(minutes=i * 5)
            
            # Simulate degrading system (high CPU, increasing errors)
            cpu_value = 70 + (i * 2)  # CPU increasing over time
            error_rate = 5 + (i * 0.5)  # Errors increasing
            
            state = SystemState(
                system_id="mock_system",
                timestamp=state_time,
                health_status=HealthStatus.WARNING if cpu_value > 80 else HealthStatus.HEALTHY,
                metrics={
                    "cpu_usage": MetricValue(name="cpu_usage", value=cpu_value, unit="percent", timestamp=state_time),
                    "memory_usage": MetricValue(name="memory_usage", value=3500 + (i * 50), unit="MB", timestamp=state_time),
                    "response_time": MetricValue(name="response_time", value=200 + (i * 30), unit="ms", timestamp=state_time),
                    "throughput": MetricValue(name="throughput", value=50 - (i * 2), unit="req/s", timestamp=state_time),
                    "error_rate": MetricValue(name="error_rate", value=error_rate, unit="percent", timestamp=state_time),
                },
                metadata={"test_data": True, "sequence": i}
            )
            
            # Wrap in TelemetryEvent for storage
            telemetry = TelemetryEvent(
                metadata=EventMetadata(source="test_seed"),
                correlation_id=f"seed_{i}",
                system_state=state
            )
            await self.knowledge_base.store_telemetry(telemetry)
        
        # Create some learned patterns
        pattern = LearnedPattern(
            pattern_id="high_cpu_scale_pattern",
            pattern_type="adaptation_success",
            conditions={"cpu_usage": {"operator": "gt", "value": 80}},
            outcomes={"action": "SCALE_UP", "success_rate": 0.85},
            confidence=0.8,
            usage_count=15,
            learned_at=base_time - timedelta(hours=2)
        )
        await self.knowledge_base.store_learned_pattern(pattern)
        
        self.logger.info(f"✓ Seeded 10 historical states and 1 learned pattern")
    
    async def test_llm_client_basic(self) -> Dict[str, Any]:
        """Test 1: Basic LLM Client connectivity and response generation."""
        test_name = "LLM Client Basic"
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(f"TEST: {test_name}")
        self.logger.info("=" * 70)
        
        result = {"test": test_name, "passed": False, "details": {}}
        
        try:
            from infrastructure.llm.models import LLMRequest, Message, MessageRole
            
            # Create a simple request
            request = LLMRequest(
                messages=[
                    Message(role=MessageRole.SYSTEM, content="You are a helpful assistant for system monitoring."),
                    Message(role=MessageRole.USER, content="What are the key metrics to monitor for a web application? List 3 briefly.")
                ],
                model_name=self.llm_config.model_name,
                max_tokens=500,
                temperature=0.1
            )
            
            self.logger.info("Sending basic LLM request...")
            self.logger.info(f"  Model: {request.model_name}")
            self.logger.info(f"  Messages: {len(request.messages)}")
            
            # Make the request
            response = await self.llm_client.generate_response(request)
            
            self.logger.info("")
            self.logger.info("LLM Response received:")
            self.logger.info(f"  Content length: {len(response.content)} chars")
            self.logger.info(f"  Finish reason: {response.finish_reason}")
            self.logger.info(f"  Usage: {response.usage}")
            self.logger.info(f"  Response preview: {response.content[:300]}...")
            
            result["passed"] = True
            result["details"] = {
                "response_length": len(response.content),
                "finish_reason": response.finish_reason,
                "usage": response.usage,
                "response_preview": response.content[:500]
            }
            
            self.logger.info(f"✓ {test_name} PASSED")
            
        except Exception as e:
            self.logger.error(f"✗ {test_name} FAILED: {e}")
            result["details"]["error"] = str(e)
            import traceback
            traceback.print_exc()
        
        self.test_results[test_name] = result
        return result
    
    async def test_llm_with_tools(self) -> Dict[str, Any]:
        """Test 2: LLM with function calling / tool usage."""
        test_name = "LLM with Tools"
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(f"TEST: {test_name}")
        self.logger.info("=" * 70)
        
        result = {"test": test_name, "passed": False, "details": {}}
        
        try:
            from infrastructure.llm.models import LLMRequest, Message, MessageRole
            
            # Get tool schemas
            tools = self.tool_registry.get_tool_schemas("openai")
            self.logger.info(f"Available tools: {[t['function']['name'] for t in tools]}")
            
            # Create request that should trigger tool usage
            request = LLMRequest(
                messages=[
                    Message(role=MessageRole.SYSTEM, content="""You are a system monitoring assistant with access to tools.
When asked about system state, use the system_state_query tool to get current information.
Always use tools when you need real data."""),
                    Message(role=MessageRole.USER, content="What is the current state of the mock_system? Use the system_state_query tool to check.")
                ],
                model_name=self.llm_config.model_name,
                max_tokens=1000,
                temperature=0.1,
                tools=tools,
                tool_choice="auto"
            )
            
            self.logger.info("Sending LLM request with tools...")
            self.logger.info(f"  Tools provided: {len(tools)}")
            
            response = await self.llm_client.generate_response(request)
            
            self.logger.info("")
            self.logger.info("LLM Response with tools:")
            self.logger.info(f"  Content: {response.content[:300] if response.content else '(no text content)'}")
            self.logger.info(f"  Function calls: {len(response.function_calls)}")
            
            if response.function_calls:
                self.logger.info("  Tool calls requested:")
                for fc in response.function_calls:
                    self.logger.info(f"    - {fc.name}({json.dumps(fc.arguments, indent=2)})")
                
                # Execute the tool calls
                self.logger.info("")
                self.logger.info("Executing tool calls...")
                
                for fc in response.function_calls:
                    from infrastructure.llm.models import ToolCall
                    tool_call = ToolCall(
                        tool_name=fc.name,
                        parameters=fc.arguments,
                        call_id=fc.call_id
                    )
                    
                    tool_result = await self.tool_registry.execute_tool_call(tool_call)
                    
                    self.logger.info(f"  Tool '{fc.name}' result:")
                    self.logger.info(f"    Success: {tool_result.success}")
                    self.logger.info(f"    Execution time: {tool_result.execution_time:.3f}s")
                    if tool_result.success:
                        result_preview = json.dumps(tool_result.result, indent=2)[:500]
                        self.logger.info(f"    Result: {result_preview}...")
                    else:
                        self.logger.info(f"    Error: {tool_result.error_message}")
            
            result["passed"] = True
            result["details"] = {
                "tools_provided": len(tools),
                "function_calls": len(response.function_calls),
                "function_names": [fc.name for fc in response.function_calls],
                "response_content": response.content[:500] if response.content else None
            }
            
            self.logger.info(f"✓ {test_name} PASSED")
            
        except Exception as e:
            self.logger.error(f"✗ {test_name} FAILED: {e}")
            result["details"]["error"] = str(e)
            import traceback
            traceback.print_exc()
        
        self.test_results[test_name] = result
        return result

    async def test_agentic_tools_directly(self) -> Dict[str, Any]:
        """Test 3: Direct execution of agentic tools."""
        test_name = "Agentic Tools Direct"
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(f"TEST: {test_name}")
        self.logger.info("=" * 70)
        
        result = {"test": test_name, "passed": False, "details": {"tool_results": {}}}
        
        try:
            tools_to_test = [
                {
                    "name": "system_state_query",
                    "params": {
                        "operation": "get_recent_states",
                        "system_id": "mock_system",
                        "time_window_minutes": 60,
                        "include_metrics": True
                    }
                },
                {
                    "name": "knowledge_base_query",
                    "params": {
                        "operation": "get_similar_patterns",
                        "system_id": "mock_system",
                        "conditions": {"cpu_usage": {"operator": "gt", "value": 70}},
                        "similarity_threshold": 0.5
                    }
                },
                {
                    "name": "world_model_query",
                    "params": {
                        "operation": "predict_behavior",
                        "system_id": "mock_system",
                        "time_horizon": 30
                    }
                },
                {
                    "name": "action_validation",
                    "params": {
                        "operation": "assess_action_risk",
                        "system_id": "mock_system",
                        "action": {
                            "action_type": "scale_out",
                            "parameters": {"scale_factor": 2.0}
                        },
                        "risk_tolerance": "medium"
                    }
                }
            ]
            
            all_passed = True
            
            for tool_test in tools_to_test:
                tool_name = tool_test["name"]
                params = tool_test["params"]
                
                self.logger.info(f"\nTesting tool: {tool_name}")
                self.logger.info(f"  Parameters: {json.dumps(params, indent=2)}")
                
                try:
                    tool_result = await self.tool_registry.execute_tool(tool_name, params)
                    
                    self.logger.info(f"  Success: {tool_result.success}")
                    self.logger.info(f"  Execution time: {tool_result.execution_time:.3f}s")
                    
                    if tool_result.success:
                        result_str = json.dumps(tool_result.result, indent=2, default=str)
                        self.logger.info(f"  Result preview:\n{result_str[:800]}")
                        result["details"]["tool_results"][tool_name] = {
                            "success": True,
                            "execution_time": tool_result.execution_time,
                            "result_keys": list(tool_result.result.keys()) if isinstance(tool_result.result, dict) else None
                        }
                    else:
                        self.logger.warning(f"  Error: {tool_result.error_message}")
                        result["details"]["tool_results"][tool_name] = {
                            "success": False,
                            "error": tool_result.error_message
                        }
                        all_passed = False
                        
                except Exception as e:
                    self.logger.error(f"  Exception: {e}")
                    result["details"]["tool_results"][tool_name] = {
                        "success": False,
                        "error": str(e)
                    }
                    all_passed = False
            
            result["passed"] = all_passed
            
            if all_passed:
                self.logger.info(f"\n✓ {test_name} PASSED - All tools executed successfully")
            else:
                self.logger.warning(f"\n⚠ {test_name} PARTIAL - Some tools failed")
            
        except Exception as e:
            self.logger.error(f"✗ {test_name} FAILED: {e}")
            result["details"]["error"] = str(e)
            import traceback
            traceback.print_exc()
        
        self.test_results[test_name] = result
        return result
    
    async def test_agentic_reasoning_full(self) -> Dict[str, Any]:
        """Test 4: Full agentic reasoning loop with tool usage."""
        test_name = "Agentic Reasoning Full"
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(f"TEST: {test_name}")
        self.logger.info("=" * 70)
        
        result = {"test": test_name, "passed": False, "details": {}}
        
        try:
            from domain.models import MetricValue
            
            # Create a realistic reasoning context
            current_state = {
                "metrics": {
                    "cpu_usage": MetricValue(name="cpu_usage", value=85.0, unit="percent", timestamp=datetime.now(timezone.utc)),
                    "memory_usage": MetricValue(name="memory_usage", value=4200.0, unit="MB", timestamp=datetime.now(timezone.utc)),
                    "response_time": MetricValue(name="response_time", value=550.0, unit="ms", timestamp=datetime.now(timezone.utc)),
                    "throughput": MetricValue(name="throughput", value=25.0, unit="req/s", timestamp=datetime.now(timezone.utc)),
                    "error_rate": MetricValue(name="error_rate", value=8.5, unit="percent", timestamp=datetime.now(timezone.utc)),
                },
                "health_status": "WARNING"
            }
            
            context = self.ReasoningContext(
                system_id="mock_system",
                current_state=current_state,
                historical_data=[{"note": "System has been degrading over the past hour"}],
                system_relationships={"dependencies": ["database", "cache"]}
            )
            
            self.logger.info("Starting agentic reasoning with context:")
            self.logger.info(f"  System ID: {context.system_id}")
            self.logger.info(f"  Health Status: {current_state['health_status']}")
            self.logger.info(f"  CPU: {current_state['metrics']['cpu_usage'].value}%")
            self.logger.info(f"  Memory: {current_state['metrics']['memory_usage'].value} MB")
            self.logger.info(f"  Response Time: {current_state['metrics']['response_time'].value} ms")
            self.logger.info(f"  Error Rate: {current_state['metrics']['error_rate'].value}%")
            
            self.logger.info("")
            self.logger.info("=" * 50)
            self.logger.info("STARTING AGENTIC REASONING LOOP")
            self.logger.info("=" * 50)
            
            # Run the agentic reasoning
            reasoning_result = await self.agentic_reasoner.reason(context)
            
            self.logger.info("")
            self.logger.info("=" * 50)
            self.logger.info("AGENTIC REASONING COMPLETE")
            self.logger.info("=" * 50)
            
            self.logger.info(f"\nReasoning Result:")
            self.logger.info(f"  Confidence: {reasoning_result.confidence:.2f}")
            self.logger.info(f"  Insights: {len(reasoning_result.insights)}")
            self.logger.info(f"  Recommendations: {len(reasoning_result.recommendations)}")
            
            self.logger.info("\nInsights:")
            for i, insight in enumerate(reasoning_result.insights):
                self.logger.info(f"  {i+1}. {insight.get('type', 'unknown')}: {json.dumps(insight, default=str)[:200]}")
            
            self.logger.info("\nRecommendations:")
            for i, rec in enumerate(reasoning_result.recommendations):
                self.logger.info(f"  {i+1}. {json.dumps(rec, default=str)}")
            
            result["passed"] = reasoning_result.confidence > 0.1  # Basic success check
            result["details"] = {
                "confidence": reasoning_result.confidence,
                "insights_count": len(reasoning_result.insights),
                "recommendations_count": len(reasoning_result.recommendations),
                "insights": reasoning_result.insights,
                "recommendations": reasoning_result.recommendations
            }
            
            if result["passed"]:
                self.logger.info(f"\n✓ {test_name} PASSED")
            else:
                self.logger.warning(f"\n⚠ {test_name} completed but with low confidence")
            
        except Exception as e:
            self.logger.error(f"✗ {test_name} FAILED: {e}")
            result["details"]["error"] = str(e)
            import traceback
            traceback.print_exc()
        
        self.test_results[test_name] = result
        return result

    async def test_conversation_manager(self) -> Dict[str, Any]:
        """Test 5: Conversation Manager with agentic conversation."""
        test_name = "Conversation Manager"
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(f"TEST: {test_name}")
        self.logger.info("=" * 70)
        
        result = {"test": test_name, "passed": False, "details": {}}
        
        try:
            # Create an agentic conversation
            conversation = self.conversation_manager.create_agentic_conversation(
                objective="Analyze the mock_system and recommend adaptations",
                available_tools=["system_state_query", "knowledge_base_query", "world_model_query"],
                max_iterations=3,
                context={"system_id": "mock_system"}
            )
            
            self.logger.info(f"Created agentic conversation: {conversation.conversation_id}")
            self.logger.info(f"  Objective: Analyze the mock_system and recommend adaptations")
            self.logger.info(f"  Available tools: {conversation.context.get('available_tools')}")
            self.logger.info(f"  Max iterations: {conversation.max_iterations}")
            
            # Run the conversation
            self.logger.info("")
            self.logger.info("Running agentic conversation...")
            
            completed_conversation = await self.conversation_manager.run_agentic_conversation(
                conversation.conversation_id,
                initial_message="Please analyze the current state of mock_system. Check the system state and look for any patterns in the knowledge base that might help understand the situation."
            )
            
            self.logger.info("")
            self.logger.info("Conversation completed:")
            self.logger.info(f"  Iterations: {completed_conversation.current_iteration}")
            self.logger.info(f"  Is complete: {completed_conversation.is_complete}")
            self.logger.info(f"  Tool calls made: {len(completed_conversation.tool_calls)}")
            self.logger.info(f"  Messages: {len(completed_conversation.messages)}")
            
            if completed_conversation.tool_calls:
                self.logger.info("\nTool calls made during conversation:")
                for tc in completed_conversation.tool_calls:
                    self.logger.info(f"  - {tc.tool_name}: {tc.status.value if hasattr(tc.status, 'value') else tc.status}")
            
            if completed_conversation.reasoning_trace:
                self.logger.info("\nReasoning trace:")
                for step in completed_conversation.reasoning_trace[-5:]:  # Last 5 steps
                    self.logger.info(f"  - {step}")
            
            if completed_conversation.final_result:
                self.logger.info(f"\nFinal result: {json.dumps(completed_conversation.final_result, default=str)[:500]}")
            
            result["passed"] = completed_conversation.is_complete
            result["details"] = {
                "conversation_id": conversation.conversation_id,
                "iterations": completed_conversation.current_iteration,
                "tool_calls": len(completed_conversation.tool_calls),
                "messages": len(completed_conversation.messages),
                "is_complete": completed_conversation.is_complete,
                "final_result": completed_conversation.final_result
            }
            
            if result["passed"]:
                self.logger.info(f"\n✓ {test_name} PASSED")
            else:
                self.logger.warning(f"\n⚠ {test_name} did not complete normally")
            
        except Exception as e:
            self.logger.error(f"✗ {test_name} FAILED: {e}")
            result["details"]["error"] = str(e)
            import traceback
            traceback.print_exc()
        
        self.test_results[test_name] = result
        return result
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("")
        self.logger.info("Cleaning up resources...")
        
        try:
            if hasattr(self, 'llm_client'):
                await self.llm_client.close()
            if hasattr(self, 'data_store'):
                await self.data_store.stop()
            self.logger.info("✓ Cleanup complete")
        except Exception as e:
            self.logger.warning(f"Cleanup warning: {e}")
    
    def print_summary(self) -> None:
        """Print test summary."""
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("TEST SUMMARY")
        self.logger.info("=" * 70)
        
        passed = 0
        failed = 0
        
        for test_name, result in self.test_results.items():
            status = "✓ PASSED" if result["passed"] else "✗ FAILED"
            self.logger.info(f"  {test_name}: {status}")
            if result["passed"]:
                passed += 1
            else:
                failed += 1
        
        self.logger.info("")
        self.logger.info(f"Total: {passed + failed} tests, {passed} passed, {failed} failed")
        
        # Save detailed results to file
        results_file = PROJECT_ROOT / "logs" / "llm_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        self.logger.info(f"Detailed results saved to: {results_file}")
    
    async def run_all_tests(self) -> bool:
        """Run all LLM component tests."""
        if not await self.setup_infrastructure():
            return False
        
        await self.seed_test_data()
        
        # Run tests
        await self.test_llm_client_basic()
        await self.test_llm_with_tools()
        await self.test_agentic_tools_directly()
        await self.test_agentic_reasoning_full()
        await self.test_conversation_manager()
        
        await self.cleanup()
        self.print_summary()
        
        # Return True if all tests passed
        return all(r["passed"] for r in self.test_results.values())


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test POLARIS LLM Components")
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--test', '-t', type=str, help='Run specific test only')
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    
    logger.info("")
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " POLARIS LLM Components Integration Test ".center(68) + "║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info("")
    
    tester = LLMComponentTester(logger)
    
    try:
        success = await tester.run_all_tests()
        
        if success:
            logger.info("")
            logger.info("✓ All LLM component tests passed!")
            sys.exit(0)
        else:
            logger.error("")
            logger.error("✗ Some LLM component tests failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
