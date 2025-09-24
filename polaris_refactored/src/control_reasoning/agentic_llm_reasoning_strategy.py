"""
Agentic LLM Reasoning Strategy

Implements an intelligent reasoning strategy that uses LLM agents with dynamic tool usage
to perform sophisticated analysis and planning for system adaptation.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from .reasoning_engine import ReasoningStrategy, ReasoningContext, ReasoningResult
from ..infrastructure.llm.client import LLMClient
from ..infrastructure.llm.models import (
    LLMRequest, Message, MessageRole, AgenticConversation, ToolCall
)
from ..infrastructure.llm.tool_registry import ToolRegistry
from ..infrastructure.llm.conversation_manager import ConversationManager
from ..infrastructure.llm.prompt_manager import PromptManager
from ..infrastructure.llm.exceptions import LLMAPIError, LLMToolError
from ..infrastructure.observability.factory import get_control_logger
from .agentic_tools import create_agentic_tool_registry
from ..digital_twin.world_model import PolarisWorldModel
from ..digital_twin.knowledge_base import PolarisKnowledgeBase


class AgenticLLMReasoningStrategy(ReasoningStrategy):
    """
    Agentic LLM reasoning strategy that uses dynamic tool invocation for intelligent analysis.
    
    This strategy implements an agentic approach where the LLM can dynamically query
    the world model, knowledge base, and system state to build comprehensive context
    and make informed reasoning decisions.
    
    Key Features:
    - Multi-turn conversation management for iterative reasoning
    - Dynamic tool invocation based on reasoning needs
    - Context building from multiple framework components
    - Reasoning trace logging for transparency
    - Fallback to simpler strategies on failure
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        world_model: PolarisWorldModel,
        knowledge_base: PolarisKnowledgeBase,
        max_iterations: int = 10,
        confidence_threshold: float = 0.7
    ):
        self.llm_client = llm_client
        self.world_model = world_model
        self.knowledge_base = knowledge_base
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        
        # Create tool registry with agentic tools
        self.tool_registry = create_agentic_tool_registry(world_model, knowledge_base)
        
        # Initialize conversation and prompt managers
        self.conversation_manager = ConversationManager()
        self.prompt_manager = PromptManager()
        
        # Setup logging
        self.logger = get_control_logger("agentic_llm_reasoning")
        
        # Initialize prompts
        self._setup_prompts()
    
    def _setup_prompts(self) -> None:
        """Setup prompt templates for agentic reasoning."""
        
        # System prompt for the agentic reasoning assistant
        system_prompt = """You are an intelligent system adaptation reasoning agent for the POLARIS framework. Your role is to analyze system states, identify issues, and recommend appropriate adaptation actions.

You have access to several tools that allow you to:
1. Query the world model for predictions and simulations
2. Retrieve historical patterns and adaptation history from the knowledge base
3. Get current system state and health information
4. Validate potential adaptation actions for feasibility and risk

Your reasoning process should be:
1. Understand the current situation from the provided context
2. Use tools to gather additional information as needed
3. Analyze patterns and trends to identify root causes
4. Consider potential adaptation actions and their impacts
5. Provide clear recommendations with confidence levels

Always explain your reasoning process and cite the information sources you used. Be thorough but efficient in your tool usage."""

        # Initial analysis prompt
        analysis_prompt = """Analyze the following system situation and provide adaptation recommendations:

System ID: {system_id}
Current State: {current_state}
Historical Data: {historical_data}
System Relationships: {system_relationships}

Please analyze this situation step by step:
1. First, get the current system state and recent trends
2. Look for similar patterns in historical data
3. If issues are detected, predict future behavior and simulate potential actions
4. Validate any recommended actions for feasibility and risk
5. Provide final recommendations with confidence levels

Use the available tools to gather information and build a comprehensive understanding of the situation."""

        # Register prompts
        self.prompt_manager.add_template("system_prompt", system_prompt)
        self.prompt_manager.add_template("analysis_prompt", analysis_prompt)
    
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """
        Perform agentic reasoning using LLM with dynamic tool invocation.
        
        Args:
            context: Reasoning context with system information
            
        Returns:
            ReasoningResult with insights and recommendations
        """
        try:
            # Create new conversation for this reasoning session
            conversation = AgenticConversation(
                max_iterations=self.max_iterations,
                context={
                    "system_id": context.system_id,
                    "reasoning_start": datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Build initial context and start reasoning
            initial_message = self._build_initial_message(context)
            conversation.add_message(initial_message)
            
            # Perform iterative reasoning with tool usage
            final_result = await self._perform_iterative_reasoning(conversation)
            
            # Extract insights and recommendations from the final result
            return self._extract_reasoning_result(final_result, conversation)
            
        except Exception as e:
            self.logger.error(f"Agentic reasoning failed: {str(e)}")
            
            # Return basic fallback result
            return ReasoningResult(
                insights=[{
                    "type": "agentic_reasoning_error",
                    "message": f"LLM reasoning failed: {str(e)}",
                    "fallback": True
                }],
                confidence=0.1,
                recommendations=[]
            )
    
    def _build_initial_message(self, context: ReasoningContext) -> Message:
        """Build the initial message for the reasoning conversation."""
        
        # Format context data for the prompt
        prompt_data = {
            "system_id": context.system_id,
            "current_state": self._format_state_for_prompt(context.current_state),
            "historical_data": self._format_historical_data(context.historical_data),
            "system_relationships": self._format_relationships(context.system_relationships)
        }
        
        # Render the analysis prompt
        analysis_content = self.prompt_manager.render_template("analysis_prompt", **prompt_data)
        
        return Message(
            role=MessageRole.USER,
            content=analysis_content,
            metadata={"reasoning_context": context.system_id}
        )
    
    def _format_state_for_prompt(self, state: Dict[str, Any]) -> str:
        """Format current state data for inclusion in prompts."""
        if not state:
            return "No current state data available"
        
        formatted_parts = []
        
        # Format metrics
        metrics = state.get("metrics", {})
        if metrics:
            formatted_parts.append("Metrics:")
            for name, value in metrics.items():
                if hasattr(value, 'value'):
                    formatted_parts.append(f"  - {name}: {value.value} {getattr(value, 'unit', '')}")
                else:
                    formatted_parts.append(f"  - {name}: {value}")
        
        # Format health status
        health_status = state.get("health_status")
        if health_status:
            formatted_parts.append(f"Health Status: {health_status}")
        
        # Format other metadata
        for key, value in state.items():
            if key not in ["metrics", "health_status"]:
                formatted_parts.append(f"{key}: {value}")
        
        return "\n".join(formatted_parts) if formatted_parts else "No state information available"
    
    def _format_historical_data(self, historical_data: List[Dict[str, Any]]) -> str:
        """Format historical data for inclusion in prompts."""
        if not historical_data:
            return "No historical data available"
        
        return f"Historical data points: {len(historical_data)} entries (most recent data available)"
    
    def _format_relationships(self, relationships: Dict[str, Any]) -> str:
        """Format system relationships for inclusion in prompts."""
        if not relationships:
            return "No system relationship data available"
        
        return f"System relationships: {len(relationships)} dependencies defined"
    
    async def _perform_iterative_reasoning(self, conversation: AgenticConversation) -> Dict[str, Any]:
        """
        Perform iterative reasoning with the LLM agent using tool calls.
        
        Args:
            conversation: The conversation context to manage
            
        Returns:
            Final reasoning result
        """
        while not conversation.is_complete and conversation.current_iteration < conversation.max_iterations:
            try:
                # Prepare messages for LLM request
                messages = self._prepare_messages_for_llm(conversation)
                
                # Get available tools in the correct format
                tools = self.tool_registry.get_tool_schemas("openai")
                
                # Create LLM request
                request = LLMRequest(
                    messages=messages,
                    model_name=self.llm_client.config.model_name,
                    max_tokens=self.llm_client.config.max_tokens,
                    temperature=self.llm_client.config.temperature,
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else None
                )
                
                # Get LLM response
                response = await self.llm_client.generate_response(request)
                
                # Add LLM response to conversation
                assistant_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                    metadata={"response_id": response.response_id}
                )
                conversation.add_message(assistant_message)
                
                # Process any function calls
                if response.function_calls:
                    await self._process_function_calls(response.function_calls, conversation)
                else:
                    # No more tool calls, reasoning is complete
                    conversation.complete({
                        "final_response": response.content,
                        "reasoning_trace": conversation.reasoning_trace,
                        "tool_calls_made": len(conversation.tool_calls)
                    })
                
                conversation.increment_iteration()
                
            except Exception as e:
                self.logger.error(f"Error in reasoning iteration {conversation.current_iteration}: {str(e)}")
                conversation.complete({
                    "error": str(e),
                    "partial_result": True,
                    "completed_iterations": conversation.current_iteration
                })
                break
        
        # Return final result or timeout result
        if not conversation.is_complete:
            conversation.complete({
                "timeout": True,
                "max_iterations_reached": True,
                "completed_iterations": conversation.current_iteration
            })
        
        return conversation.final_result or {}
    
    def _prepare_messages_for_llm(self, conversation: AgenticConversation) -> List[Message]:
        """Prepare messages for LLM request, including system prompt."""
        messages = []
        
        # Add system prompt
        system_prompt = self.prompt_manager.render_template("system_prompt")
        messages.append(Message(
            role=MessageRole.SYSTEM,
            content=system_prompt
        ))
        
        # Add conversation messages
        messages.extend(conversation.messages)
        
        return messages
    
    async def _process_function_calls(
        self, 
        function_calls: List['FunctionCall'], 
        conversation: AgenticConversation
    ) -> None:
        """Process function calls from LLM response."""
        
        for func_call in function_calls:
            try:
                # Convert FunctionCall to ToolCall
                tool_call = ToolCall(
                    tool_name=func_call.name,
                    parameters=func_call.arguments,
                    call_id=func_call.call_id
                )
                
                # Execute the tool
                tool_result = await self.tool_registry.execute_tool_call(tool_call)
                
                # Add tool call and result to conversation
                conversation.add_tool_call(tool_call)
                
                # Add reasoning step
                conversation.add_reasoning_step(
                    f"Used tool {func_call.name} with parameters {func_call.arguments}"
                )
                
                # Create tool response message
                tool_message = Message(
                    role=MessageRole.TOOL,
                    content=json.dumps(tool_result.result) if tool_result.success else f"Tool error: {tool_result.error_message}",
                    tool_call_id=func_call.call_id,
                    metadata={
                        "tool_name": func_call.name,
                        "success": tool_result.success,
                        "execution_time": tool_result.execution_time
                    }
                )
                conversation.add_message(tool_message)
                
            except Exception as e:
                self.logger.error(f"Error processing function call {func_call.name}: {str(e)}")
                
                # Add error message to conversation
                error_message = Message(
                    role=MessageRole.TOOL,
                    content=f"Tool execution failed: {str(e)}",
                    tool_call_id=func_call.call_id,
                    metadata={"error": True}
                )
                conversation.add_message(error_message)
    
    def _extract_reasoning_result(
        self, 
        final_result: Dict[str, Any], 
        conversation: AgenticConversation
    ) -> ReasoningResult:
        """Extract ReasoningResult from the final conversation result."""
        
        insights = []
        recommendations = []
        confidence = 0.5  # Default confidence
        
        # Extract insights from reasoning trace and tool results
        insights.append({
            "type": "agentic_reasoning_summary",
            "iterations_completed": conversation.current_iteration,
            "tools_used": len(conversation.tool_calls),
            "reasoning_steps": len(conversation.reasoning_trace)
        })
        
        # Add tool usage insights
        tool_usage = {}
        for tool_call in conversation.tool_calls:
            tool_name = tool_call.tool_name
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        
        if tool_usage:
            insights.append({
                "type": "tool_usage_summary",
                "tools_used": tool_usage
            })
        
        # Try to extract structured recommendations from final response
        final_response = final_result.get("final_response", "")
        if final_response:
            insights.append({
                "type": "llm_analysis",
                "content": final_response[:500] + "..." if len(final_response) > 500 else final_response
            })
            
            # Simple heuristic to extract confidence from response
            if "high confidence" in final_response.lower():
                confidence = 0.8
            elif "medium confidence" in final_response.lower():
                confidence = 0.6
            elif "low confidence" in final_response.lower():
                confidence = 0.3
            elif "confident" in final_response.lower():
                confidence = 0.7
        
        # Extract recommendations from tool results
        for tool_call in conversation.tool_calls:
            if tool_call.tool_name == "action_validation" and tool_call.result:
                # Extract action recommendations from validation results
                result_data = tool_call.result
                if isinstance(result_data, dict):
                    action_data = result_data.get("action")
                    if action_data:
                        recommendations.append({
                            "action_type": action_data.get("action_type"),
                            "parameters": action_data.get("parameters", {}),
                            "source": "agentic_reasoning",
                            "validation_performed": True
                        })
        
        # If no specific recommendations found, create generic ones based on analysis
        if not recommendations and "scale" in final_response.lower():
            if "scale out" in final_response.lower() or "scale up" in final_response.lower():
                recommendations.append({
                    "action_type": "scale_out",
                    "parameters": {"scale_factor": 2.0},
                    "source": "agentic_reasoning",
                    "confidence": confidence
                })
            elif "scale in" in final_response.lower() or "scale down" in final_response.lower():
                recommendations.append({
                    "action_type": "scale_in",
                    "parameters": {"scale_factor": 0.5},
                    "source": "agentic_reasoning",
                    "confidence": confidence
                })
        
        # Adjust confidence based on successful tool usage
        if len(conversation.tool_calls) > 0:
            successful_tools = sum(1 for tc in conversation.tool_calls if tc.result and not tc.error_message)
            tool_success_rate = successful_tools / len(conversation.tool_calls)
            confidence = min(0.9, confidence + (tool_success_rate * 0.2))
        
        return ReasoningResult(
            insights=insights,
            confidence=confidence,
            recommendations=recommendations
        )