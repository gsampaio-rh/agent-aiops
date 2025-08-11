"""
LangGraph Agent Service

Implements a LangGraph-based agent with workflow management and tool integration.
Maintains compatibility with existing AgentServiceInterface while leveraging
LangGraph's powerful workflow capabilities.
"""

import time
import json
from typing import Dict, List, Any, Optional, Iterator, TypedDict, Annotated
from functools import partial

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
    from langchain_core.tools import BaseTool
    from langchain_core.runnables import RunnableLambda
except ImportError as e:
    raise ImportError(f"LangGraph dependencies not installed: {e}. Run: pip install -r requirements.txt")

from core.interfaces.agent_service import AgentServiceInterface, ToolInterface
from core.interfaces.llm_service import LLMServiceInterface
from core.models.agent import AgentStep, AgentResponse, StepType, ToolInfo
from core.exceptions import AgentError, AgentProcessingError, ErrorCodes
from services.ollama_service import OllamaService
from utils.logger import (
    get_logger, log_performance, log_agent_step, request_context,
    log_llm_conversation, log_agent_workflow, log_tool_execution,
    create_request_tracer, log_request_step, log_request_complete,
    log_error_with_context
)


class AgentState(TypedDict):
    """State definition for the LangGraph agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    agent_steps: List[Dict[str, Any]]
    current_step: str
    iteration_count: int
    max_iterations: int
    available_tools: List[str]


class LangChainToolAdapter(BaseTool):
    """Adapter to convert ToolInterface to LangChain Tool format."""
    
    wrapped_tool: ToolInterface
    
    def __init__(self, tool: ToolInterface):
        tool_info = tool.get_tool_info()
        super().__init__(
            name=tool_info.name,
            description=tool_info.description,
            wrapped_tool=tool
        )
    
    def _run(self, query: str, **kwargs) -> str:
        """Execute the wrapped tool and return results."""
        try:
            result = self.wrapped_tool.execute(query, **kwargs)
            if result.get("success", True):
                return str(result.get("results", result.get("output", str(result))))
            else:
                return f"Tool execution failed: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Tool execution error: {str(e)}"


class WebSearchTool(ToolInterface):
    """Web search tool for the agent."""
    
    def __init__(self):
        self.name = "web_search"
        self.description = "Search the web for current information, facts, news, and general knowledge. Use this when you need up-to-date information or when the user asks about current events, recent developments, or specific factual information that might not be in your training data."
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute web search."""
        start_time = time.time()
        provider = kwargs.get("provider", "duckduckgo")
        max_results = kwargs.get("max_results", 5)
        
        try:
            # Import here to avoid circular imports
            from services.search_service import WebSearchService
            from core.models.search import SearchQuery
            
            search_service = WebSearchService()
            
            # Create search query
            search_query = SearchQuery(
                query=query,
                provider=provider,
                max_results=max_results
            )
            
            # Use new interface method
            result = search_service.search(search_query)
            duration_ms = round((time.time() - start_time) * 1000)
            
            if result.is_successful() and result.results:
                # Use the built-in formatting method
                formatted_results = result.get_formatted_results(max_results)
                
                result_data = {
                    "success": True,
                    "results": formatted_results,
                    "metadata": {
                        "provider": result.provider_name,
                        "total_results": result.total_results,
                        "search_time_ms": result.search_time_ms
                    }
                }
                
                # Log successful tool execution
                log_tool_execution(
                    tool_name="web_search",
                    action="search",
                    query=query,
                    result=result_data,
                    duration_ms=duration_ms,
                    success=True,
                    provider=result.provider_name,
                    results_count=result.total_results
                )
                
                return result_data
            else:
                result_data = {
                    "success": False,
                    "error": result.error or "No results found",
                    "metadata": {
                        "provider": result.provider_name,
                        "total_results": result.total_results,
                        "search_time_ms": result.search_time_ms
                    }
                }
                
                # Log failed search (no results)
                log_tool_execution(
                    tool_name="web_search",
                    action="search",
                    query=query,
                    result=result_data,
                    duration_ms=duration_ms,
                    success=False,
                    provider=result.provider_name,
                    error_reason="no_results"
                )
                
                return result_data
                
        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000)
            
            result_data = {
                "success": False,
                "error": f"Search failed: {str(e)}",
                "metadata": {"exception": str(e)}
            }
            
            # Log exception in tool execution
            log_tool_execution(
                tool_name="web_search",
                action="search",
                query=query,
                result=result_data,
                duration_ms=duration_ms,
                success=False,
                provider=provider,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            
            return result_data
    
    def get_tool_info(self) -> ToolInfo:
        """Get tool information for the agent."""
        return ToolInfo(
            name=self.name,
            description=self.description
        )


class LangGraphAgent(AgentServiceInterface):
    """LangGraph-based agent with workflow management."""
    
    def __init__(self, model: str = "llama3.2:3b", 
                 llm_service: Optional[LLMServiceInterface] = None):
        self.llm_service = llm_service or OllamaService()
        self.model = model
        self.tools: Dict[str, ToolInterface] = {}
        self.langchain_tools: List[BaseTool] = []
        self.max_iterations = 10
        self.logger = get_logger(__name__)
        
        # Initialize LangGraph workflow
        self.workflow = None
        self.memory = MemorySaver()
        self._initialize_workflow()
        
        self.logger.info("Initialized LangGraphAgent", model=model)
    
    def _initialize_workflow(self):
        """Initialize the LangGraph workflow."""
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("reasoning", self._reasoning_node)
        workflow.add_node("tool_selection", self._tool_selection_node)
        workflow.add_node("tool_execution", self._tool_execution_node)
        workflow.add_node("final_answer", self._final_answer_node)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "reasoning",
            self._should_use_tool,
            {
                "use_tool": "tool_selection",
                "final_answer": "final_answer"
            }
        )
        
        workflow.add_edge("tool_selection", "tool_execution")
        
        # Add conditional edge from tool_execution to handle retries
        workflow.add_conditional_edges(
            "tool_execution",
            self._should_retry_after_error,
            {
                "retry": "reasoning",
                "continue": "final_answer"
            }
        )
        
        workflow.add_edge("final_answer", END)
        
        # Set entry point
        workflow.set_entry_point("reasoning")
        
        # Compile the workflow
        self.workflow = workflow.compile(checkpointer=self.memory)
    
    def _reasoning_node(self, state: AgentState) -> AgentState:
        """Node for agent reasoning and planning with conversation memory."""
        self.logger.debug("Executing reasoning node")
        
        # Create reasoning step
        step_data = {
            "step_type": "thought",
            "content": "Analyzing the user's request and conversation history...",
            "timestamp": time.time(),
            "metadata": {"node": "reasoning"}
        }
        
        state["agent_steps"].append(step_data)
        state["current_step"] = "reasoning"
        
        # Generate reasoning using LLM with full conversation context
        messages = state["messages"]
        
        # Create system prompt that includes memory instructions
        system_prompt = self._get_system_prompt_with_memory()
        
        # Build complete conversation history for context
        reasoning_messages = [{"role": "system", "content": system_prompt}]
        
        # Add full conversation history (excluding the current query message)
        conversation_history = self._build_conversation_history(messages)
        reasoning_messages.extend(conversation_history)
        
        # Add current user query
        reasoning_messages.append({"role": "user", "content": state["user_query"]})
        
        self.logger.debug("Built reasoning context", 
                         total_messages=len(reasoning_messages),
                         history_length=len(conversation_history))
        
        reasoning_response = ""
        for chunk in self.llm_service.chat_stream(
            model=self.model,
            messages=reasoning_messages,
            temperature=0.7,
            max_tokens=1000
        ):
            if chunk.get("content"):
                reasoning_response += chunk["content"]
        
        # Parse the reasoning response to determine next action
        step_data["content"] = reasoning_response
        state["agent_steps"][-1] = step_data
        
        return state
    
    def _tool_selection_node(self, state: AgentState) -> AgentState:
        """Node for selecting and preparing tools."""
        self.logger.debug("Executing tool selection node")
        
        step_data = {
            "step_type": "tool_selection",
            "content": f"Selecting appropriate tool from: {', '.join(state['available_tools'])}",
            "timestamp": time.time(),
            "metadata": {"node": "tool_selection", "available_tools": state["available_tools"]}
        }
        
        state["agent_steps"].append(step_data)
        state["current_step"] = "tool_selection"
        
        return state
    
    def _tool_execution_node(self, state: AgentState) -> AgentState:
        """Node for executing selected tools."""
        self.logger.debug("Executing tool execution node")
        
        # Find the tool use command from previous steps
        tool_command = None
        tool_name = None
        tool_query = None
        
        # Look through recent steps for TOOL_USE command
        for step in reversed(state["agent_steps"]):
            content = step.get("content", "")
            if "TOOL_USE:" in content:
                # Parse tool command: "TOOL_USE: tool_name: query"
                try:
                    tool_part = content.split("TOOL_USE:", 1)[1].strip()
                    
                    # More robust parsing: look for known tool names at the start
                    available_tools = list(self.tools.keys())
                    tool_found = False
                    
                    for available_tool in available_tools:
                        if tool_part.startswith(available_tool + ":"):
                            tool_name = available_tool
                            tool_query = tool_part[len(available_tool) + 1:].strip()
                            tool_command = tool_part
                            tool_found = True
                            break
                    
                    # Fallback to simple split if no known tool found
                    if not tool_found and ":" in tool_part:
                        potential_tool, potential_query = tool_part.split(":", 1)
                        potential_tool = potential_tool.strip()
                        potential_query = potential_query.strip()
                        
                        # Check if it's a valid tool name
                        if potential_tool in available_tools:
                            tool_name = potential_tool
                            tool_query = potential_query
                            tool_command = tool_part
                        else:
                            # Invalid tool name - we'll handle this below
                            tool_name = potential_tool
                            tool_query = potential_query
                            tool_command = tool_part
                    
                    break
                except Exception as e:
                    self.logger.error("Failed to parse tool command", error=str(e), content=content)
        
        if not tool_command or not tool_name:
            # No valid tool command found
            step_data = {
                "step_type": "error",
                "content": "No valid tool command found",
                "timestamp": time.time(),
                "metadata": {"node": "tool_execution", "error": "no_tool_command"}
            }
            state["agent_steps"].append(step_data)
            return state
        
        # Create tool use step
        step_data = {
            "step_type": "tool_use",
            "content": f"Executing {tool_name} with query: {tool_query}",
            "timestamp": time.time(),
            "metadata": {"node": "tool_execution", "tool": tool_name, "query": tool_query}
        }
        state["agent_steps"].append(step_data)
        
        # Check if the tool exists before proceeding
        if tool_name not in self.tools:
            # Invalid tool selected - inform the LLM to retry with correct tool
            available_tools_list = ', '.join(self.tools.keys())
            error_step = {
                "step_type": "error",
                "content": f"Tool '{tool_name}' is not available. Available tools are: {available_tools_list}. Please retry with TOOL_USE: [tool_name]: [query] format.",
                "timestamp": time.time(),
                "metadata": {
                    "node": "tool_execution",
                    "tool": tool_name,
                    "available_tools": list(self.tools.keys()),
                    "error": "invalid_tool_name",
                    "retry_needed": True
                }
            }
            state["agent_steps"].append(error_step)
            
            # Set the state to go back to reasoning to let the LLM retry
            state["current_step"] = "error_retry"
            return state
        
        # Check for duplicate requests (prevent the double request issue)
        recent_requests = [step for step in state["agent_steps"][-3:] 
                          if step.get("step_type") == "tool_execution_request" 
                          and step.get("metadata", {}).get("tool") == tool_name
                          and step.get("metadata", {}).get("query") == tool_query]
        
        if recent_requests:
            self.logger.warning(f"Duplicate tool execution request detected for {tool_name}, skipping")
            return state
        
        # Log tool execution request for UI workflow (DO NOT execute automatically)
        self.logger.info(f"Tool execution requested by agent: {tool_name} with query: {tool_query}")
        
        # Create a tool execution request step instead of executing directly
        tool_request_step = {
            "step_type": "tool_execution_request",
            "content": f"Requesting user permission to execute {tool_name} with query: {tool_query}",
            "timestamp": time.time(),
            "metadata": {
                "node": "tool_execution",
                "tool": tool_name,
                "query": tool_query,
                "requires_user_permission": True,
                "status": "pending_permission"
            }
        }
        state["agent_steps"].append(tool_request_step)
        
        # Note: Tool execution will be handled by the UI workflow after user permission
        # The agent workflow will pause here and wait for the UI to handle the tool execution
        
        state["current_step"] = "tool_execution"
        state["iteration_count"] += 1
        
        return state
    
    def _final_answer_node(self, state: AgentState) -> AgentState:
        """Node for generating final answer."""
        self.logger.debug("Executing final answer node")
        
        # Check if we have tool results to synthesize
        tool_results = []
        for step in state["agent_steps"]:
            if step.get("step_type") == "tool_result":
                tool_results.append(step.get("content", ""))
        
        if tool_results:
            # Generate response based on tool results with conversation context
            synthesis_messages = [
                {"role": "system", "content": "You are analyzing tool results. Use conversation history for context and provide a comprehensive answer based on the tool results. Reference previous conversation when relevant."}
            ]
            
            # Add conversation history for context
            conversation_history = self._build_conversation_history(state["messages"])
            synthesis_messages.extend(conversation_history)
            
            # Add tool results and query
            synthesis_messages.append({
                "role": "user", 
                "content": f"Original query: {state['user_query']}\n\nTool results:\n{chr(10).join(tool_results)}\n\nBased on these results and our conversation history, provide a comprehensive answer:"
            })
            
            final_response = ""
            for chunk in self.llm_service.chat_stream(
                model=self.model,
                messages=synthesis_messages,
                temperature=0.7,
                max_tokens=1000
            ):
                if chunk.get("content"):
                    final_response += chunk["content"]
            
            content = final_response
        else:
            # No tool results, generate direct response with conversation context
            # Build complete message history including conversation context
            direct_messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Use conversation history to provide contextual responses. Reference previous information when relevant."}
            ]
            
            # Add conversation history for context
            conversation_history = self._build_conversation_history(state["messages"])
            direct_messages.extend(conversation_history)
            
            # Add current query
            direct_messages.append({"role": "user", "content": state["user_query"]})
            
            direct_response = ""
            for chunk in self.llm_service.chat_stream(
                model=self.model,
                messages=direct_messages,
                temperature=0.7,
                max_tokens=1000
            ):
                if chunk.get("content"):
                    direct_response += chunk["content"]
            
            content = direct_response
        
        step_data = {
            "step_type": "final_answer",
            "content": content,
            "timestamp": time.time(),
            "metadata": {
                "node": "final_answer",
                "tool_results_used": len(tool_results) > 0,
                "tool_results_count": len(tool_results)
            }
        }
        
        state["agent_steps"].append(step_data)
        state["current_step"] = "final_answer"
        
        return state
    
    def _should_use_tool(self, state: AgentState) -> str:
        """Determine if a tool should be used."""
        # Check if we've hit max iterations
        if state["iteration_count"] >= state["max_iterations"]:
            return "final_answer"
        
        # Get the last step content
        last_step = state["agent_steps"][-1] if state["agent_steps"] else {}
        content = last_step.get("content", "")
        
        # Look for explicit tool usage command
        if "TOOL_USE:" in content:
            return "use_tool"
        elif "FINAL_ANSWER:" in content:
            return "final_answer"
        elif any(f"TOOL_USE: {tool_name}" in content for tool_name in state["available_tools"]):
            return "use_tool"
        else:
            # Default to final answer if no clear tool usage detected
            return "final_answer"
    
    def _should_retry_after_error(self, state: AgentState) -> str:
        """Determine if we should retry after an error or continue to final answer."""
        # Check if the current state indicates a retry is needed
        if state.get("current_step") == "error_retry":
            # Check iteration count to avoid infinite loops
            if state["iteration_count"] < state["max_iterations"]:
                return "retry"
        
        # Check if the last step was an error with retry_needed flag
        if state["agent_steps"]:
            last_step = state["agent_steps"][-1]
            if (last_step.get("step_type") == "error" and 
                last_step.get("metadata", {}).get("retry_needed") and
                state["iteration_count"] < state["max_iterations"]):
                return "retry"
        
        return "continue"
    

    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        tools_description = self.get_tools_description()
        
        return f"""You are a helpful AI assistant with access to tools. You MUST use tools when the user asks for:
- Current information, facts, news, or recent developments
- Information that might change over time
- Specific factual queries that benefit from web search
- Technical questions that need up-to-date information

Available tools:
{tools_description}

IMPORTANT: Follow this exact pattern for ALL responses:

1. THOUGHT: Think about what the user is asking and whether you need to use a tool
2. If you need a tool (for current info, facts, news, specific queries):
   - TOOL_SELECTION: Choose which tool and explain why
   - TOOL_USE: [tool_name]: [exact command or query]
3. If you don't need tools (for general knowledge, definitions, explanations):
   - FINAL_ANSWER: [your direct response]

CRITICAL RULES:
- ALWAYS use the exact tool names available: {', '.join(self.tools.keys()) if self.tools else 'No tools available'}
- For web searches: USE "web_search"
- For terminal commands: USE "terminal"
- After TOOL_USE, STOP immediately - do not add anything else
- The system will execute the tool and provide results automatically
- Use EXACT format: "TOOL_USE: [exact_tool_name]: [command_or_query]"
- Never add explanations after TOOL_USE

EXAMPLES:
- "What are the health benefits of meditation?" → TOOL_USE: web_search: health benefits of meditation
- "Check TLS certificate expiration" → TOOL_USE: terminal: echo | openssl s_client -connect localhost:8443 -servername localhost
- "What's the weather today?" → TOOL_USE: web_search: weather today
- "What is 2+2?" → FINAL_ANSWER: 4

IMPORTANT TERMINAL USAGE GUIDELINES:
- For TLS/SSL checks: Use "echo | openssl s_client" instead of interactive openssl
- For network testing: Use "curl --max-time 10" or "ping -c 4" instead of unlimited commands
- For file viewing: Use "cat" or "head -20" instead of interactive viewers like "less"
- Avoid: ssh, vim, nano, mysql, psql - these are interactive and will hang

Remember: Use tools for facts and current information. STOP after TOOL_USE."""
    
    def _get_system_prompt_with_memory(self) -> str:
        """Get the system prompt with conversation memory instructions."""
        tools_description = self.get_tools_description()
        
        return f"""You are a helpful AI assistant with access to tools and conversation memory. You can reference previous parts of our conversation to provide contextual responses.

CONVERSATION MEMORY:
- You have access to the full conversation history
- Reference previous topics, user details, and context when relevant
- Maintain continuity across the conversation
- Remember user preferences, names, and information shared earlier

You MUST use tools when the user asks for:
- Current information, facts, news, or recent developments
- Information that might change over time
- Specific factual queries that benefit from web search
- Technical questions that need up-to-date information

Available tools:
{tools_description}

IMPORTANT: Follow this exact pattern for ALL responses:

1. THOUGHT: Think about what the user is asking, reference any relevant conversation history, and determine if you need tools
2. If you need a tool (for current info, facts, news, specific queries):
   - TOOL_SELECTION: Choose which tool and explain why
   - TOOL_USE: [tool_name]: [exact command or query]
3. If you don't need tools (can answer from memory/knowledge or conversation context):
   - FINAL_ANSWER: [your response using conversation context when relevant]

CRITICAL RULES:
- Always consider conversation history when formulating responses
- Reference previous information when relevant (e.g., "As you mentioned earlier...")
- ALWAYS use the exact tool names available: {', '.join(self.tools.keys()) if self.tools else 'No tools available'}
- For web searches: USE "web_search"
- For terminal commands: USE "terminal"
- After TOOL_USE, STOP immediately - do not add anything else
- Use EXACT format: "TOOL_USE: [exact_tool_name]: [command_or_query]"
- Never add explanations after TOOL_USE

EXAMPLES:
- User mentions their name, later asks "What's my name?" → FINAL_ANSWER: [reference the name from conversation]
- "What are the latest news?" → TOOL_USE: web_search: latest news today
- "Check TLS certificate expiration" → TOOL_USE: terminal: echo | openssl s_client -connect localhost:8443 -servername localhost
- "Tell me more about that topic we discussed" → FINAL_ANSWER: [reference previous topic discussion]

IMPORTANT TERMINAL USAGE GUIDELINES:
- For TLS/SSL checks: Use "echo | openssl s_client" instead of interactive openssl
- For network testing: Use "curl --max-time 10" or "ping -c 4" instead of unlimited commands
- For file viewing: Use "cat" or "head -20" instead of interactive viewers like "less"
- Avoid: ssh, vim, nano, mysql, psql - these are interactive and will hang

Remember: Use conversation memory for context, tools for current information."""
    
    def _build_conversation_history(self, messages: List[BaseMessage], max_messages: int = None) -> List[Dict[str, str]]:
        """Build conversation history from LangGraph messages for context."""
        history = []
        
        # Convert LangGraph messages to simple format, excluding the current query
        for msg in messages[:-1]:  # Exclude current message to avoid duplication
            if hasattr(msg, 'content') and hasattr(msg, 'type'):
                if msg.type == "human":
                    history.append({"role": "user", "content": msg.content})
                elif msg.type == "ai":
                    history.append({"role": "assistant", "content": msg.content})
        
        # Apply sliding window to prevent token limit issues
        if max_messages is None:
            # Use default from configuration
            from config.constants import AGENT_CONFIG
            max_messages = AGENT_CONFIG.get("max_conversation_history", 20)
        
        if len(history) > max_messages:
            # Keep the most recent messages
            history = history[-max_messages:]
            
            # Log that we're truncating history
            self.logger.debug("Truncated conversation history", 
                            original_length=len(messages) - 1,
                            truncated_length=len(history),
                            max_messages=max_messages)
        
        return history
    
    def _format_conversation_summary(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history into a readable summary for context."""
        if not history:
            return "No previous conversation history."
        
        summary_parts = []
        summary_parts.append(f"Conversation history ({len(history)} messages):")
        
        for i, msg in enumerate(history[-10:]):  # Show last 10 for summary
            role = msg["role"].title()
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            summary_parts.append(f"{i+1}. {role}: {content}")
        
        return "\n".join(summary_parts)
    
    def register_tool(self, tool: ToolInterface) -> None:
        """Register a new tool with the agent."""
        try:
            tool_info = tool.get_tool_info()
            self.tools[tool_info.name] = tool
            
            # Create LangChain adapter
            langchain_tool = LangChainToolAdapter(tool)
            self.langchain_tools.append(langchain_tool)
            
            self.logger.info("Registered tool", tool_name=tool_info.name)
        except Exception as e:
            raise AgentError(
                f"Failed to register tool: {str(e)}",
                error_code=ErrorCodes.AGENT_INIT_FAILED,
                details={"tool": getattr(tool, 'name', 'unknown'), "error": str(e)}
            )
    
    def get_available_tools(self) -> List[ToolInfo]:
        """Get list of available tools."""
        return [tool.get_tool_info() for tool in self.tools.values()]
    
    def get_tools_description(self) -> str:
        """Get formatted description of all tools."""
        descriptions = []
        for tool in self.tools.values():
            tool_info = tool.get_tool_info()
            descriptions.append(f"- {tool_info.name}: {tool_info.description}")
        return "\n".join(descriptions)
    
    def execute_tool(self, tool_name: str, query: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific tool directly. Used by UI components for tool confirmation."""
        self.logger.debug("Executing tool directly", tool=tool_name, query=query)
        
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "output": "",
                "metadata": {"available_tools": list(self.tools.keys())}
            }
        
        try:
            tool = self.tools[tool_name]
            result = tool.execute(query, **kwargs)
            
            # Ensure result has expected format
            if isinstance(result, dict):
                return {
                    "success": result.get("success", True),
                    "output": result.get("results", result.get("output", str(result))),
                    "error": result.get("error", ""),
                    "metadata": result.get("metadata", {})
                }
            else:
                return {
                    "success": True,
                    "output": str(result),
                    "error": "",
                    "metadata": {}
                }
                
        except Exception as e:
            self.logger.error("Tool execution failed", tool=tool_name, query=query, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "metadata": {"exception_type": type(e).__name__}
            }
    
    def update_model(self, model: str) -> None:
        """Update the model used by the agent."""
        try:
            # Validate that the model exists
            available_models = self.llm_service.get_available_models()
            if model not in available_models:
                raise AgentError(
                    f"Model '{model}' not available",
                    error_code=ErrorCodes.AGENT_INIT_FAILED,
                    details={"model": model, "available": available_models}
                )
            
            old_model = self.model
            self.model = model
            self.logger.info("Updated agent model", old_model=old_model, new_model=model)
        except Exception as e:
            raise AgentError(
                f"Failed to update model: {str(e)}",
                error_code=ErrorCodes.AGENT_INIT_FAILED,
                details={"model": model, "error": str(e)}
            )
    
    def process_query_stream(self, user_query: str, conversation_history: List[Dict[str, str]] = None, **kwargs) -> Iterator[AgentStep]:
        """Process user query and yield steps as they happen."""
        start_time = time.time()
        correlation_id = create_request_tracer(user_query, self.model)
        
        with request_context(user_query=user_query, model=self.model) as _:
            self.logger.info("Starting LangGraph agent query processing", 
                           query=user_query[:100] + "..." if len(user_query) > 100 else user_query,
                           model=self.model,
                           correlation_id=correlation_id)
            
            # Log conversation context if provided
            if conversation_history:
                log_llm_conversation(
                    conversation_id=correlation_id,
                    messages=conversation_history + [{"role": "user", "content": user_query}],
                    model=self.model,
                    action="agent_processing"
                )
            
            try:
                # Prepare initial state
                messages = []
                if conversation_history:
                    for msg in conversation_history:
                        if msg.get("role") == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        elif msg.get("role") == "assistant":
                            messages.append(AIMessage(content=msg["content"]))
                
                # Add current query
                messages.append(HumanMessage(content=user_query))
                
                initial_state = AgentState(
                    messages=messages,
                    user_query=user_query,
                    agent_steps=[],
                    current_step="",
                    iteration_count=0,
                    max_iterations=self.max_iterations,
                    available_tools=list(self.tools.keys())
                )
                
                # Run the workflow
                config = {"configurable": {"thread_id": correlation_id}}
                yielded_steps = 0  # Track how many steps we've already yielded
                
                for event in self.workflow.stream(initial_state, config):
                    self.logger.debug("LangGraph event", event=event)
                    
                    # Log workflow steps for detailed tracking
                    for node_name, state_data in event.items():
                        log_agent_workflow(
                            workflow_id=correlation_id,
                            step_name=node_name,
                            step_type=node_name,
                            step_data=state_data if isinstance(state_data, dict) else {"state": str(state_data)}
                        )
                        
                        log_request_step(
                            correlation_id=correlation_id,
                            step=f"workflow_{node_name}",
                            data={"node": node_name, "state_keys": list(state_data.keys()) if isinstance(state_data, dict) else []}
                        )
                    
                    # Extract agent steps from the current state
                    current_state = event.get("reasoning") or event.get("tool_selection") or event.get("tool_execution") or event.get("final_answer")
                    
                    if current_state and "agent_steps" in current_state:
                        # Only yield new steps that haven't been yielded yet
                        total_steps = len(current_state["agent_steps"])
                        
                        for i in range(yielded_steps, total_steps):
                            step_data = current_state["agent_steps"][i]
                            
                            # Convert to AgentStep
                            step = AgentStep(
                                step_type=StepType(step_data["step_type"]),
                                content=step_data["content"],
                                timestamp=step_data["timestamp"],
                                metadata=step_data.get("metadata", {})
                            )
                            
                            # Enhanced logging for agent steps
                            log_agent_step(step.step_type.value, step.content, 
                                         correlation_id=correlation_id,
                                         step_index=i,
                                         **step.metadata)
                            yield step
                            yielded_steps += 1
                
                # Log completion
                total_duration = round((time.time() - start_time) * 1000)
                log_request_complete(
                    correlation_id=correlation_id,
                    total_duration_ms=total_duration,
                    success=True,
                    steps_generated=yielded_steps
                )
                
            except Exception as e:
                # Enhanced error logging with full context
                error_context = {
                    "user_query": user_query,
                    "model": self.model,
                    "conversation_history_length": len(conversation_history) if conversation_history else 0,
                    "available_tools": list(self.tools.keys()),
                    "correlation_id": correlation_id
                }
                
                log_error_with_context(e, error_context, "agent_processing")
                
                total_duration = round((time.time() - start_time) * 1000)
                log_request_complete(
                    correlation_id=correlation_id,
                    total_duration_ms=total_duration,
                    success=False,
                    error=str(e)
                )
                
                self.logger.exception("LangGraph agent processing failed", error=str(e))
                yield AgentStep(
                    step_type=StepType.ERROR,
                    content=f"Agent processing failed: {str(e)}",
                    timestamp=time.time(),
                    metadata={"error": str(e), "correlation_id": correlation_id}
                )
    
    def process_query(self, user_query: str, conversation_history: List[Dict[str, str]] = None, **kwargs) -> AgentResponse:
        """Process user query and return complete response."""
        # Validate query first
        self.validate_query(user_query)
        
        start_time = time.time()
        response = AgentResponse(query=user_query)
        
        try:
            for step in self.process_query_stream(user_query, conversation_history, **kwargs):
                response.add_step(step)
            
            response.total_time_ms = round((time.time() - start_time) * 1000)
            return response
            
        except Exception as e:
            error_step = AgentStep(
                step_type=StepType.ERROR,
                content=f"Query processing failed: {str(e)}",
                timestamp=time.time(),
                metadata={"error": str(e)}
            )
            response.add_step(error_step)
            response.error = str(e)
            response.total_time_ms = round((time.time() - start_time) * 1000)
            
            raise AgentProcessingError(
                f"Failed to process query: {str(e)}",
                error_code=ErrorCodes.AGENT_PROCESSING_FAILED,
                details={"query": user_query, "error": str(e)}
            )
