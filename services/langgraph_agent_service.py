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
from utils.logger import get_logger, log_performance, log_agent_step, request_context


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
    
    def __init__(self, tool: ToolInterface):
        self.wrapped_tool = tool
        tool_info = tool.get_tool_info()
        super().__init__(
            name=tool_info.name,
            description=tool_info.description
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
        workflow.add_conditional_edges(
            "tool_execution",
            self._should_continue,
            {
                "continue": "reasoning",
                "finish": "final_answer"
            }
        )
        
        workflow.add_edge("final_answer", END)
        
        # Set entry point
        workflow.set_entry_point("reasoning")
        
        # Compile the workflow
        self.workflow = workflow.compile(checkpointer=self.memory)
    
    def _reasoning_node(self, state: AgentState) -> AgentState:
        """Node for agent reasoning and planning."""
        self.logger.debug("Executing reasoning node")
        
        # Create reasoning step
        step_data = {
            "step_type": "thought",
            "content": "Analyzing the user's request and determining next steps...",
            "timestamp": time.time(),
            "metadata": {"node": "reasoning"}
        }
        
        state["agent_steps"].append(step_data)
        state["current_step"] = "reasoning"
        
        # Generate reasoning using LLM
        messages = state["messages"]
        
        # Create system prompt
        system_prompt = self._get_system_prompt()
        
        # Get LLM response for reasoning
        reasoning_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["user_query"]}
        ]
        
        # Add conversation history
        for msg in messages[:-1]:  # Exclude current message
            if hasattr(msg, 'content') and hasattr(msg, 'type'):
                role = "user" if msg.type == "human" else "assistant"
                reasoning_messages.append({"role": role, "content": msg.content})
        
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
        
        step_data = {
            "step_type": "tool_use",
            "content": "Executing selected tool...",
            "timestamp": time.time(),
            "metadata": {"node": "tool_execution"}
        }
        
        state["agent_steps"].append(step_data)
        state["current_step"] = "tool_execution"
        
        # Tool execution logic would go here
        # For now, simulate tool execution
        
        return state
    
    def _final_answer_node(self, state: AgentState) -> AgentState:
        """Node for generating final answer."""
        self.logger.debug("Executing final answer node")
        
        step_data = {
            "step_type": "final_answer",
            "content": "Generating final response based on analysis and tool results...",
            "timestamp": time.time(),
            "metadata": {"node": "final_answer"}
        }
        
        state["agent_steps"].append(step_data)
        state["current_step"] = "final_answer"
        
        return state
    
    def _should_use_tool(self, state: AgentState) -> str:
        """Determine if a tool should be used."""
        # Simple logic for now - check if reasoning mentions tool usage
        last_step = state["agent_steps"][-1] if state["agent_steps"] else {}
        content = last_step.get("content", "")
        
        # Look for tool usage patterns
        if any(tool_name in content.lower() for tool_name in state["available_tools"]):
            return "use_tool"
        elif "TOOL_USE:" in content:
            return "use_tool"
        else:
            return "final_answer"
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if processing should continue or finish."""
        if state["iteration_count"] >= state["max_iterations"]:
            return "finish"
        
        # Check if we have a complete answer
        last_step = state["agent_steps"][-1] if state["agent_steps"] else {}
        if last_step.get("step_type") == "tool_result":
            return "finish"
        
        return "continue"
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        tools_description = self.get_tools_description()
        
        return f"""You are a helpful AI assistant with access to tools. You can reason step by step and use tools when needed.

Available tools:
{tools_description}

IMPORTANT: Follow this exact pattern for ALL responses:

1. THOUGHT: Think about what the user is asking and whether you need to use a tool
2. If you need a tool:
   - TOOL_SELECTION: Choose which tool and explain why
   - TOOL_USE: [tool_name]: [exact command or query]
3. If you don't need tools:
   - FINAL_ANSWER: [your direct response]

CRITICAL RULES:
- After TOOL_USE, STOP immediately - do not add anything else
- The system will execute the tool and provide results automatically
- Use EXACT format: "TOOL_USE: terminal: ls -la" or "TOOL_USE: web_search: Python tutorial"
- Never add explanations after TOOL_USE

Remember: STOP after TOOL_USE. The system handles tool execution and results."""
    
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
        
        with request_context(user_query=user_query, model=self.model) as correlation_id:
            self.logger.info("Starting LangGraph agent query processing", 
                           query=user_query[:100] + "..." if len(user_query) > 100 else user_query,
                           model=self.model,
                           correlation_id=correlation_id)
            
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
                
                for event in self.workflow.stream(initial_state, config):
                    self.logger.debug("LangGraph event", event=event)
                    
                    # Extract agent steps from the current state
                    current_state = event.get("reasoning") or event.get("tool_selection") or event.get("tool_execution") or event.get("final_answer")
                    
                    if current_state and "agent_steps" in current_state:
                        for step_data in current_state["agent_steps"]:
                            # Convert to AgentStep
                            step = AgentStep(
                                step_type=StepType(step_data["step_type"]),
                                content=step_data["content"],
                                timestamp=step_data["timestamp"],
                                metadata=step_data.get("metadata", {})
                            )
                            
                            log_agent_step(step.step_type.value, step.content)
                            yield step
                
            except Exception as e:
                self.logger.exception("LangGraph agent processing failed", error=str(e))
                yield AgentStep(
                    step_type=StepType.ERROR,
                    content=f"Agent processing failed: {str(e)}",
                    timestamp=time.time(),
                    metadata={"error": str(e)}
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
