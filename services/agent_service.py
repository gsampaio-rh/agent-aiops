"""
React Agent Service

Implements a reasoning agent that can use tools and shows its thought process.
Inspired by ReAct (Reasoning + Acting) pattern.
Implements AgentServiceInterface for consistent API.
"""

import time
import json
from typing import Dict, List, Any, Optional, Iterator
import uuid

from core.interfaces.agent_service import AgentServiceInterface, ToolInterface
from core.interfaces.llm_service import LLMServiceInterface
from core.interfaces.search_service import SearchServiceInterface
from core.interfaces.mcp_service import MCPServiceInterface
from core.models.agent import AgentStep, AgentResponse, StepType, ToolInfo
from core.models.search import SearchQuery
from core.exceptions import (
    AgentError, AgentProcessingError, ToolExecutionError, 
    ErrorCodes, create_error_with_code, MCPError
)
from services.ollama_service import OllamaService
from services.search_service import WebSearchService
from services.terminal_tool import TerminalTool
from services.mcp_service import DesktopCommanderMCP
from config.constants import AGENT_CONFIG, MCP_CONFIG
from utils.logger import get_logger, log_performance, log_agent_step, request_context


class AgentTool(ToolInterface):
    """Base class for agent tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        raise NotImplementedError
    
    def get_tool_info(self) -> ToolInfo:
        """Get tool information for the agent."""
        return ToolInfo(
            name=self.name,
            description=self.description
        )


class WebSearchTool(AgentTool):
    """Web search tool for the agent."""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for current information, facts, news, and general knowledge. Use this when you need up-to-date information or when the user asks about current events, recent developments, or specific factual information that might not be in your training data."
        )
        self.search_service = WebSearchService()
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute web search."""
        provider = kwargs.get("provider", "duckduckgo")
        max_results = kwargs.get("max_results", 5)
        
        try:
            # Import here to avoid circular imports and ensure fresh instance
            from services.search_service import WebSearchService
            search_service = WebSearchService()
            
            # Create search query
            search_query = SearchQuery(
                query=query,
                provider=provider,
                max_results=max_results
            )
            
            # Use new interface method
            result = search_service.search(search_query)
            
            if result.is_successful() and result.results:
                # Use the built-in formatting method
                formatted_results = result.get_formatted_results(max_results)
                
                return {
                    "success": True,
                    "results": formatted_results,
                    "metadata": {
                        "provider": result.provider_name,
                        "total_results": result.total_results,
                        "search_time_ms": result.search_time_ms
                    }
                }
            else:
                return {
                    "success": False,
                    "error": result.error or "No results found",
                    "metadata": {
                        "provider": result.provider_name,
                        "total_results": result.total_results,
                        "search_time_ms": result.search_time_ms
                    }
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "metadata": {}
            }


class ReactAgent(AgentServiceInterface):
    """React Agent that can reason and use tools."""
    
    def __init__(self, model: str = "llama3.2:3b", 
                 llm_service: Optional[LLMServiceInterface] = None,
                 search_service: Optional[SearchServiceInterface] = None):
        self.llm_service = llm_service or OllamaService()
        self.search_service = search_service or WebSearchService()
        self.model = model
        self.tools: Dict[str, ToolInterface] = {}
        self.max_iterations = AGENT_CONFIG.get("max_iterations", 10)
        self.logger = get_logger(__name__)
        
        # MCP services
        self.mcp_services: Dict[str, MCPServiceInterface] = {}
        self._mcp_connected = False
        
        # Initialize MCP services if enabled
        self._initialize_mcp_services()
        
        # Register default tools
        self.register_tool(WebSearchTool())
        
        # Register terminal tool if enabled
        if AGENT_CONFIG.get("enable_terminal", False):
            # Get the desktop_commander MCP service for the terminal tool
            mcp_service = self.mcp_services.get("desktop_commander")
            self.register_tool(TerminalTool(require_confirmation=False, mcp_client=mcp_service))
        
        self.logger.info("Initialized ReactAgent", model=model, tools=list(self.tools.keys()))
    
    def _initialize_mcp_services(self) -> None:
        """Initialize MCP services based on configuration."""
        try:
            mcp_servers = AGENT_CONFIG.get("mcp_servers", [])
            
            for server_name in mcp_servers:
                if server_name == "desktop_commander" and MCP_CONFIG.get("desktop_commander", {}).get("enabled", False):
                    self.mcp_services[server_name] = DesktopCommanderMCP()
                    self.logger.info("Initialized MCP service", service=server_name)
                    
        except Exception as e:
            self.logger.error("Failed to initialize MCP services", error=str(e))
    
    async def connect_mcp_services(self) -> None:
        """Connect to all configured MCP services."""
        if self._mcp_connected:
            return
            
        try:
            for service_name, service in self.mcp_services.items():
                self.logger.info("Connecting to MCP service", service=service_name)
                connected = await service.connect()
                if connected:
                    self.logger.info("Successfully connected to MCP service", service=service_name)
                    
                    # List available tools for debugging
                    try:
                        tools = await service.list_tools()
                        self.logger.info("MCP service tools available", 
                                       service=service_name, 
                                       tools=[tool.get('name') for tool in tools])
                    except Exception as e:
                        self.logger.warning("Failed to list tools for MCP service", 
                                          service=service_name, error=str(e))
                else:
                    self.logger.error("Failed to connect to MCP service", service=service_name)
            
            self._mcp_connected = True
            self.logger.info("All MCP services connected")
            
        except Exception as e:
            self.logger.error("Failed to connect MCP services", error=str(e))
            raise AgentError(
                f"Failed to connect MCP services: {str(e)}",
                error_code=ErrorCodes.AGENT_INIT_FAILED,
                details={"error": str(e)}
            )
    
    async def disconnect_mcp_services(self) -> None:
        """Disconnect from all MCP services."""
        try:
            for service_name, service in self.mcp_services.items():
                if service.is_connected():
                    await service.disconnect()
                    self.logger.info("Disconnected from MCP service", service=service_name)
            
            self._mcp_connected = False
            self.logger.info("All MCP services disconnected")
            
        except Exception as e:
            self.logger.error("Failed to disconnect MCP services", error=str(e))
    
    def get_mcp_service(self, service_name: str) -> Optional[MCPServiceInterface]:
        """Get MCP service by name."""
        return self.mcp_services.get(service_name)
    
    @property
    def system_prompt(self) -> str:
        """Backward compatibility property for system prompt."""
        return self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return """You are a helpful AI assistant with access to web search and terminal command execution. You can reason step by step and use tools when needed.

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

TOOL-SPECIFIC GUIDELINES:

web_search:
- Format: "TOOL_USE: web_search: simple search terms"
- Example: "TOOL_USE: web_search: weather forecast London"

terminal:
- Format: "TOOL_USE: terminal: exact_command"
- Examples: 
  * "TOOL_USE: terminal: ls -la"
  * "TOOL_USE: terminal: pwd"
  * "TOOL_USE: terminal: git status"
  * "TOOL_USE: terminal: python --version"

Remember: STOP after TOOL_USE. The system handles tool execution and results."""
    
    def register_tool(self, tool: ToolInterface) -> None:
        """
        Register a new tool with the agent.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            AgentError: If tool registration fails
        """
        try:
            tool_info = tool.get_tool_info()
            self.tools[tool_info.name] = tool
            self.logger.info("Registered tool", tool_name=tool_info.name)
        except Exception as e:
            raise AgentError(
                f"Failed to register tool: {str(e)}",
                error_code=ErrorCodes.AGENT_INIT_FAILED,
                details={"tool": getattr(tool, 'name', 'unknown'), "error": str(e)}
            )
    
    def get_available_tools(self) -> List[ToolInfo]:
        """
        Get list of available tools.
        
        Returns:
            List[ToolInfo]: Available tool information
        """
        return [tool.get_tool_info() for tool in self.tools.values()]
    
    def update_model(self, model: str) -> None:
        """
        Update the model used by the agent.
        
        Args:
            model: New model name
            
        Raises:
            AgentError: If model update fails
        """
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
    
    def get_tools_description(self) -> str:
        """Get formatted description of all tools."""
        descriptions = []
        for tool in self.tools.values():
            tool_info = tool.get_tool_info()
            descriptions.append(f"- {tool_info.name}: {tool_info.description}")
        return "\n".join(descriptions)
    
    def parse_agent_response(self, response: str) -> List[AgentStep]:
        """Parse agent response into structured steps."""
        steps = []
        current_time = time.time()
        
        # Split response by known step markers
        lines = response.strip().split('\n')
        current_step_type = None
        current_content = []
        
        step_markers = {
            'THOUGHT:': StepType.THOUGHT,
            'TOOL_SELECTION:': StepType.TOOL_SELECTION,
            'TOOL_USE:': StepType.TOOL_USE,
            'TOOL_RESULT:': StepType.TOOL_RESULT,
            'FINAL_ANSWER:': StepType.FINAL_ANSWER
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts a new step
            step_found = False
            for marker, step_type in step_markers.items():
                if line.startswith(marker):
                    # Save previous step if exists
                    if current_step_type and current_content:
                        steps.append(AgentStep(
                            step_type=current_step_type,
                            content='\n'.join(current_content).strip(),
                            timestamp=current_time
                        ))
                    
                    # Start new step
                    current_step_type = step_type
                    current_content = [line[len(marker):].strip()]
                    step_found = True
                    break
            
            if not step_found and current_step_type:
                current_content.append(line)
        
        # Add final step
        if current_step_type and current_content:
            steps.append(AgentStep(
                step_type=current_step_type,
                content='\n'.join(current_content).strip(),
                timestamp=current_time
            ))
        
        return steps
    
    def execute_tool(self, tool_name: str, query: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific tool."""
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "metadata": {}
            }
        
        tool = self.tools[tool_name]
        return tool.execute(query, **kwargs)
    
    def process_query_stream(self, user_query: str, **kwargs) -> Iterator[AgentStep]:
        """Process user query and yield steps as they happen."""
        start_time = time.time()
        
        # Create a request context for correlation tracking
        with request_context(user_query=user_query, model=self.model) as correlation_id:
            self.logger.info("Starting agent query processing", 
                           query=user_query[:100] + "..." if len(user_query) > 100 else user_query,
                           model=self.model,
                           correlation_id=correlation_id)
        
            try:
                # Connect to MCP services if not already connected
                if not self._mcp_connected and self.mcp_services:
                    yield AgentStep(
                        step_type=StepType.THOUGHT,
                        content="Connecting to MCP services...",
                        timestamp=time.time(),
                        metadata={"status": "connecting_mcp"}
                    )
                    
                    # Run the async connection in the current thread
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # We're in an async context, create a task
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(asyncio.run, self.connect_mcp_services())
                                future.result()
                        else:
                            # No loop running, we can use run_until_complete
                            loop.run_until_complete(self.connect_mcp_services())
                    except RuntimeError:
                        # Fallback: create new loop in thread
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, self.connect_mcp_services())
                            future.result()
                    
                    self.logger.info("MCP services connected successfully")
                
                # Prepare the conversation
                system_message = self._get_system_prompt().format(
                    tools_description=self.get_tools_description()
                )
                
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_query}
                ]
                
                # Yield initial thought step
                yield AgentStep(
                    step_type=StepType.THOUGHT,
                    content="Let me think about this query...",
                    timestamp=time.time(),
                    metadata={"status": "starting"}
                )
                
                # Get agent response
                full_response = ""
                for chunk in self.llm_service.chat_stream(
                    model=self.model,
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.7),
                    max_tokens=kwargs.get("max_tokens", 1000)
                ):
                    if chunk.get("content"):
                        full_response += chunk["content"]
                
                # Parse the response into steps
                parsed_steps = self.parse_agent_response(full_response)
                
                # Log parsed steps
                self.logger.debug("Parsed agent steps", step_count=len(parsed_steps), 
                                step_types=[step.step_type.value for step in parsed_steps])
                
                # Track if we've already yielded a final answer
                has_final_answer = any(step.step_type == StepType.FINAL_ANSWER for step in parsed_steps)
                
                # Yield each step and execute tools if needed
                for step in parsed_steps:
                    # Log each step
                    log_agent_step(step.step_type.value, step.content)
                    
                    # Skip FINAL_ANSWER from initial response if we have tools to execute
                    if step.step_type == StepType.FINAL_ANSWER and any(s.step_type == StepType.TOOL_USE for s in parsed_steps):
                        continue
                        
                    yield step
                    
                    # If this is a tool use step, execute the tool
                    if step.step_type == StepType.TOOL_USE:
                        # Parse tool usage
                        content = step.content.strip()
                        if ':' in content:
                            tool_part, query_part = content.split(':', 1)
                            tool_name = tool_part.strip()
                            tool_query = query_part.strip()
                            
                            # Clean up the query - remove quotes and OR operators for better search
                            tool_query = tool_query.replace('"', '').replace(' OR ', ' ')
                            if tool_query.startswith('(') and tool_query.endswith(')'):
                                tool_query = tool_query[1:-1]
                            
                            # Execute the tool with unique ID to prevent duplicates
                            import uuid
                            execution_id = str(uuid.uuid4())[:8]
                            
                            self.logger.info("Executing tool", tool=tool_name, query=tool_query, execution_id=execution_id)
                            
                            yield AgentStep(
                                step_type=StepType.TOOL_USE,
                                content=f"Executing {tool_name} with query: {tool_query}",
                                timestamp=time.time(),
                                metadata={"tool": tool_name, "query": tool_query, "status": "executing", "execution_id": execution_id}
                            )
                            
                            tool_result = self.execute_tool(tool_name, tool_query)
                            
                            # If DuckDuckGo fails, automatically try SearX as fallback
                            if not tool_result["success"] and tool_name == "web_search":
                                self.logger.warning("Primary search failed, trying fallback", tool=tool_name, error=tool_result.get("error"))
                                
                                yield AgentStep(
                                    step_type=StepType.TOOL_USE,
                                    content=f"DuckDuckGo failed, trying SearX with query: {tool_query}",
                                    timestamp=time.time(),
                                    metadata={"tool": "web_search", "provider": "searx", "query": tool_query, "status": "fallback", "execution_id": execution_id}
                                )
                                
                                # Try with SearX provider
                                fallback_result = self.execute_tool(tool_name, tool_query, provider="searx")
                                if fallback_result["success"]:
                                    tool_result = fallback_result
                                    self.logger.info("Fallback search succeeded")
                            
                            # Yield tool result
                            if tool_result["success"]:
                                # Handle different result formats for different tools
                                if tool_name == "terminal":
                                    result_content = tool_result.get("output", "")
                                    results_count = len(result_content)
                                else:
                                    result_content = tool_result.get("results", "")
                                    results_count = len(result_content) if isinstance(result_content, str) else len(result_content or [])
                                
                                self.logger.info("Tool execution succeeded", tool=tool_name, results_count=results_count)
                                
                                yield AgentStep(
                                    step_type=StepType.TOOL_RESULT,
                                    content=result_content,
                                    timestamp=time.time(),
                                    metadata={
                                        "tool": tool_name,
                                        "success": True,
                                        **tool_result.get("metadata", {})
                                    }
                                )
                                
                                # Now get the agent's analysis of the tool results
                                if tool_name == "terminal":
                                    analysis_messages = [
                                        {"role": "system", "content": "You are analyzing terminal command output. Provide a clear explanation of what the command shows and answer the user's question based on the output."},
                                        {"role": "user", "content": f"Original query: {user_query}\n\nCommand executed: {tool_query}\nCommand output:\n{result_content}\n\nBased on this terminal output, provide a comprehensive answer to the user's question:"}
                                    ]
                                else:
                                    analysis_messages = [
                                        {"role": "system", "content": "You are analyzing search results. Provide a direct, comprehensive answer based on the search results. Do not repeat the thinking process or use step markers. Just give the final answer."},
                                        {"role": "user", "content": f"Original query: {user_query}\n\nSearch results:\n{result_content}\n\nBased on these search results, provide a comprehensive answer to the user's question:"}
                                    ]
                                
                                analysis_response = ""
                                for chunk in self.llm_service.chat_stream(
                                    model=self.model,
                                    messages=analysis_messages,
                                    temperature=kwargs.get("temperature", 0.7),
                                    max_tokens=kwargs.get("max_tokens", 1000)
                                ):
                                    if chunk.get("content"):
                                        analysis_response += chunk["content"]
                                
                                total_time_ms = round((time.time() - start_time) * 1000)
                                self.logger.info("Agent query processing completed", total_time_ms=total_time_ms)
                                
                                yield AgentStep(
                                    step_type=StepType.FINAL_ANSWER,
                                    content=analysis_response,
                                    timestamp=time.time(),
                                    metadata={"total_time_ms": total_time_ms}
                                )
                            else:
                                self.logger.error("Tool execution failed", tool=tool_name, error=tool_result.get("error"))
                                yield AgentStep(
                                    step_type=StepType.ERROR,
                                    content=f"Tool execution failed: {tool_result['error']}",
                                    timestamp=time.time(),
                                    metadata={"tool": tool_name, "error": tool_result["error"]}
                                )
                
            except Exception as e:
                self.logger.exception("Agent processing failed", error=str(e))
                yield AgentStep(
                    step_type=StepType.ERROR,
                    content=f"Agent processing failed: {str(e)}",
                    timestamp=time.time(),
                    metadata={"error": str(e)}
                )
    
    def process_query(self, user_query: str, **kwargs) -> AgentResponse:
        """
        Process user query and return complete response.
        
        Args:
            user_query: User's input query
            **kwargs: Additional processing parameters
            
        Returns:
            AgentResponse: Complete agent response with all steps
            
        Raises:
            AgentProcessingError: If query processing fails
        """
        # Validate query first
        self.validate_query(user_query)
        
        start_time = time.time()
        response = AgentResponse(query=user_query)
        
        try:
            for step in self.process_query_stream(user_query, **kwargs):
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
