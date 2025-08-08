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
from core.models.agent import AgentStep, AgentResponse, StepType, ToolInfo
from core.models.search import SearchQuery
from core.exceptions import (
    AgentError, AgentProcessingError, ToolExecutionError, 
    ErrorCodes, create_error_with_code
)
from services.ollama_service import OllamaService
from services.search_service import WebSearchService
from config.constants import AGENT_CONFIG
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
        
        # Register default tools
        self.register_tool(WebSearchTool())
        
        self.logger.info("Initialized ReactAgent", model=model, tools=list(self.tools.keys()))
        
        # System prompt for the agent
        self.system_prompt = """You are a helpful AI assistant that can reason step by step and use tools when needed.

Available tools:
{tools_description}

When responding to a user query, follow this pattern:

1. THOUGHT: Think about what the user is asking and whether you need to use a tool
2. TOOL_SELECTION: If you need a tool, decide which one and why
3. TOOL_USE: Use the tool with appropriate parameters

If you use a tool, STOP after TOOL_USE. Do not provide TOOL_RESULT or FINAL_ANSWER - these will be generated after the tool execution.

If you don't need any tools, you can go directly to FINAL_ANSWER after THOUGHT.

Use the format:
THOUGHT: [your reasoning]
TOOL_SELECTION: [if needed, which tool and why]  
TOOL_USE: [tool_name: simple search query]
OR
THOUGHT: [your reasoning]
FINAL_ANSWER: [direct answer if no tools needed]

IMPORTANT: When using web_search, use simple, clear search terms. For example:
- Good: "Brazil facts"
- Good: "Python tutorial"  
- Bad: "Brazil facts" OR "Brazil overview"
- Bad: (complex query with operators)

Be conversational and helpful in your reasoning."""
    
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
                # Prepare the conversation
                system_message = self.system_prompt.format(
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
                                self.logger.info("Tool execution succeeded", tool=tool_name, results_count=len(tool_result.get("results", "")))
                                
                                yield AgentStep(
                                    step_type=StepType.TOOL_RESULT,
                                    content=tool_result["results"],
                                    timestamp=time.time(),
                                    metadata={
                                        "tool": tool_name,
                                        "success": True,
                                        **tool_result.get("metadata", {})
                                    }
                                )
                                
                                # Now get the agent's analysis of the tool results
                                analysis_messages = [
                                    {"role": "system", "content": "You are analyzing search results. Provide a direct, comprehensive answer based on the search results. Do not repeat the thinking process or use step markers. Just give the final answer."},
                                    {"role": "user", "content": f"Original query: {user_query}\n\nSearch results:\n{tool_result['results']}\n\nBased on these search results, provide a comprehensive answer to the user's question:"}
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
