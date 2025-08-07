"""
React Agent Service

Implements a reasoning agent that can use tools and shows its thought process.
Inspired by ReAct (Reasoning + Acting) pattern.
"""

import time
import json
from typing import Dict, List, Any, Optional, Iterator, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from services.ollama_service import OllamaService
from services.search_service import WebSearchService


class StepType(Enum):
    """Types of agent steps."""
    THOUGHT = "thought"
    TOOL_SELECTION = "tool_selection"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"


@dataclass
class AgentStep:
    """Represents a single step in the agent's reasoning process."""
    step_type: StepType
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_type": self.step_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata or {}
        }


class AgentTool:
    """Base class for agent tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        raise NotImplementedError
    
    def get_tool_info(self) -> Dict[str, str]:
        """Get tool information for the agent."""
        return {
            "name": self.name,
            "description": self.description
        }


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
            result = self.search_service.search(
                query=query,
                provider=provider,
                max_results=max_results
            )
            
            if result["status"] == "success" and result["results"]:
                # Format results for the agent
                formatted_results = []
                for i, res in enumerate(result["results"][:max_results], 1):
                    formatted_results.append(f"{i}. {res['title']}\n   {res['snippet']}\n   Source: {res['url']}")
                
                return {
                    "success": True,
                    "results": "\n\n".join(formatted_results),
                    "metadata": {
                        "provider": result.get("provider_name", provider),
                        "total_results": result["total_results"],
                        "search_time_ms": result.get("search_time_ms", 0)
                    }
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "No results found"),
                    "metadata": result
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "metadata": {}
            }


class ReactAgent:
    """React Agent that can reason and use tools."""
    
    def __init__(self, model: str = "llama3.2:3b"):
        self.ollama_service = OllamaService()
        self.model = model
        self.tools = {}
        self.max_iterations = 10
        
        # Register default tools
        self.register_tool(WebSearchTool())
        
        # System prompt for the agent
        self.system_prompt = """You are a helpful AI assistant that can reason step by step and use tools when needed.

Available tools:
{tools_description}

When responding to a user query, follow this pattern:

1. THOUGHT: Think about what the user is asking and whether you need to use a tool
2. TOOL_SELECTION: If you need a tool, decide which one and why
3. TOOL_USE: Use the tool with appropriate parameters
4. TOOL_RESULT: Analyze the tool results
5. FINAL_ANSWER: Provide a comprehensive answer to the user

For each step, be explicit about your reasoning. Use the format:
THOUGHT: [your reasoning]
TOOL_SELECTION: [if needed, which tool and why]
TOOL_USE: [tool_name: simple search query]
TOOL_RESULT: [analysis of results]
FINAL_ANSWER: [your final response]

IMPORTANT: When using web_search, use simple, clear search terms. For example:
- Good: "Brazil facts"
- Good: "Python tutorial"
- Bad: "Brazil facts" OR "Brazil overview"
- Bad: (complex query with operators)

If you don't need any tools, you can go directly to FINAL_ANSWER after THOUGHT.

Be conversational and helpful in your final answer, incorporating any tool results naturally."""
    
    def register_tool(self, tool: AgentTool):
        """Register a new tool."""
        self.tools[tool.name] = tool
    
    def get_tools_description(self) -> str:
        """Get formatted description of all tools."""
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")
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
    
    def execute_tool(self, tool_name: str, query: str) -> Dict[str, Any]:
        """Execute a specific tool."""
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "metadata": {}
            }
        
        tool = self.tools[tool_name]
        return tool.execute(query)
    
    def process_query_stream(self, user_query: str, **kwargs) -> Iterator[AgentStep]:
        """Process user query and yield steps as they happen."""
        start_time = time.time()
        
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
            for chunk in self.ollama_service.chat_stream(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000)
            ):
                if chunk.get("content"):
                    full_response += chunk["content"]
            
            # Parse the response into steps
            parsed_steps = self.parse_agent_response(full_response)
            
            # Yield each step and execute tools if needed
            for step in parsed_steps:
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
                        
                        # Execute the tool
                        yield AgentStep(
                            step_type=StepType.TOOL_USE,
                            content=f"Executing {tool_name} with query: {tool_query}",
                            timestamp=time.time(),
                            metadata={"tool": tool_name, "query": tool_query, "status": "executing"}
                        )
                        
                        tool_result = self.execute_tool(tool_name, tool_query)
                        
                        # Yield tool result
                        if tool_result["success"]:
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
                            analysis_messages = messages + [
                                {"role": "assistant", "content": full_response},
                                {"role": "user", "content": f"Here are the search results:\n\n{tool_result['results']}\n\nPlease analyze these results and provide your final answer."}
                            ]
                            
                            analysis_response = ""
                            for chunk in self.ollama_service.chat_stream(
                                model=self.model,
                                messages=analysis_messages,
                                temperature=kwargs.get("temperature", 0.7),
                                max_tokens=kwargs.get("max_tokens", 1000)
                            ):
                                if chunk.get("content"):
                                    analysis_response += chunk["content"]
                            
                            yield AgentStep(
                                step_type=StepType.FINAL_ANSWER,
                                content=analysis_response,
                                timestamp=time.time(),
                                metadata={"total_time_ms": round((time.time() - start_time) * 1000)}
                            )
                        else:
                            yield AgentStep(
                                step_type=StepType.ERROR,
                                content=f"Tool execution failed: {tool_result['error']}",
                                timestamp=time.time(),
                                metadata={"tool": tool_name, "error": tool_result["error"]}
                            )
            
        except Exception as e:
            yield AgentStep(
                step_type=StepType.ERROR,
                content=f"Agent processing failed: {str(e)}",
                timestamp=time.time(),
                metadata={"error": str(e)}
            )
    
    def process_query(self, user_query: str, **kwargs) -> List[AgentStep]:
        """Process user query and return all steps."""
        steps = []
        for step in self.process_query_stream(user_query, **kwargs):
            steps.append(step)
        return steps
