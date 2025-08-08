"""
Agent Service Interface

Defines contracts for AI agents that can reason and use tools.
Supports the ReAct pattern (Reasoning + Acting).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator, Optional
from core.models.agent import AgentResponse, AgentStep, ToolInfo


class ToolInterface(ABC):
    """Interface for agent tools."""
    
    @abstractmethod
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.
        
        Args:
            query: The query or input for the tool
            **kwargs: Additional tool-specific parameters
            
        Returns:
            Dict[str, Any]: Tool execution results with metadata
        """
        pass
    
    @abstractmethod
    def get_tool_info(self) -> ToolInfo:
        """
        Get information about this tool.
        
        Returns:
            ToolInfo: Tool metadata including name and description
        """
        pass


class AgentServiceInterface(ABC):
    """Abstract interface for AI agent services."""
    
    @abstractmethod
    def process_query_stream(self, user_query: str, **kwargs) -> Iterator[AgentStep]:
        """
        Process user query and yield steps as they happen.
        
        Args:
            user_query: The user's input query
            **kwargs: Additional processing parameters
            
        Yields:
            AgentStep: Individual reasoning and action steps
            
        Raises:
            AgentError: If processing fails
        """
        pass
    
    @abstractmethod
    def process_query(self, user_query: str, **kwargs) -> AgentResponse:
        """
        Process user query and return complete response.
        
        Args:
            user_query: The user's input query
            **kwargs: Additional processing parameters
            
        Returns:
            AgentResponse: Complete agent response with all steps
            
        Raises:
            AgentError: If processing fails
        """
        pass
    
    @abstractmethod
    def register_tool(self, tool: ToolInterface) -> None:
        """
        Register a new tool with the agent.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            AgentError: If tool registration fails
        """
        pass
    
    @abstractmethod
    def get_available_tools(self) -> List[ToolInfo]:
        """
        Get list of available tools.
        
        Returns:
            List[ToolInfo]: Available tool information
        """
        pass
    
    @abstractmethod
    def update_model(self, model: str) -> None:
        """
        Update the model used by the agent.
        
        Args:
            model: New model name
            
        Raises:
            AgentError: If model update fails
        """
        pass
    
    def validate_query(self, query: str) -> None:
        """
        Validate user query input.
        
        Args:
            query: User query to validate
            
        Raises:
            ValidationError: If query is invalid
        """
        from core.exceptions import ValidationError
        
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")
        
        if len(query.strip()) < 3:
            raise ValidationError("Query must be at least 3 characters long")
        
        if len(query) > 10000:  # 10KB limit
            raise ValidationError("Query is too long (max 10,000 characters)")
