"""
Agent-specific exceptions.
"""

from typing import Optional, Dict, Any
from .base import AgentAIOpsException, ErrorCodes


class AgentError(AgentAIOpsException):
    """Base exception for agent-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", ErrorCodes.AGENT_PROCESSING_FAILED),
            **kwargs
        )


class AgentProcessingError(AgentError):
    """Raised when agent processing fails."""
    
    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if query:
            details["query"] = query[:100] + "..." if len(query) > 100 else query
        
        super().__init__(
            message,
            error_code=ErrorCodes.AGENT_PROCESSING_FAILED,
            details=details,
            **kwargs
        )


class ToolExecutionError(AgentError):
    """Raised when tool execution fails."""
    
    def __init__(self, message: str, tool_name: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if tool_name:
            details["tool"] = tool_name
        
        super().__init__(
            message,
            error_code=ErrorCodes.AGENT_TOOL_FAILED,
            details=details,
            **kwargs
        )
