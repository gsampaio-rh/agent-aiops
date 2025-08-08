"""
Core exceptions for Agent-AIOps.

This module provides a comprehensive exception hierarchy with
error codes for better error handling and debugging.
"""

from .base import (
    ErrorCodes,
    AgentAIOpsException,
    ValidationError,
    ServiceError,
    create_error_with_code
)
from .service_exceptions import (
    OllamaServiceError,
    OllamaConnectionError,
    OllamaModelError,
    SearchError,
    SearchProviderError,
    SearchTimeoutError
)
from .agent_exceptions import (
    AgentError,
    AgentProcessingError,
    ToolExecutionError
)
from .mcp_exceptions import (
    MCPError,
    MCPConnectionError,
    MCPToolError,
    MCPTimeoutError,
    MCPPermissionError,
    MCPInvalidResponseError
)

__all__ = [
    # Base exceptions
    "ErrorCodes",
    "AgentAIOpsException",
    "ValidationError",
    "ServiceError",
    "create_error_with_code",
    
    # Service exceptions
    "OllamaServiceError",
    "OllamaConnectionError", 
    "OllamaModelError",
    "SearchError",
    "SearchProviderError",
    "SearchTimeoutError",
    
    # Agent exceptions
    "AgentError",
    "AgentProcessingError",
    "ToolExecutionError",
    
    # MCP exceptions
    "MCPError",
    "MCPConnectionError",
    "MCPToolError",
    "MCPTimeoutError",
    "MCPPermissionError",
    "MCPInvalidResponseError"
]
