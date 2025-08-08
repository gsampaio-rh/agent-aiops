"""
MCP-specific exception classes.
"""

from typing import Optional, Dict, Any
from .base import AgentAIOpsException, ServiceError, ErrorCodes


class MCPError(ServiceError):
    """Base exception for MCP-related errors."""
    
    def __init__(
        self,
        message: str,
        mcp_server: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if mcp_server:
            details["mcp_server"] = mcp_server
        
        super().__init__(
            message,
            service_name="MCP",
            error_code=kwargs.get("error_code", ErrorCodes.MCP_REQUEST_FAILED),
            details=details,
            original_exception=kwargs.get("original_exception")
        )


class MCPConnectionError(MCPError):
    """Raised when MCP server connection fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCodes.MCP_CONNECTION_FAILED,
            **kwargs
        )


class MCPToolError(MCPError):
    """Raised when MCP tool execution fails."""
    
    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if tool_name:
            details["tool_name"] = tool_name
        
        super().__init__(
            message,
            error_code=ErrorCodes.MCP_TOOL_NOT_FOUND,
            details=details,
            **kwargs
        )


class MCPTimeoutError(MCPError):
    """Raised when MCP operation times out."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        details = kwargs.get("details", {})
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        
        super().__init__(
            message,
            error_code=ErrorCodes.MCP_TIMEOUT,
            details=details,
            **kwargs
        )


class MCPPermissionError(MCPError):
    """Raised when MCP operation is denied due to permissions."""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if operation:
            details["operation"] = operation
        
        super().__init__(
            message,
            error_code=ErrorCodes.MCP_PERMISSION_DENIED,
            details=details,
            **kwargs
        )


class MCPInvalidResponseError(MCPError):
    """Raised when MCP server returns invalid response."""
    
    def __init__(self, message: str, response_data: Optional[Dict[str, Any]] = None, **kwargs):
        details = kwargs.get("details", {})
        if response_data:
            details["response_data"] = response_data
        
        super().__init__(
            message,
            error_code=ErrorCodes.MCP_INVALID_RESPONSE,
            details=details,
            **kwargs
        )
