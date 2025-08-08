"""
MCP Service Interface

Defines contracts for Model Context Protocol (MCP) client services.
Supports JSON-RPC communication with MCP servers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import asyncio


class MCPServiceInterface(ABC):
    """Abstract interface for MCP client services."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to MCP server.
        
        Returns:
            bool: True if connection successful, False otherwise
            
        Raises:
            MCPError: If connection fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from MCP server.
        
        Raises:
            MCPError: If disconnection fails
        """
        pass
    
    @abstractmethod
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an MCP tool with given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Dict[str, Any]: Tool execution results
            
        Raises:
            MCPError: If tool execution fails
        """
        pass
    
    @abstractmethod
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available MCP tools.
        
        Returns:
            List[Dict[str, Any]]: Available tool information
            
        Raises:
            MCPError: If listing tools fails
        """
        pass
    
    @abstractmethod
    async def list_resources(self) -> List[Dict[str, Any]]:
        """
        Get list of available MCP resources.
        
        Returns:
            List[Dict[str, Any]]: Available resource information
            
        Raises:
            MCPError: If listing resources fails
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if connected to MCP server.
        
        Returns:
            bool: True if connected, False otherwise
        """
        pass
    
    def validate_connection(self) -> None:
        """
        Validate that connection is active.
        
        Raises:
            MCPError: If not connected
        """
        from core.exceptions import MCPError, ErrorCodes
        
        if not self.is_connected():
            raise MCPError(
                "Not connected to MCP server",
                error_code=ErrorCodes.MCP_CONNECTION_FAILED,
                details={"service": self.__class__.__name__}
            )
