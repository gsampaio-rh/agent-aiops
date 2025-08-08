"""
Desktop Commander MCP Client Service

Implements MCP client for communicating with the Desktop Commander MCP server.
Provides terminal command execution, file operations, and process management.
"""

import asyncio
import json
import subprocess
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from core.interfaces.mcp_service import MCPServiceInterface
from core.exceptions import (
    MCPError, MCPConnectionError, MCPToolError, MCPTimeoutError, 
    MCPInvalidResponseError, ErrorCodes
)
from utils.logger import get_logger, log_performance


class DesktopCommanderMCP(MCPServiceInterface):
    """Client for Desktop Commander MCP server."""
    
    def __init__(self, timeout: int = 30):
        self.logger = get_logger(__name__)
        self.process: Optional[subprocess.Popen] = None
        self.connected = False
        self.timeout = timeout
        self.request_id = 0
        
        # Available tools cache
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._resources_cache: Optional[List[Dict[str, Any]]] = None
    
    def _get_next_request_id(self) -> int:
        """Get next request ID for JSON-RPC."""
        self.request_id += 1
        return self.request_id
    
    async def _send_json_rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send JSON-RPC request to MCP server.
        
        Args:
            method: JSON-RPC method name
            params: Method parameters
            
        Returns:
            Dict[str, Any]: JSON-RPC response
            
        Raises:
            MCPError: If request fails
        """
        if not self.process:
            raise MCPConnectionError("MCP process not started")
        
        request = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": method
        }
        
        if params:
            request["params"] = params
        
        try:
            # Send request
            request_line = json.dumps(request) + "\n"
            self.process.stdin.write(request_line)
            self.process.stdin.flush()
            
            self.logger.debug(f"Sent MCP request: {method}", request_id=request["id"])
            
            # Read response with timeout - may need to skip notifications
            response = None
            request_id = request["id"]
            
            try:
                # Keep reading until we get a response with matching ID or timeout
                start_time = time.time()
                while time.time() - start_time < self.timeout:
                    response_line = await asyncio.wait_for(
                        asyncio.create_task(self._read_line()),
                        timeout=self.timeout
                    )
                    
                    if not response_line:
                        raise MCPConnectionError("MCP server closed connection")
                    
                    # Parse response
                    try:
                        parsed_response = json.loads(response_line.strip())
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON in MCP response: {e}")
                        continue
                    
                    # Check if this is a notification (no id field) or response
                    if "id" in parsed_response and parsed_response["id"] == request_id:
                        # This is our response
                        response = parsed_response
                        break
                    elif "method" in parsed_response and "notifications/" in parsed_response["method"]:
                        # This is a notification, log it and continue
                        self.logger.debug(f"MCP notification: {parsed_response.get('method')}", 
                                        notification=parsed_response)
                        continue
                    else:
                        # Some other response, log and continue
                        self.logger.warning(f"Unexpected MCP response", response=parsed_response)
                        continue
                
                if response is None:
                    raise MCPTimeoutError(
                        f"No matching response received after {self.timeout}s",
                        timeout_seconds=self.timeout,
                        details={"method": method, "request_id": request_id}
                    )
                    
            except asyncio.TimeoutError:
                raise MCPTimeoutError(
                    f"MCP request timed out after {self.timeout}s",
                    timeout_seconds=self.timeout,
                    details={"method": method, "request_id": request_id}
                )
            
            # Check for JSON-RPC error
            if "error" in response:
                error = response["error"]
                raise MCPError(
                    f"MCP server error: {error.get('message', 'Unknown error')}",
                    details={
                        "error_code": error.get("code"),
                        "error_data": error.get("data"),
                        "method": method
                    }
                )
            
            self.logger.debug(f"Received MCP response: {method}", 
                            request_id=request["id"],
                            response_keys=list(response.keys()),
                            full_response=response)
            
            result = response.get("result", {})
            self.logger.debug(f"Extracted result from MCP response", result=result)
            
            return result
            
        except subprocess.BrokenPipeError:
            raise MCPConnectionError("MCP server process terminated")
        except Exception as e:
            if isinstance(e, (MCPError, MCPConnectionError, MCPTimeoutError, MCPInvalidResponseError)):
                raise
            raise MCPError(f"Unexpected error in MCP communication: {str(e)}", original_exception=e)
    
    async def _read_line(self) -> str:
        """Read a line from MCP server output asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process.stdout.readline)
    
    @log_performance("mcp_connect")
    async def connect(self) -> bool:
        """
        Connect to Desktop Commander MCP server.
        
        Returns:
            bool: True if connection successful
            
        Raises:
            MCPConnectionError: If connection fails
        """
        if self.connected:
            self.logger.warning("Already connected to MCP server")
            return True
        
        try:
            self.logger.info("Starting Desktop Commander MCP server")
            
            # Start MCP server process
            self.process = subprocess.Popen(
                ["npx", "-y", "@wonderwhy-er/desktop-commander"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0  # Unbuffered for real-time communication
            )
            
            # Wait a moment for server to start
            await asyncio.sleep(1)
            
            # Check if process is still running
            if self.process.poll() is not None:
                stderr_output = self.process.stderr.read()
                raise MCPConnectionError(
                    f"MCP server failed to start: {stderr_output}",
                    details={"return_code": self.process.returncode}
                )
            
            # Initialize MCP connection
            await self._initialize_mcp()
            
            self.connected = True
            self.logger.info("Successfully connected to Desktop Commander MCP")
            return True
            
        except Exception as e:
            if self.process:
                self.process.terminate()
                self.process = None
            
            if isinstance(e, MCPConnectionError):
                raise
            
            raise MCPConnectionError(
                f"Failed to connect to MCP server: {str(e)}",
                original_exception=e
            )
    
    async def _initialize_mcp(self) -> None:
        """Initialize MCP connection with handshake."""
        try:
            # Send initialize request
            init_response = await self._send_json_rpc("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {}
                },
                "clientInfo": {
                    "name": "agent-aiops",
                    "version": "1.0.0"
                }
            })
            
            self.logger.debug("MCP initialization successful", 
                            protocol_version=init_response.get("protocolVersion"),
                            server_name=init_response.get("serverInfo", {}).get("name"))
            
            # Send initialized notification  
            # Note: This is a notification, so we don't expect a response
            request = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            request_line = json.dumps(request) + "\n"
            self.process.stdin.write(request_line)
            self.process.stdin.flush()
            
        except Exception as e:
            raise MCPConnectionError(f"MCP initialization failed: {str(e)}", original_exception=e)
    
    @log_performance("mcp_disconnect")
    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        if not self.connected:
            return
        
        try:
            if self.process:
                self.process.terminate()
                
                # Wait for graceful shutdown
                try:
                    await asyncio.wait_for(
                        asyncio.create_task(self._wait_for_process()),
                        timeout=5
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("MCP server did not terminate gracefully, killing")
                    self.process.kill()
                
                self.process = None
            
            self.connected = False
            self._tools_cache = None
            self._resources_cache = None
            
            self.logger.info("Disconnected from MCP server")
            
        except Exception as e:
            self.logger.error(f"Error during MCP disconnect: {e}")
            raise MCPError(f"Failed to disconnect: {str(e)}", original_exception=e)
    
    async def _wait_for_process(self) -> None:
        """Wait for MCP process to terminate."""
        if self.process:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.process.wait)
    
    @log_performance("mcp_execute_tool")
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an MCP tool.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Dict[str, Any]: Tool execution results
            
        Raises:
            MCPToolError: If tool execution fails
        """
        self.validate_connection()
        
        try:
            self.logger.info(f"Executing MCP tool: {tool_name}", 
                           tool_name=tool_name,
                           arguments=arguments)
            
            result = await self._send_json_rpc("tools/call", {
                "name": tool_name,
                "arguments": arguments
            })
            
            self.logger.info(f"MCP tool execution completed: {tool_name}",
                           tool_name=tool_name,
                           success=True)
            
            return result
            
        except MCPError:
            raise
        except Exception as e:
            raise MCPToolError(
                f"Tool execution failed: {str(e)}",
                tool_name=tool_name,
                details={"arguments": arguments},
                original_exception=e
            )
    
    @log_performance("mcp_list_tools")
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available MCP tools.
        
        Returns:
            List[Dict[str, Any]]: Available tool information
        """
        self.validate_connection()
        
        if self._tools_cache is not None:
            return self._tools_cache
        
        try:
            result = await self._send_json_rpc("tools/list")
            tools = result.get("tools", [])
            
            self._tools_cache = tools
            self.logger.info(f"Retrieved {len(tools)} MCP tools", tool_count=len(tools))
            
            return tools
            
        except Exception as e:
            raise MCPError(f"Failed to list tools: {str(e)}", original_exception=e)
    
    @log_performance("mcp_list_resources")
    async def list_resources(self) -> List[Dict[str, Any]]:
        """
        Get list of available MCP resources.
        
        Returns:
            List[Dict[str, Any]]: Available resource information
        """
        self.validate_connection()
        
        if self._resources_cache is not None:
            return self._resources_cache
        
        try:
            result = await self._send_json_rpc("resources/list")
            resources = result.get("resources", [])
            
            self._resources_cache = resources
            self.logger.info(f"Retrieved {len(resources)} MCP resources", resource_count=len(resources))
            
            return resources
            
        except Exception as e:
            raise MCPError(f"Failed to list resources: {str(e)}", original_exception=e)
    
    def is_connected(self) -> bool:
        """Check if connected to MCP server."""
        return self.connected and self.process and self.process.poll() is None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
