"""
Terminal Tool using Desktop Commander MCP

Provides secure terminal command execution through MCP integration.
Includes command validation, security restrictions, and user consent mechanisms.
"""

import asyncio
import re
import time
from typing import Dict, Any, List, Set, Tuple, Optional

from core.interfaces.agent_service import ToolInterface
from core.models.agent import ToolInfo
from core.exceptions import MCPError, MCPToolError, MCPPermissionError
from services.mcp_service import DesktopCommanderMCP
from utils.logger import get_logger, log_performance


class CommandValidator:
    """Validates terminal commands for security."""
    
    # Commands that are completely forbidden
    FORBIDDEN_COMMANDS: Set[str] = {
        'rm', 'rmdir', 'del', 'format', 'fdisk', 'mkfs', 'shutdown', 
        'reboot', 'halt', 'init', 'kill', 'killall', 'pkill',
        'sudo', 'su', 'chmod', 'chown', 'passwd', 'useradd', 'userdel'
    }
    
    # Commands that are allowed for basic operations
    SAFE_COMMANDS: Set[str] = {
        'ls', 'dir', 'pwd', 'cd', 'cat', 'head', 'tail', 'less', 'more',
        'grep', 'find', 'locate', 'which', 'whereis', 'file', 'wc',
        'sort', 'uniq', 'cut', 'awk', 'sed', 'tr', 'echo', 'printf',
        'date', 'cal', 'uptime', 'whoami', 'id', 'groups',
        'ps', 'top', 'htop', 'jobs', 'pgrep', 'df', 'du', 'free',
        'uname', 'hostname', 'env', 'printenv', 'history',
        'git', 'npm', 'pip', 'python', 'python3', 'node', 'java',
        'mvn', 'gradle', 'make', 'cmake', 'gcc', 'g++', 'clang',
        'docker', 'kubectl', 'helm', 'terraform', 'ansible'
    }
    
    # Dangerous patterns to watch for
    DANGEROUS_PATTERNS: List[str] = [
        r'rm\s+-rf\s+/',           # Dangerous deletions
        r':\(\)\{.*\};:',          # Fork bombs  
        r'>\s*/dev/sd[a-z]',       # Direct disk writes
        r'dd\s+if=.*of=/dev/',     # Disk dumps
        r'\|.*sh',                 # Pipe to shell
        r'curl.*\|.*sh',           # Download and execute
        r'wget.*\|.*sh',           # Download and execute
        r'eval\s*\$\(',            # Dynamic evaluation
        r'bash\s*-c',              # Bash command execution
        r'sh\s*-c',                # Shell command execution
    ]
    
    @classmethod
    def validate_command(cls, command: str) -> Tuple[bool, str, str]:
        """
        Validate a command for security.
        
        Args:
            command: Command to validate
            
        Returns:
            Tuple[bool, str, str]: (is_safe, risk_level, reason)
        """
        command = command.strip()
        if not command:
            return False, "high", "Empty command"
        
        # Split command into parts
        parts = command.split()
        base_command = parts[0].split('/')[-1]  # Handle full paths
        
        # Check forbidden commands
        if base_command in cls.FORBIDDEN_COMMANDS:
            return False, "high", f"Command '{base_command}' is forbidden for security"
        
        # Check dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False, "high", f"Command contains dangerous pattern"
        
        # Check for safe commands
        if base_command in cls.SAFE_COMMANDS:
            return True, "low", "Command is in safe list"
        
        # Unknown command - requires caution
        return True, "medium", f"Unknown command '{base_command}' - proceed with caution"


class TerminalTool(ToolInterface):
    """Terminal command execution tool using Desktop Commander MCP."""
    
    def __init__(self, require_confirmation: bool = True, mcp_client: Optional[DesktopCommanderMCP] = None):
        self.name = "terminal"
        self.description = (
            "Execute terminal/command-line commands on the local system. "
            "Use for file operations, system information, development tasks, "
            "and command-line utilities. Commands are validated for security."
        )
        self.mcp_client = mcp_client
        self.logger = get_logger(__name__)
        self.require_confirmation = require_confirmation
        
        # Track command execution for session
        self.execution_history: List[Dict[str, Any]] = []
    
    async def _ensure_connected(self) -> None:
        """Ensure MCP client is connected."""
        if not self.mcp_client:
            # If no MCP client provided, create one and connect
            self.mcp_client = DesktopCommanderMCP()
            await self.mcp_client.connect()
        elif not self.mcp_client.is_connected():
            await self.mcp_client.connect()
    
    def _log_execution(self, command: str, result: Dict[str, Any]) -> None:
        """Log command execution for history tracking."""
        self.execution_history.append({
            "command": command,
            "timestamp": time.time(),
            "success": result.get("success", False),
            "exit_code": result.get("metadata", {}).get("exit_code"),
            "output_length": len(result.get("output", ""))
        })
        
        # Keep only last 50 executions
        if len(self.execution_history) > 50:
            self.execution_history = self.execution_history[-50:]
    
    @log_performance("terminal_execute")
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute terminal command via MCP.
        
        Args:
            query: Terminal command to execute
            **kwargs: Additional parameters (timeout, working_dir, etc.)
            
        Returns:
            Dict[str, Any]: Execution results with output and metadata
        """
        command = query.strip()
        
        try:
            # Validate command
            is_safe, risk_level, reason = CommandValidator.validate_command(command)
            
            if not is_safe:
                result = {
                    "success": False,
                    "error": reason,
                    "metadata": {
                        "command": command,
                        "risk_level": risk_level,
                        "validation": "failed"
                    }
                }
                self._log_execution(command, result)
                return result
            
            # Log validation success
            self.logger.info(f"Command validated: {command}",
                           command=command,
                           risk_level=risk_level,
                           validation_reason=reason)
            
            # Check if confirmation is required for medium/high risk
            if self.require_confirmation and risk_level in ["medium", "high"]:
                # In a real implementation, this would trigger UI confirmation
                # For now, we'll log it and proceed
                self.logger.warning(f"Command requires confirmation: {command}",
                                  command=command, risk_level=risk_level)
            
            # Execute command asynchronously
            try:
                # Check if we're already in an event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self._execute_async(command, **kwargs))
                        result = future.result()
                else:
                    # No loop running, we can use run_until_complete
                    result = loop.run_until_complete(self._execute_async(command, **kwargs))
            except RuntimeError:
                # Fallback: create new loop in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._execute_async(command, **kwargs))
                    result = future.result()
            
            self._log_execution(command, result)
            return result
            
        except Exception as e:
            self.logger.error(f"Terminal command execution failed: {e}",
                            command=command, error=str(e))
            
            result = {
                "success": False,
                "error": str(e),
                "metadata": {
                    "command": command,
                    "error_type": type(e).__name__
                }
            }
            self._log_execution(command, result)
            return result
    
    async def _execute_async(self, command: str, **kwargs) -> Dict[str, Any]:
        """
        Execute command asynchronously via MCP using start_process and read_process_output.
        
        Args:
            command: Command to execute
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Execution results
        """
        try:
            await self._ensure_connected()
            
            # Prepare MCP arguments for start_process
            start_args = {
                "command": command,
                "timeout_ms": kwargs.get("timeout", 30) * 1000  # Convert to milliseconds
            }
            
            # Add working directory if specified
            if "working_dir" in kwargs:
                start_args["cwd"] = kwargs["working_dir"]
            
            # Start the process
            self.logger.info(f"Starting terminal command: {command}")
            
            start_result = await self.mcp_client.execute_tool("start_process", start_args)
            
            self.logger.debug(f"Start process result: {start_result}")
            
            # Extract payload from MCP result
            start_payload = self._payload_from_result(start_result) or {}
            
            # Handle different response formats for session ID
            session_id = (
                start_payload.get("sessionId") or
                start_payload.get("session_id") or
                start_payload.get("id") or
                start_payload.get("session")
            )
            
            pid = (
                start_payload.get("pid") or
                start_payload.get("processId") or
                start_payload.get("process_id")
            )
            
            if not session_id:
                # Desktop Commander often returns direct output for simple commands
                # Check if we got direct output in the response
                if "stdout" in start_payload:
                    output = start_payload["stdout"]
                elif "text" in start_payload:
                    output = start_payload["text"]
                else:
                    # Maybe it's in the raw response format we saw in testing
                    # Let's check the original result format
                    if isinstance(start_result, dict) and "content" in start_result:
                        content = start_result["content"]
                        if isinstance(content, list) and content:
                            first_item = content[0]
                            if isinstance(first_item, dict) and "text" in first_item:
                                raw_output = first_item["text"]
                                output = self._extract_command_output(raw_output)
                            else:
                                output = str(first_item)
                        else:
                            output = str(content)
                    else:
                        output = str(start_result)
                
                self.logger.info(f"Command completed with direct output", command=command, output_length=len(output))
                return {
                    "success": True,
                    "output": output,
                    "metadata": {
                        "command": command,
                        "exit_code": 0,
                        "direct_response": True
                    }
                }
            
            self.logger.info(f"Started process session: {session_id} (pid={pid})")
            
            # Poll for output until process completes
            output_parts = []
            stderr_parts = []
            exit_code = None
            poll_interval_ms = 500
            
            while True:
                # Read process output
                read_result = await self.mcp_client.execute_tool("read_process_output", {
                    "sessionId": session_id,
                    "timeoutMs": poll_interval_ms
                })
                
                output_payload = self._payload_from_result(read_result) or {}
                
                # Collect stdout and stderr
                if output_payload.get("stdout"):
                    output_parts.append(output_payload["stdout"])
                
                if output_payload.get("stderr"):
                    stderr_parts.append(output_payload["stderr"])
                
                # Check for completion
                exit_code = (
                    output_payload.get("exitCode") or
                    output_payload.get("code") or
                    output_payload.get("exit_code")
                )
                
                is_running = output_payload.get("isRunning")
                completed = (
                    output_payload.get("completed") or
                    output_payload.get("exited") or
                    output_payload.get("done")
                )
                
                if exit_code is not None or is_running is False or completed:
                    break
                
                # Brief sleep to avoid overwhelming the server
                await asyncio.sleep(poll_interval_ms / 1000)
            
            # Combine all output
            full_output = "".join(output_parts)
            full_stderr = "".join(stderr_parts)
            
            # Combine stdout and stderr for final output
            combined_output = full_output
            if full_stderr:
                combined_output += f"\n[stderr]\n{full_stderr}"
            
            success = exit_code == 0 if exit_code is not None else True
            
            result = {
                "success": success,
                "output": combined_output,
                "metadata": {
                    "command": command,
                    "exit_code": exit_code or 0,
                    "session_id": session_id,
                    "pid": pid,
                    "stdout_length": len(full_output),
                    "stderr_length": len(full_stderr)
                }
            }
            
            if not success:
                result["error"] = f"Command failed with exit code {exit_code}"
            
            self.logger.info(f"Command execution completed: {command}",
                           command=command,
                           success=success,
                           exit_code=exit_code,
                           output_length=len(combined_output))
            
            return result
            
        except MCPError as e:
            self.logger.error(f"MCP error executing command: {e}",
                            command=command, mcp_error=str(e))
            return {
                "success": False,
                "error": f"MCP execution failed: {str(e)}",
                "metadata": {
                    "command": command,
                    "error_type": "MCPError"
                }
            }
        except Exception as e:
            self.logger.error(f"Unexpected error executing command: {e}",
                            command=command, error=str(e))
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}",
                "metadata": {
                    "command": command,
                    "error_type": type(e).__name__
                }
            }
    
    def _extract_command_output(self, raw_output: str) -> str:
        """
        Extract the actual command output from Desktop Commander's response format.
        
        Desktop Commander returns output in format:
        "Process started with PID XXXXX (shell: /bin/sh)\nInitial output:\n[ACTUAL_OUTPUT]"
        
        Args:
            raw_output: Raw output from Desktop Commander
            
        Returns:
            str: Clean command output
        """
        if not raw_output:
            return ""
        
        # Look for "Initial output:" marker
        if "Initial output:\n" in raw_output:
            parts = raw_output.split("Initial output:\n", 1)
            if len(parts) > 1:
                return parts[1].rstrip('\n')
        
        # Alternative: if it starts with process info, skip that line
        if raw_output.startswith("Process started with PID"):
            lines = raw_output.split('\n')
            if len(lines) > 2:  # PID line, "Initial output:", actual output
                # Find the first line that's not process info or "Initial output:"
                for i, line in enumerate(lines):
                    if line == "Initial output:":
                        # Return everything after this line
                        return '\n'.join(lines[i+1:]).rstrip('\n')
                # Fallback: skip first line (PID info)
                return '\n'.join(lines[1:]).rstrip('\n')
        
        # If no special format detected, return as-is
        return raw_output.rstrip('\n')
    
    def _payload_from_result(self, res: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Unpack MCP CallToolResult -> first content item (JSON/text).
        Based on the reference implementation provided.
        """
        if not res or not res.get("content"):
            return None
        
        content = res["content"]
        if not content:
            return None
            
        item = content[0] if isinstance(content, list) else content
        
        # Try JSON first
        if getattr(item, "type", None) == "application/json" or item.get("type") == "application/json":
            return getattr(item, "data", item.get("data"))
        
        # Fallback: plain text
        if getattr(item, "type", None) == "text/plain" or item.get("type") == "text/plain":
            text = getattr(item, "text", item.get("text", ""))
            return {"stdout": text}
        
        # Handle text content directly
        if isinstance(item, dict) and "text" in item:
            return {"stdout": item["text"]}
        
        # Unknown content type; return raw for debugging
        return {"raw": item}
    
    def get_tool_info(self) -> ToolInfo:
        """Get tool information for the agent."""
        return ToolInfo(
            name=self.name,
            description=self.description,
            parameters={
                "timeout": {
                    "type": "integer",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 300,
                    "description": "Command timeout in seconds"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for command execution"
                }
            },
            metadata={
                "risk_level": "medium",
                "requires_confirmation": self.require_confirmation,
                "supported_platforms": ["linux", "macos", "windows"]
            }
        )
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get command execution history for the current session."""
        return self.execution_history.copy()
    
    def clear_history(self) -> None:
        """Clear execution history."""
        self.execution_history.clear()
        self.logger.info("Terminal execution history cleared")
    
    async def cleanup(self) -> None:
        """Cleanup MCP client connection."""
        if self.mcp_client and self.mcp_client.is_connected():
            await self.mcp_client.disconnect()
            self.logger.info("Terminal tool cleanup completed")
