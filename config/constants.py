"""
Configuration constants for the Agent-AIOps application.
"""

from typing import Dict, Any, List

# Agent configuration
AGENT_CONFIG: Dict[str, Any] = {
    "max_iterations": 10,
    "enable_web_search": True,
    "enable_terminal": True,
    "show_thinking_process": True,
    "mcp_servers": ["desktop_commander"],
    "enable_conversation_memory": True,
    "max_conversation_history": 20,  # Maximum messages to keep in context
}

# MCP (Model Context Protocol) configuration
MCP_CONFIG: Dict[str, Any] = {
    "desktop_commander": {
        "enabled": True,
        "command_timeout": 30,
        "max_concurrent_commands": 3,
        "require_confirmation": True,
        "allowed_directories": [
            "~/",
            "/tmp/",
            "/var/log/",
            "./",  # Current directory
            "../"  # Parent directory (limited)
        ],
        "security_level": "restricted",  # strict, restricted, permissive
        "max_output_size": 10000,  # Maximum command output size in characters
        "connection_retry_attempts": 3,
        "connection_retry_delay": 2.0  # seconds
    }
}

# Terminal tool security settings
TERMINAL_SECURITY: Dict[str, Any] = {
    "enable_command_validation": True,
    "require_user_confirmation": True,
    "log_all_commands": True,
    "block_dangerous_commands": True,
    "max_execution_time": 300,  # 5 minutes
    "allowed_command_patterns": [
        r"^ls\b",
        r"^pwd$",
        r"^cd\s+",
        r"^cat\s+",
        r"^grep\s+",
        r"^find\s+",
        r"^git\s+",
        r"^npm\s+",
        r"^pip\s+",
        r"^python\d?\s+",
        r"^node\s+",
        r"^docker\s+",
        r"^kubectl\s+"
    ],
    "forbidden_command_patterns": [
        r"\brm\s+-rf\s+/",
        r"\bsudo\b",
        r"\bsu\b",
        r"\bchmod\s+777",
        r"\bmkfs\b",
        r"\bformat\b",
        r"\bshutdown\b",
        r"\breboot\b"
    ]
}

# Search configuration
SEARCH_CONFIG: Dict[str, Any] = {
    "providers": {
        "duckduckgo": {
            "enabled": True,
            "priority": 1,
            "timeout": 10,
            "max_results": 5
        },
        "searx": {
            "enabled": True,
            "priority": 2,
            "timeout": 15,
            "max_results": 5,
            "instances": [
                "https://search.sapti.me",
                "https://searx.be",
                "https://searx.info"
            ]
        }
    },
    "fallback_enabled": True,
    "cache_results": True,
    "cache_ttl": 3600  # 1 hour
}

# Legacy search providers configuration (for backward compatibility)
SEARCH_PROVIDERS: Dict[str, Any] = {
    "duckduckgo": {
        "name": "DuckDuckGo",
        "enabled": True,
        "timeout": 10,
        "max_results": 5,
        "base_url": "https://duckduckgo.com"
    },
    "searx": {
        "name": "SearX",
        "enabled": True,
        "timeout": 15,
        "max_results": 5,
        "instances": [
            "https://search.sapti.me",
            "https://searx.be",
            "https://searx.info"
        ]
    }
}

# Tool execution settings
TOOL_EXECUTION: Dict[str, Any] = {
    "max_parallel_tools": 2,
    "tool_timeout": 60,
    "enable_tool_caching": True,
    "cache_ttl": 1800,  # 30 minutes
    "retry_failed_tools": True,
    "max_retries": 2
}
