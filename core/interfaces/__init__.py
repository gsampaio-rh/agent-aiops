"""
Core interfaces for Agent-AIOps services.

This module defines abstract base classes that establish contracts
for all service implementations, ensuring consistency and enabling
dependency injection.
"""

from .llm_service import LLMServiceInterface
from .agent_service import AgentServiceInterface, ToolInterface
from .search_service import SearchServiceInterface

__all__ = [
    "LLMServiceInterface",
    "AgentServiceInterface", 
    "ToolInterface",
    "SearchServiceInterface"
]
