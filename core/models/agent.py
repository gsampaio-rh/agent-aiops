"""
Agent-related data models.

Contains models for agent steps, responses, and tool information.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class StepType(Enum):
    """Types of agent reasoning steps."""
    
    THOUGHT = "thought"
    TOOL_SELECTION = "tool_selection"
    TOOL_USE = "tool_use"
    TOOL_EXECUTION_REQUEST = "tool_execution_request"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"


@dataclass
class AgentStep:
    """
    Represents a single step in the agent's reasoning process.
    """
    
    step_type: StepType
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "step_type": self.step_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentStep":
        """Create AgentStep from dictionary."""
        return cls(
            step_type=StepType(data["step_type"]),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {})
        )


@dataclass
class AgentResponse:
    """
    Complete response from an agent query including all steps.
    """
    
    query: str
    steps: List[AgentStep] = field(default_factory=list)
    total_time_ms: Optional[int] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: AgentStep) -> None:
        """Add a step to the response."""
        self.steps.append(step)
    
    def get_final_answer(self) -> Optional[str]:
        """Get the final answer from the response."""
        for step in reversed(self.steps):
            if step.step_type == StepType.FINAL_ANSWER:
                return step.content
        return None
    
    def get_steps_by_type(self, step_type: StepType) -> List[AgentStep]:
        """Get all steps of a specific type."""
        return [step for step in self.steps if step.step_type == step_type]
    
    def has_error(self) -> bool:
        """Check if the response contains any errors."""
        return self.error is not None or any(
            step.step_type == StepType.ERROR for step in self.steps
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "steps": [step.to_dict() for step in self.steps],
            "total_time_ms": self.total_time_ms,
            "error": self.error,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentResponse":
        """Create AgentResponse from dictionary."""
        response = cls(
            query=data["query"],
            total_time_ms=data.get("total_time_ms"),
            error=data.get("error"),
            metadata=data.get("metadata", {})
        )
        
        for step_data in data.get("steps", []):
            response.add_step(AgentStep.from_dict(step_data))
        
        return response


@dataclass
class ToolInfo:
    """
    Information about an agent tool.
    """
    
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolInfo":
        """Create ToolInfo from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data.get("parameters", {}),
            metadata=data.get("metadata", {})
        )
