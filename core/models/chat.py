"""
Chat-related data models.

Contains models for chat messages and sessions.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class MessageRole(Enum):
    """Chat message roles."""
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """
    Represents a single chat message.
    """
    
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """Create ChatMessage from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def user_message(cls, content: str, **metadata) -> "ChatMessage":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content, metadata=metadata)
    
    @classmethod
    def assistant_message(cls, content: str, **metadata) -> "ChatMessage":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content, metadata=metadata)
    
    @classmethod
    def system_message(cls, content: str, **metadata) -> "ChatMessage":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content, metadata=metadata)


@dataclass
class ChatSession:
    """
    Represents a chat session with multiple messages.
    """
    
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the session."""
        self.messages.append(message)
        self.updated_at = time.time()
    
    def add_user_message(self, content: str, **metadata) -> None:
        """Add a user message to the session."""
        message = ChatMessage.user_message(content, **metadata)
        self.add_message(message)
    
    def add_assistant_message(self, content: str, **metadata) -> None:
        """Add an assistant message to the session."""
        message = ChatMessage.assistant_message(content, **metadata)
        self.add_message(message)
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Get messages in format suitable for LLM APIs."""
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in self.messages
        ]
    
    def get_last_n_messages(self, n: int) -> List[ChatMessage]:
        """Get the last n messages from the session."""
        return self.messages[-n:] if n > 0 else []
    
    def clear_messages(self) -> None:
        """Clear all messages from the session."""
        self.messages.clear()
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatSession":
        """Create ChatSession from dictionary."""
        session = cls(
            session_id=data["session_id"],
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata", {})
        )
        
        for msg_data in data.get("messages", []):
            session.add_message(ChatMessage.from_dict(msg_data))
        
        return session
