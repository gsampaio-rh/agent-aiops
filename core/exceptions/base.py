"""
Base exception classes and error codes.
"""

import time
from enum import Enum
from typing import Optional, Dict, Any


class ErrorCodes(Enum):
    """Standardized error codes for the application."""
    
    # Service errors (1000-1999)
    OLLAMA_CONNECTION_FAILED = "OLLAMA_001"
    OLLAMA_REQUEST_FAILED = "OLLAMA_002"
    OLLAMA_MODEL_NOT_FOUND = "OLLAMA_003"
    OLLAMA_INVALID_RESPONSE = "OLLAMA_004"
    
    # Agent errors (2000-2999)
    AGENT_INIT_FAILED = "AGENT_001"
    AGENT_PROCESSING_FAILED = "AGENT_002"
    AGENT_TOOL_FAILED = "AGENT_003"
    AGENT_INVALID_STATE = "AGENT_004"
    
    # Search errors (3000-3999)
    SEARCH_PROVIDER_FAILED = "SEARCH_001"
    SEARCH_TIMEOUT = "SEARCH_002"
    SEARCH_INVALID_QUERY = "SEARCH_003"
    SEARCH_NO_RESULTS = "SEARCH_004"
    
    # Validation errors (4000-4999)
    INVALID_QUERY = "VALIDATION_001"
    INVALID_PARAMETERS = "VALIDATION_002"
    INVALID_MODEL = "VALIDATION_003"
    INVALID_CONFIG = "VALIDATION_004"
    
    # General errors (5000-5999)
    UNEXPECTED_ERROR = "GENERAL_001"
    CONFIGURATION_ERROR = "GENERAL_002"
    DEPENDENCY_ERROR = "GENERAL_003"


class AgentAIOpsException(Exception):
    """Base exception for all Agent-AIOps errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCodes] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_exception = original_exception
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "message": self.message,
            "error_code": self.error_code.value if self.error_code else None,
            "details": self.details,
            "timestamp": self.timestamp,
            "exception_type": self.__class__.__name__,
            "original_exception": str(self.original_exception) if self.original_exception else None
        }
    
    def __str__(self) -> str:
        """String representation with error code."""
        if self.error_code:
            return f"[{self.error_code.value}] {self.message}"
        return self.message


class ValidationError(AgentAIOpsException):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        
        super().__init__(
            message,
            error_code=kwargs.get("error_code", ErrorCodes.INVALID_PARAMETERS),
            details=details,
            original_exception=kwargs.get("original_exception")
        )


class ServiceError(AgentAIOpsException):
    """Base class for service-related errors."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if service_name:
            details["service"] = service_name
        
        super().__init__(
            message,
            error_code=kwargs.get("error_code"),
            details=details,
            original_exception=kwargs.get("original_exception")
        )


def create_error_with_code(
    error_code: ErrorCodes,
    message: str,
    exception_class: type = AgentAIOpsException,
    **kwargs
) -> AgentAIOpsException:
    """
    Factory function to create exceptions with specific error codes.
    
    Args:
        error_code: The error code to assign
        message: Error message
        exception_class: Exception class to instantiate
        **kwargs: Additional arguments for the exception
    
    Returns:
        AgentAIOpsException: Configured exception instance
    """
    return exception_class(
        message=message,
        error_code=error_code,
        **kwargs
    )
