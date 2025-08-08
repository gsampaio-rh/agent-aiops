"""
Service-specific exceptions.
"""

from typing import Optional, Dict, Any
from .base import ServiceError, ErrorCodes


class OllamaServiceError(ServiceError):
    """Base exception for Ollama service errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            service_name="ollama",
            error_code=kwargs.get("error_code", ErrorCodes.OLLAMA_REQUEST_FAILED),
            **kwargs
        )


class OllamaConnectionError(OllamaServiceError):
    """Raised when connection to Ollama fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCodes.OLLAMA_CONNECTION_FAILED,
            **kwargs
        )


class OllamaModelError(OllamaServiceError):
    """Raised when model-related operations fail."""
    
    def __init__(self, message: str, model: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if model:
            details["model"] = model
        
        super().__init__(
            message,
            error_code=ErrorCodes.OLLAMA_MODEL_NOT_FOUND,
            details=details,
            **kwargs
        )


class SearchError(ServiceError):
    """Base exception for search service errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            service_name="search",
            error_code=kwargs.get("error_code", ErrorCodes.SEARCH_PROVIDER_FAILED),
            **kwargs
        )


class SearchProviderError(SearchError):
    """Raised when a search provider fails."""
    
    def __init__(self, message: str, provider: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if provider:
            details["provider"] = provider
        
        super().__init__(
            message,
            error_code=ErrorCodes.SEARCH_PROVIDER_FAILED,
            details=details,
            **kwargs
        )


class SearchTimeoutError(SearchError):
    """Raised when search operations timeout."""
    
    def __init__(self, message: str, timeout: Optional[int] = None, **kwargs):
        details = kwargs.get("details", {})
        if timeout:
            details["timeout"] = timeout
        
        super().__init__(
            message,
            error_code=ErrorCodes.SEARCH_TIMEOUT,
            details=details,
            **kwargs
        )
