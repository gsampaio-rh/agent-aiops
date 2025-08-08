"""
LLM Service Interface

Defines the contract for Language Model services like Ollama.
This interface ensures consistent API across different LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator, Optional


class LLMServiceInterface(ABC):
    """Abstract interface for Language Model services."""
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List[str]: Available model names
            
        Raises:
            ServiceError: If unable to fetch models
        """
        pass
    
    @abstractmethod
    def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.9
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream chat completion from the model.
        
        Args:
            model: Model name to use
            messages: List of chat messages in format [{"role": "user", "content": "..."}]
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            
        Yields:
            Dict containing response chunks and metadata
            
        Raises:
            ServiceError: If the request fails
            ValidationError: If parameters are invalid
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the service is available and healthy.
        
        Returns:
            bool: True if service is healthy, False otherwise
        """
        pass
    
    def validate_parameters(self, temperature: float, max_tokens: int, top_p: float) -> None:
        """
        Validate model parameters.
        
        Args:
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            
        Raises:
            ValidationError: If any parameter is invalid
        """
        from core.exceptions import ValidationError
        
        if not 0.0 <= temperature <= 2.0:
            raise ValidationError(f"Temperature must be between 0.0 and 2.0, got {temperature}")
        
        if not 1 <= max_tokens <= 10000:
            raise ValidationError(f"Max tokens must be between 1 and 10000, got {max_tokens}")
        
        if not 0.0 <= top_p <= 1.0:
            raise ValidationError(f"Top-p must be between 0.0 and 1.0, got {top_p}")
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model: Model name
            
        Returns:
            Dict[str, Any]: Model information and capabilities
            
        Raises:
            ServiceError: If model info cannot be retrieved
        """
        # Default implementation - subclasses can override
        return {
            "name": model,
            "status": "available" if model in self.get_available_models() else "unavailable"
        }
