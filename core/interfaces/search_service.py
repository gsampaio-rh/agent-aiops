"""
Search Service Interface

Defines the contract for web search services with multiple providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from core.models.search import SearchQuery, SearchResponse


class SearchServiceInterface(ABC):
    """Abstract interface for search services."""
    
    @abstractmethod
    def search(self, query: SearchQuery) -> SearchResponse:
        """
        Perform web search using specified provider.
        
        Args:
            query: SearchQuery object with search parameters
            
        Returns:
            SearchResponse: Complete search response with results and metadata
            
        Raises:
            SearchError: If search fails
        """
        pass
    
    @abstractmethod
    def get_available_providers(self) -> List[str]:
        """
        Get list of available search providers.
        
        Returns:
            List[str]: Available provider names
        """
        pass
    
    @abstractmethod
    def test_provider(self, provider: str) -> Dict[str, Any]:
        """
        Test if a specific provider is working.
        
        Args:
            provider: Provider name to test
            
        Returns:
            Dict[str, Any]: Test results with status and performance metrics
        """
        pass
    
    def validate_query(self, query: SearchQuery) -> None:
        """
        Validate search query parameters.
        
        Args:
            query: Search query to validate
            
        Raises:
            ValidationError: If query parameters are invalid
        """
        from core.exceptions import ValidationError
        
        if not query.query or not query.query.strip():
            raise ValidationError("Search query cannot be empty")
        
        if len(query.query) > 1000:
            raise ValidationError("Search query is too long (max 1,000 characters)")
        
        if query.max_results < 1 or query.max_results > 50:
            raise ValidationError("Max results must be between 1 and 50")
        
        available_providers = self.get_available_providers()
        if query.provider not in available_providers:
            raise ValidationError(
                f"Unknown provider '{query.provider}'. Available: {', '.join(available_providers)}"
            )
