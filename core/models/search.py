"""
Search-related data models.

Contains models for search queries, responses, and results.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SearchQuery:
    """
    Represents a search query with all parameters.
    """
    
    query: str
    provider: str = "duckduckgo"
    max_results: int = 10
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "provider": self.provider,
            "max_results": self.max_results,
            "parameters": self.parameters,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchQuery":
        """Create SearchQuery from dictionary."""
        return cls(
            query=data["query"],
            provider=data.get("provider", "duckduckgo"),
            max_results=data.get("max_results", 10),
            parameters=data.get("parameters", {}),
            timestamp=data.get("timestamp", time.time())
        )


@dataclass
class SearchResult:
    """
    Represents a single search result.
    """
    
    title: str
    url: str
    snippet: str
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        """Create SearchResult from dictionary."""
        return cls(
            title=data["title"],
            url=data["url"],
            snippet=data["snippet"],
            source=data.get("source", ""),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time())
        )
    
    def get_formatted_result(self, max_snippet_length: int = 200) -> str:
        """Get formatted string representation of the result."""
        snippet = self.snippet
        if len(snippet) > max_snippet_length:
            snippet = snippet[:max_snippet_length] + "..."
        
        return f"**{self.title}**\n{snippet}\nSource: {self.url}"


@dataclass
class SearchResponse:
    """
    Complete response from a search operation.
    """
    
    query: str
    provider: str
    provider_name: str = ""
    results: List[SearchResult] = field(default_factory=list)
    total_results: int = 0
    search_time_ms: int = 0
    timestamp: float = field(default_factory=time.time)
    status: str = "success"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: SearchResult) -> None:
        """Add a search result to the response."""
        self.results.append(result)
        self.total_results = len(self.results)
    
    def is_successful(self) -> bool:
        """Check if the search was successful."""
        return self.status == "success" and self.error is None
    
    def get_formatted_results(self, max_results: Optional[int] = None) -> str:
        """Get formatted string representation of all results."""
        if not self.results:
            return "No search results found."
        
        results_to_show = self.results[:max_results] if max_results else self.results
        
        formatted_results = []
        for i, result in enumerate(results_to_show, 1):
            formatted_results.append(f"{i}. {result.get_formatted_result()}")
        
        return "\n\n".join(formatted_results)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "provider": self.provider,
            "provider_name": self.provider_name,
            "results": [result.to_dict() for result in self.results],
            "total_results": self.total_results,
            "search_time_ms": self.search_time_ms,
            "timestamp": self.timestamp,
            "status": self.status,
            "error": self.error,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResponse":
        """Create SearchResponse from dictionary."""
        response = cls(
            query=data["query"],
            provider=data["provider"],
            provider_name=data.get("provider_name", ""),
            total_results=data.get("total_results", 0),
            search_time_ms=data.get("search_time_ms", 0),
            timestamp=data.get("timestamp", time.time()),
            status=data.get("status", "success"),
            error=data.get("error"),
            metadata=data.get("metadata", {})
        )
        
        for result_data in data.get("results", []):
            response.add_result(SearchResult.from_dict(result_data))
        
        return response
