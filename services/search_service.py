"""
Enhanced Web Search Service

Provides multiple search providers with error handling and result normalization.
"""

import requests
import time
from typing import Dict, List, Any, Optional
from urllib.parse import quote_plus, urljoin
import json
import re
from bs4 import BeautifulSoup


class SearchResult:
    """Standardized search result data structure."""
    
    def __init__(self, title: str, url: str, snippet: str, source: str = "", 
                 metadata: Optional[Dict[str, Any]] = None):
        self.title = self._clean_text(title)
        self.url = url
        self.snippet = self._clean_text(snippet)
        self.source = source
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', str(text))
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class WebSearchService:
    """Enhanced web search service with multiple providers."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        self.timeout = 15
        
        self.providers = {
            "duckduckgo": {
                "name": "DuckDuckGo",
                "function": self._search_duckduckgo,
                "supports_instant": True
            },
            "searx": {
                "name": "SearX",
                "function": self._search_searx,
                "supports_instant": False
            },
            "brave": {
                "name": "Brave Search",
                "function": self._search_brave,
                "supports_instant": False
            },
            "startpage": {
                "name": "Startpage",
                "function": self._search_startpage,
                "supports_instant": False
            }
        }
    
    def get_available_providers(self) -> List[str]:
        """Get list of available search providers."""
        return list(self.providers.keys())
    
    def search(self, query: str, provider: str = "duckduckgo", 
               max_results: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Perform web search using specified provider.
        
        Args:
            query: Search query
            provider: Search provider to use
            max_results: Maximum number of results
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dict containing search results and metadata
        """
        start_time = time.time()
        
        try:
            if provider not in self.providers:
                raise ValueError(f"Unknown provider: {provider}. Available: {list(self.providers.keys())}")
            
            provider_info = self.providers[provider]
            search_function = provider_info["function"]
            
            results = search_function(query, max_results, **kwargs)
            
            # Convert SearchResult objects to dicts
            results_dict = [r.to_dict() if isinstance(r, SearchResult) else r for r in results]
            
            return {
                "query": query,
                "provider": provider,
                "provider_name": provider_info["name"],
                "results": results_dict,
                "total_results": len(results_dict),
                "search_time_ms": round((time.time() - start_time) * 1000),
                "timestamp": time.time(),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "query": query,
                "provider": provider,
                "provider_name": self.providers.get(provider, {}).get("name", provider),
                "results": [],
                "total_results": 0,
                "search_time_ms": round((time.time() - start_time) * 1000),
                "timestamp": time.time(),
                "status": "error",
                "error": str(e)
            }
    
    def _search_duckduckgo(self, query: str, max_results: int, **kwargs) -> List[SearchResult]:
        """Search using DuckDuckGo Instant Answer API."""
        results = []
        
        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
                "no_redirect": "1"
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            # Add instant answer if available
            if data.get("Answer"):
                results.append(SearchResult(
                    title="Instant Answer",
                    url=data.get("AnswerURL", ""),
                    snippet=data.get("Answer", ""),
                    source="DuckDuckGo Instant",
                    metadata={"type": "instant_answer"}
                ))
            
            # Add abstract if available
            if data.get("Abstract"):
                results.append(SearchResult(
                    title=data.get("AbstractText", "Abstract")[:100] + "...",
                    url=data.get("AbstractURL", ""),
                    snippet=data.get("Abstract", ""),
                    source="DuckDuckGo Abstract",
                    metadata={"type": "abstract"}
                ))
            
            # Add related topics
            for topic in data.get("RelatedTopics", []):
                if len(results) >= max_results:
                    break
                
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(SearchResult(
                        title=topic.get("Text", "")[:80] + "...",
                        url=topic.get("FirstURL", ""),
                        snippet=topic.get("Text", ""),
                        source="DuckDuckGo Related",
                        metadata={"type": "related_topic"}
                    ))
            
            # Add definition if available
            if data.get("Definition"):
                results.append(SearchResult(
                    title="Definition",
                    url=data.get("DefinitionURL", ""),
                    snippet=data.get("Definition", ""),
                    source="DuckDuckGo Definition",
                    metadata={"type": "definition"}
                ))
            
            # If no structured results, add search link
            if not results:
                results.append(SearchResult(
                    title=f"Search for '{query}' on DuckDuckGo",
                    url=f"https://duckduckgo.com/?q={quote_plus(query)}",
                    snippet=f"No instant answers found. Click to search '{query}' on DuckDuckGo.",
                    source="DuckDuckGo Search",
                    metadata={"type": "search_link"}
                ))
            
            return results[:max_results]
            
        except Exception as e:
            # Return fallback result
            return [SearchResult(
                title="Search Error",
                url=f"https://duckduckgo.com/?q={quote_plus(query)}",
                snippet=f"Error searching DuckDuckGo: {str(e)}. Click to search manually.",
                source="DuckDuckGo Error",
                metadata={"type": "error", "error": str(e)}
            )]
    
    def _search_searx(self, query: str, max_results: int, **kwargs) -> List[SearchResult]:
        """Search using SearX public instance."""
        results = []
        
        # List of public SearX instances to try
        searx_instances = [
            "https://searx.be",
            "https://searx.xyz",
            "https://search.sapti.me",
            "https://searx.fmac.xyz"
        ]
        
        for instance in searx_instances:
            try:
                url = f"{instance}/search"
                params = {
                    "q": query,
                    "format": "json",
                    "categories": "general",
                    "pageno": 1
                }
                
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                
                for result_data in data.get("results", [])[:max_results]:
                    results.append(SearchResult(
                        title=result_data.get("title", ""),
                        url=result_data.get("url", ""),
                        snippet=result_data.get("content", ""),
                        source=f"SearX ({instance})",
                        metadata={
                            "type": "web_result",
                            "engine": result_data.get("engine", ""),
                            "instance": instance
                        }
                    ))
                
                # If we got results, break out of the loop
                if results:
                    break
                    
            except Exception as e:
                continue  # Try next instance
        
        # If no results from any instance
        if not results:
            results.append(SearchResult(
                title="SearX Search Unavailable",
                url=f"https://searx.be/search?q={quote_plus(query)}",
                snippet=f"All SearX instances are currently unavailable. Click to search manually.",
                source="SearX Error",
                metadata={"type": "error", "error": "All instances unavailable"}
            ))
        
        return results[:max_results]
    
    def _search_brave(self, query: str, max_results: int, **kwargs) -> List[SearchResult]:
        """Search using Brave Search (scraping approach)."""
        # Note: This is a demonstration. In production, use Brave Search API
        results = []
        
        try:
            url = "https://search.brave.com/search"
            params = {"q": query}
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML results (simplified parsing)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find search result containers (this is simplified and may need updates)
            result_containers = soup.find_all('div', {'data-type': 'web'})[:max_results]
            
            for container in result_containers:
                title_elem = container.find('h3') or container.find('a')
                url_elem = container.find('a')
                snippet_elem = container.find('p') or container.find('div', class_='snippet')
                
                if title_elem and url_elem:
                    results.append(SearchResult(
                        title=title_elem.get_text(strip=True),
                        url=url_elem.get('href', ''),
                        snippet=snippet_elem.get_text(strip=True) if snippet_elem else "",
                        source="Brave Search",
                        metadata={"type": "web_result", "method": "scraping"}
                    ))
            
            # Fallback if no results parsed
            if not results:
                results.append(SearchResult(
                    title=f"Search '{query}' on Brave",
                    url=f"https://search.brave.com/search?q={quote_plus(query)}",
                    snippet="Brave Search results. Click to view on Brave Search.",
                    source="Brave Search",
                    metadata={"type": "search_link"}
                ))
                
        except Exception as e:
            results.append(SearchResult(
                title="Brave Search Error",
                url=f"https://search.brave.com/search?q={quote_plus(query)}",
                snippet=f"Error accessing Brave Search: {str(e)}",
                source="Brave Search Error",
                metadata={"type": "error", "error": str(e)}
            ))
        
        return results[:max_results]
    
    def _search_startpage(self, query: str, max_results: int, **kwargs) -> List[SearchResult]:
        """Search using Startpage (privacy-focused search)."""
        # Placeholder implementation
        return [SearchResult(
            title=f"Search '{query}' on Startpage",
            url=f"https://www.startpage.com/do/dsearch?query={quote_plus(query)}",
            snippet="Startpage provides private Google search results. Click to search.",
            source="Startpage",
            metadata={"type": "search_link", "note": "Placeholder implementation"}
        )]
    
    def test_providers(self, test_query: str = "python programming") -> Dict[str, Any]:
        """Test all available providers with a simple query."""
        results = {}
        
        for provider in self.providers.keys():
            try:
                start_time = time.time()
                search_result = self.search(test_query, provider, max_results=3)
                search_result["test_duration"] = round((time.time() - start_time) * 1000)
                results[provider] = search_result
            except Exception as e:
                results[provider] = {
                    "status": "error",
                    "error": str(e),
                    "provider": provider
                }
        
        return results
