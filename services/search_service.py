"""
Enhanced Web Search Service

Provides multiple search providers with error handling and result normalization.
Implements SearchServiceInterface for consistent API.
"""

import requests
import time
from typing import Dict, List, Any, Optional
from urllib.parse import quote_plus, urljoin
import json
import re
from bs4 import BeautifulSoup

from core.interfaces.search_service import SearchServiceInterface
from core.models.search import SearchQuery, SearchResponse, SearchResult
from core.exceptions import SearchError, SearchProviderError, SearchTimeoutError, ErrorCodes
from config.constants import SEARCH_PROVIDERS
from utils.logger import get_logger, log_performance, log_search_query


class WebSearchService(SearchServiceInterface):
    """Enhanced web search service with multiple providers."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        self.timeout = 10  # Shorter timeout to prevent hanging
        self.logger = get_logger(__name__)
        
        # Load provider configuration from constants
        self.providers = {}
        for provider_name, config in SEARCH_PROVIDERS.items():
            self.providers[provider_name] = {
                "name": config["name"],
                "function": getattr(self, f"_search_{provider_name}"),
                "supports_instant": config.get("supports_instant", False),
                "timeout": config.get("timeout", self.timeout)
            }
    
    def get_available_providers(self) -> List[str]:
        """Get list of available search providers."""
        return list(self.providers.keys())
    
    @log_performance("web_search") 
    def search(self, query: SearchQuery) -> SearchResponse:
        """
        Perform web search using specified provider.
        
        Args:
            query: SearchQuery object with search parameters
            
        Returns:
            SearchResponse: Complete search response with results and metadata
            
        Raises:
            SearchProviderError: If the search provider fails
            SearchTimeoutError: If the search times out
            ValidationError: If query parameters are invalid
        """
        # Validate query first
        self.validate_query(query)
        
        start_time = time.time()
        
        self.logger.info("Starting web search", 
                        query=query.query, 
                        provider=query.provider, 
                        max_results=query.max_results)
        
        try:
            if query.provider not in self.providers:
                raise SearchProviderError(
                    f"Unknown provider: {query.provider}",
                    error_code=ErrorCodes.SEARCH_PROVIDER_FAILED,
                    details={
                        "provider": query.provider,
                        "available": list(self.providers.keys())
                    }
                )
            
            provider_info = self.providers[query.provider]
            search_function = provider_info["function"]
            
            # Execute search with timeout
            try:
                results = search_function(query.query, query.max_results, **query.parameters)
            except requests.exceptions.Timeout:
                raise SearchTimeoutError(
                    f"Search request timed out for provider {query.provider}",
                    error_code=ErrorCodes.SEARCH_TIMEOUT,
                    details={"provider": query.provider, "timeout": self.timeout}
                )
            
            search_time_ms = round((time.time() - start_time) * 1000)
            
            # Create response object
            response = SearchResponse(
                query=query.query,
                provider=query.provider,
                provider_name=provider_info["name"],
                search_time_ms=search_time_ms,
                status="success"
            )
            
            # Add results
            for result in results:
                if isinstance(result, dict):
                    # Convert dict to SearchResult if needed
                    result = SearchResult(**result)
                response.add_result(result)
            
            # Log search completion
            log_search_query(query.provider, query.query, response.total_results, search_time_ms)
            self.logger.info("Web search completed", 
                           provider=query.provider, 
                           results_count=response.total_results,
                           search_time_ms=search_time_ms)
            
            return response
            
        except (SearchError, SearchProviderError, SearchTimeoutError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            search_time_ms = round((time.time() - start_time) * 1000)
            self.logger.error("Web search failed", 
                            provider=query.provider, 
                            query=query.query,
                            error=str(e),
                            search_time_ms=search_time_ms)
            
            # Create error response
            response = SearchResponse(
                query=query.query,
                provider=query.provider,
                provider_name=self.providers.get(query.provider, {}).get("name", query.provider),
                search_time_ms=search_time_ms,
                status="error",
                error=str(e)
            )
            
            return response
    
    def test_provider(self, provider: str) -> Dict[str, Any]:
        """
        Test if a specific provider is working.
        
        Args:
            provider: Provider name to test
            
        Returns:
            Dict[str, Any]: Test results with status and performance metrics
        """
        test_query = SearchQuery(
            query="python programming",
            provider=provider,
            max_results=3
        )
        
        start_time = time.time()
        
        try:
            response = self.search(test_query)
            test_duration_ms = round((time.time() - start_time) * 1000)
            
            return {
                "provider": provider,
                "status": "success" if response.is_successful() else "error",
                "results_count": response.total_results,
                "test_duration_ms": test_duration_ms,
                "error": response.error if not response.is_successful() else None
            }
        except Exception as e:
            test_duration_ms = round((time.time() - start_time) * 1000)
            return {
                "provider": provider,
                "status": "error",
                "results_count": 0,
                "test_duration_ms": test_duration_ms,
                "error": str(e)
            }
    
    def _search_duckduckgo(self, query: str, max_results: int, **kwargs) -> List[SearchResult]:
        """Search using DuckDuckGo HTML scraping for real search results."""
        results = []
        
        try:
            # Use DuckDuckGo HTML search (not the limited Instant Answer API)
            url = "https://html.duckduckgo.com/html/"
            params = {"q": query}
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
            }
            
            response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML content with BeautifulSoup
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find search result containers
            result_divs = soup.find_all('div', class_='result')
            
            for i, result_div in enumerate(result_divs[:max_results]):
                try:
                    # Extract title and URL
                    title_link = result_div.find('a', class_='result__a')
                    if not title_link:
                        continue
                        
                    title = title_link.get_text(strip=True)
                    url = title_link.get('href', '')
                    
                    # Extract snippet/description
                    snippet_elem = result_div.find('a', class_='result__snippet')
                    snippet = ""
                    if snippet_elem:
                        snippet = snippet_elem.get_text(strip=True)
                    
                    # If no snippet found, try other selectors
                    if not snippet:
                        snippet_elem = result_div.find('div', class_='result__snippet')
                        if snippet_elem:
                            snippet = snippet_elem.get_text(strip=True)
                    
                    if not snippet:
                        snippet = f"Search result {i+1} for '{query}'"
                    
                    if title and url:
                        results.append(SearchResult(
                            title=title[:150] + "..." if len(title) > 150 else title,
                            url=url,
                            snippet=snippet[:400] + "..." if len(snippet) > 400 else snippet,
                            source="DuckDuckGo",
                            metadata={"type": "web_result", "position": i+1}
                        ))
                        
                except Exception as e:
                    continue  # Skip this result if there's an error parsing it
            
            # If HTML scraping failed, try the Instant Answer API as fallback
            if not results:
                results = self._search_duckduckgo_instant_api(query, max_results)
            
            return results[:max_results]
            
        except Exception as e:
            print(f"DuckDuckGo HTML search error: {e}")
            # Fallback to Instant Answer API
            return self._search_duckduckgo_instant_api(query, max_results)
    
    def _search_duckduckgo_instant_api(self, query: str, max_results: int) -> List[SearchResult]:
        """Fallback using DuckDuckGo Instant Answer API."""
        results = []
        
        try:
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
            
        except Exception as e:
            print(f"DuckDuckGo Instant API error: {e}")
        
        # Final fallback: create a helpful result
        if not results:
            search_url = f"https://duckduckgo.com/?q={quote_plus(query)}"
            results.append(SearchResult(
                title=f"Search '{query}' on DuckDuckGo",
                url=search_url,
                snippet=f"Use this link to search manually: {search_url}",
                source="DuckDuckGo Manual",
                metadata={"type": "manual_search"}
            ))
        
        return results
    
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
    
    # Backward compatibility method for existing code
    def search_legacy(self, query: str, provider: str = "duckduckgo", 
                     max_results: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Legacy search method for backward compatibility.
        
        This method maintains the old API while using the new implementation.
        """
        search_query = SearchQuery(
            query=query,
            provider=provider,
            max_results=max_results,
            parameters=kwargs
        )
        
        try:
            response = self.search(search_query)
            
            # Convert to old format
            return {
                "query": response.query,
                "provider": response.provider,
                "provider_name": response.provider_name,
                "results": [result.to_dict() for result in response.results],
                "total_results": response.total_results,
                "search_time_ms": response.search_time_ms,
                "timestamp": response.timestamp,
                "status": response.status,
                "error": response.error
            }
        except Exception as e:
            return {
                "query": query,
                "provider": provider,
                "provider_name": SEARCH_PROVIDERS.get(provider, {}).get("name", provider),
                "results": [],
                "total_results": 0,
                "search_time_ms": 0,
                "timestamp": time.time(),
                "status": "error",
                "error": str(e)
            }
