"""
Query validation utilities.
"""

import re
from typing import List, Optional
from core.exceptions import ValidationError, ErrorCodes


class QueryValidator:
    """Validator for user queries and inputs."""
    
    # Common potentially harmful patterns
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                # JavaScript URLs
        r'on\w+\s*=',                 # Event handlers
        r'eval\s*\(',                 # eval() calls
        r'exec\s*\(',                 # exec() calls
    ]
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)',
        r'(\'|\"|;|--|\|)',
        r'(\bUNION\b|\bOR\b|\bAND\b).*(\b1=1\b|\b1=0\b)',
    ]
    
    @classmethod
    def validate_user_query(cls, query: str) -> None:
        """
        Validate user query for safety and basic requirements.
        
        Args:
            query: User input query
            
        Raises:
            ValidationError: If query is invalid or potentially dangerous
        """
        if not query:
            raise ValidationError(
                "Query cannot be empty",
                field="query",
                value=query,
                error_code=ErrorCodes.INVALID_QUERY
            )
        
        query_stripped = query.strip()
        
        if not query_stripped:
            raise ValidationError(
                "Query cannot be empty or only whitespace",
                field="query",
                value=query,
                error_code=ErrorCodes.INVALID_QUERY
            )
        
        if len(query_stripped) < 2:
            raise ValidationError(
                "Query must be at least 2 characters long",
                field="query",
                value=query,
                error_code=ErrorCodes.INVALID_QUERY
            )
        
        if len(query) > 10000:  # 10KB limit
            raise ValidationError(
                "Query is too long (maximum 10,000 characters)",
                field="query",
                value=f"{query[:50]}...",
                error_code=ErrorCodes.INVALID_QUERY
            )
        
        # Check for dangerous patterns
        cls._check_dangerous_patterns(query)
    
    @classmethod
    def validate_search_query(cls, query: str, max_results: int = 50) -> None:
        """
        Validate search query parameters.
        
        Args:
            query: Search query string
            max_results: Maximum number of results requested
            
        Raises:
            ValidationError: If parameters are invalid
        """
        cls.validate_user_query(query)
        
        if len(query) > 1000:  # Search queries should be shorter
            raise ValidationError(
                "Search query is too long (maximum 1,000 characters)",
                field="query",
                value=f"{query[:50]}...",
                error_code=ErrorCodes.INVALID_QUERY
            )
        
        if max_results < 1 or max_results > 50:
            raise ValidationError(
                "Max results must be between 1 and 50",
                field="max_results",
                value=max_results,
                error_code=ErrorCodes.INVALID_PARAMETERS
            )
    
    @classmethod
    def validate_model_parameters(
        cls,
        temperature: float,
        max_tokens: int,
        top_p: float
    ) -> None:
        """
        Validate LLM model parameters.
        
        Args:
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            
        Raises:
            ValidationError: If any parameter is invalid
        """
        if not 0.0 <= temperature <= 2.0:
            raise ValidationError(
                f"Temperature must be between 0.0 and 2.0, got {temperature}",
                field="temperature",
                value=temperature,
                error_code=ErrorCodes.INVALID_PARAMETERS
            )
        
        if not 1 <= max_tokens <= 10000:
            raise ValidationError(
                f"Max tokens must be between 1 and 10,000, got {max_tokens}",
                field="max_tokens",
                value=max_tokens,
                error_code=ErrorCodes.INVALID_PARAMETERS
            )
        
        if not 0.0 <= top_p <= 1.0:
            raise ValidationError(
                f"Top-p must be between 0.0 and 1.0, got {top_p}",
                field="top_p",
                value=top_p,
                error_code=ErrorCodes.INVALID_PARAMETERS
            )
    
    @classmethod
    def _check_dangerous_patterns(cls, text: str) -> None:
        """
        Check for potentially dangerous patterns in input text.
        
        Args:
            text: Text to check
            
        Raises:
            ValidationError: If dangerous patterns are found
        """
        text_lower = text.lower()
        
        # Check for script injection
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
                raise ValidationError(
                    "Input contains potentially dangerous content",
                    field="query",
                    value="[REDACTED]",
                    error_code=ErrorCodes.INVALID_QUERY,
                    details={"reason": "dangerous_pattern_detected"}
                )
        
        # Check for SQL injection
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                raise ValidationError(
                    "Input contains potentially dangerous SQL patterns",
                    field="query",
                    value="[REDACTED]",
                    error_code=ErrorCodes.INVALID_QUERY,
                    details={"reason": "sql_injection_pattern_detected"}
                )
    
    @classmethod
    def sanitize_query(cls, query: str) -> str:
        """
        Sanitize query by removing potentially dangerous content.
        
        Args:
            query: Input query
            
        Returns:
            str: Sanitized query
        """
        if not query:
            return ""
        
        # Remove HTML tags
        sanitized = re.sub(r'<[^>]+>', '', query)
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        return sanitized
