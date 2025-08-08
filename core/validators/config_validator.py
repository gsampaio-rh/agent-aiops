"""
Configuration validation utilities.
"""

import os
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
from core.exceptions import ValidationError, ErrorCodes


class ConfigValidator:
    """Validator for application configuration."""
    
    REQUIRED_ENV_VARS = [
        # Add required environment variables here
    ]
    
    VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    @classmethod
    def validate_ollama_config(cls, base_url: str, timeout: int = 30) -> None:
        """
        Validate Ollama service configuration.
        
        Args:
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
            
        Raises:
            ValidationError: If configuration is invalid
        """
        if not base_url:
            raise ValidationError(
                "Ollama base URL cannot be empty",
                field="ollama_base_url",
                error_code=ErrorCodes.INVALID_CONFIG
            )
        
        # Validate URL format
        try:
            parsed = urlparse(base_url)
            if not parsed.scheme or not parsed.netloc:
                raise ValidationError(
                    f"Invalid Ollama base URL format: {base_url}",
                    field="ollama_base_url",
                    value=base_url,
                    error_code=ErrorCodes.INVALID_CONFIG
                )
            
            if parsed.scheme not in ['http', 'https']:
                raise ValidationError(
                    f"Ollama base URL must use http or https scheme, got: {parsed.scheme}",
                    field="ollama_base_url",
                    value=base_url,
                    error_code=ErrorCodes.INVALID_CONFIG
                )
        except Exception as e:
            raise ValidationError(
                f"Invalid Ollama base URL: {str(e)}",
                field="ollama_base_url",
                value=base_url,
                error_code=ErrorCodes.INVALID_CONFIG,
                original_exception=e
            )
        
        # Validate timeout
        if not isinstance(timeout, int) or timeout <= 0:
            raise ValidationError(
                f"Timeout must be a positive integer, got: {timeout}",
                field="timeout",
                value=timeout,
                error_code=ErrorCodes.INVALID_CONFIG
            )
        
        if timeout > 300:  # 5 minutes max
            raise ValidationError(
                f"Timeout too large (max 300 seconds), got: {timeout}",
                field="timeout",
                value=timeout,
                error_code=ErrorCodes.INVALID_CONFIG
            )
    
    @classmethod
    def validate_logging_config(cls, config: Dict[str, Any]) -> None:
        """
        Validate logging configuration.
        
        Args:
            config: Logging configuration dictionary
            
        Raises:
            ValidationError: If configuration is invalid
        """
        required_keys = ["level", "log_dir"]
        for key in required_keys:
            if key not in config:
                raise ValidationError(
                    f"Missing required logging config key: {key}",
                    field=key,
                    error_code=ErrorCodes.INVALID_CONFIG
                )
        
        # Validate log level
        log_level = config.get("level", "INFO").upper()
        if log_level not in cls.VALID_LOG_LEVELS:
            raise ValidationError(
                f"Invalid log level: {log_level}. Must be one of {cls.VALID_LOG_LEVELS}",
                field="log_level",
                value=log_level,
                error_code=ErrorCodes.INVALID_CONFIG
            )
        
        # Validate log directory
        log_dir = config.get("log_dir")
        if not log_dir:
            raise ValidationError(
                "Log directory cannot be empty",
                field="log_dir",
                error_code=ErrorCodes.INVALID_CONFIG
            )
        
        # Check if log directory is writable (create if doesn't exist)
        try:
            os.makedirs(log_dir, exist_ok=True)
            if not os.access(log_dir, os.W_OK):
                raise ValidationError(
                    f"Log directory is not writable: {log_dir}",
                    field="log_dir",
                    value=log_dir,
                    error_code=ErrorCodes.INVALID_CONFIG
                )
        except PermissionError as e:
            raise ValidationError(
                f"Cannot create or access log directory: {log_dir}",
                field="log_dir",
                value=log_dir,
                error_code=ErrorCodes.INVALID_CONFIG,
                original_exception=e
            )
        
        # Validate file size limits
        max_file_size = config.get("max_file_size", 10 * 1024 * 1024)
        if not isinstance(max_file_size, int) or max_file_size <= 0:
            raise ValidationError(
                f"Max file size must be a positive integer, got: {max_file_size}",
                field="max_file_size",
                value=max_file_size,
                error_code=ErrorCodes.INVALID_CONFIG
            )
        
        backup_count = config.get("backup_count", 5)
        if not isinstance(backup_count, int) or backup_count < 0:
            raise ValidationError(
                f"Backup count must be a non-negative integer, got: {backup_count}",
                field="backup_count",
                value=backup_count,
                error_code=ErrorCodes.INVALID_CONFIG
            )
    
    @classmethod
    def validate_environment_variables(cls, required_vars: Optional[List[str]] = None) -> None:
        """
        Validate required environment variables.
        
        Args:
            required_vars: List of required environment variable names
            
        Raises:
            ValidationError: If required variables are missing
        """
        required_vars = required_vars or cls.REQUIRED_ENV_VARS
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValidationError(
                f"Missing required environment variables: {', '.join(missing_vars)}",
                field="environment_variables",
                value=missing_vars,
                error_code=ErrorCodes.INVALID_CONFIG
            )
    
    @classmethod
    def validate_model_name(cls, model_name: str, available_models: List[str]) -> None:
        """
        Validate model name against available models.
        
        Args:
            model_name: Model name to validate
            available_models: List of available model names
            
        Raises:
            ValidationError: If model is not available
        """
        if not model_name:
            raise ValidationError(
                "Model name cannot be empty",
                field="model",
                error_code=ErrorCodes.INVALID_MODEL
            )
        
        if not available_models:
            raise ValidationError(
                "No models available",
                field="available_models",
                error_code=ErrorCodes.INVALID_CONFIG
            )
        
        if model_name not in available_models:
            raise ValidationError(
                f"Model '{model_name}' not found. Available models: {', '.join(available_models)}",
                field="model",
                value=model_name,
                error_code=ErrorCodes.INVALID_MODEL,
                details={"available_models": available_models}
            )
