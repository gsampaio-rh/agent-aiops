"""
Advanced Logging System for Agent-AIOps

Provides enterprise-grade logging with:
- Structured logging with JSON output
- Request correlation tracking
- Performance monitoring
- Different log levels for different environments
- Beautiful terminal formatting
- File rotation and retention
"""

import logging
import logging.handlers
import json
import time
import os
import sys
import uuid
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import functools
import traceback
from contextlib import contextmanager

# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Standard colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record):
        """Format log record as structured JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_data["correlation_id"] = record.correlation_id
            
        # Add performance metrics if available
        if hasattr(record, 'duration_ms'):
            log_data["duration_ms"] = record.duration_ms
            
        # Add custom fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
            
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
            
        return json.dumps(log_data, ensure_ascii=False)


class ColoredTerminalFormatter(logging.Formatter):
    """Beautiful colored formatter for terminal output."""
    
    LEVEL_COLORS = {
        'DEBUG': Colors.DIM + Colors.CYAN,
        'INFO': Colors.BRIGHT_BLUE,
        'WARNING': Colors.BRIGHT_YELLOW,
        'ERROR': Colors.BRIGHT_RED,
        'CRITICAL': Colors.BRIGHT_MAGENTA + Colors.BOLD,
    }
    
    def format(self, record):
        """Format log record with colors and beautiful styling."""
        level_color = self.LEVEL_COLORS.get(record.levelname, Colors.WHITE)
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        
        # Format correlation ID if available
        correlation = ""
        if hasattr(record, 'correlation_id'):
            correlation = f" {Colors.DIM}[{record.correlation_id[:8]}]{Colors.RESET}"
            
        # Format duration if available
        duration = ""
        if hasattr(record, 'duration_ms'):
            duration = f" {Colors.DIM}({record.duration_ms}ms){Colors.RESET}"
            
        # Format module and function
        location = f"{Colors.DIM}{record.module}.{record.funcName}:{record.lineno}{Colors.RESET}"
        
        # Format the main message
        message = record.getMessage()
        
        # Assemble the final format
        formatted = (
            f"{Colors.DIM}{timestamp}{Colors.RESET} "
            f"{level_color}{record.levelname:<8}{Colors.RESET} "
            f"{location} "
            f"{message}"
            f"{correlation}"
            f"{duration}"
        )
        
        # Add exception formatting if present
        if record.exc_info:
            formatted += f"\n{Colors.RED}Exception: {record.exc_info[1]}{Colors.RESET}"
            
        return formatted


class RequestContext:
    """Thread-local context for tracking request correlation."""
    
    def __init__(self):
        self._correlation_id = None
        self._start_time = None
        self._context_data = {}
    
    @property
    def correlation_id(self) -> Optional[str]:
        return self._correlation_id
    
    @correlation_id.setter
    def correlation_id(self, value: str):
        self._correlation_id = value
        
    @property
    def start_time(self) -> Optional[float]:
        return self._start_time
        
    @start_time.setter
    def start_time(self, value: float):
        self._start_time = value
        
    def add_context(self, **kwargs):
        """Add context data to the current request."""
        self._context_data.update(kwargs)
        
    def get_context(self) -> Dict[str, Any]:
        """Get all context data."""
        return self._context_data.copy()
        
    def clear(self):
        """Clear all context data."""
        self._correlation_id = None
        self._start_time = None
        self._context_data = {}


# Global request context
_request_context = RequestContext()


class AgentLogger:
    """Enhanced logger with correlation tracking and performance monitoring."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_done = False
    
    def _ensure_setup(self):
        """Ensure logger is properly configured."""
        if not self._setup_done:
            setup_logging()
            self._setup_done = True
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with current context."""
        self._ensure_setup()
        
        extra = {}
        
        # Add correlation ID if available
        if _request_context.correlation_id:
            extra['correlation_id'] = _request_context.correlation_id
            
        # Add duration if we have a start time
        if _request_context.start_time and 'duration_ms' not in kwargs:
            duration_ms = round((time.time() - _request_context.start_time) * 1000)
            extra['duration_ms'] = duration_ms
            
        # Add context data
        context_data = _request_context.get_context()
        if context_data:
            extra['extra_fields'] = context_data
            
        # Add any additional fields passed in
        if kwargs:
            if 'extra_fields' in extra:
                extra['extra_fields'].update(kwargs)
            else:
                extra['extra_fields'] = kwargs
                
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
        
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log_with_context(logging.INFO, message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, **kwargs)
        
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log_with_context(logging.ERROR, message, **kwargs)
        
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
        
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self._log_with_context(logging.ERROR, message, exc_info=True, **kwargs)


def get_logger(name: str) -> AgentLogger:
    """Get a logger instance with the given name."""
    return AgentLogger(name)


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    enable_file_logging: bool = True,
    enable_json_logging: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """
    Setup comprehensive logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_file_logging: Whether to enable file logging
        enable_json_logging: Whether to enable structured JSON logging
        max_file_size: Maximum size per log file in bytes
        backup_count: Number of backup files to keep
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logs directory
    if enable_file_logging:
        Path(log_dir).mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ColoredTerminalFormatter())
    root_logger.addHandler(console_handler)
    
    if enable_file_logging:
        # Application log file (human readable)
        app_log_path = Path(log_dir) / "agent-aiops.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_path,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        app_handler.setLevel(log_level)
        app_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
        ))
        root_logger.addHandler(app_handler)
        
        if enable_json_logging:
            # Structured JSON log file (for log analysis)
            json_log_path = Path(log_dir) / "agent-aiops.json"
            json_handler = logging.handlers.RotatingFileHandler(
                json_log_path,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            json_handler.setLevel(log_level)
            json_handler.setFormatter(StructuredFormatter())
            root_logger.addHandler(json_handler)


@contextmanager
def request_context(correlation_id: Optional[str] = None, **context_data):
    """
    Context manager for tracking request correlation and timing.
    
    Usage:
        with request_context("user-query-123", user_id="user_123"):
            logger.info("Processing user query")
            # ... processing logic ...
    """
    # Generate correlation ID if not provided
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    # Set context
    old_correlation = _request_context.correlation_id
    old_start_time = _request_context.start_time
    old_context = _request_context.get_context()
    
    _request_context.correlation_id = correlation_id
    _request_context.start_time = time.time()
    _request_context.add_context(**context_data)
    
    try:
        yield correlation_id
    finally:
        # Restore previous context
        _request_context.correlation_id = old_correlation
        _request_context.start_time = old_start_time
        _request_context._context_data = old_context


def log_performance(func_name: Optional[str] = None):
    """
    Decorator for logging function performance.
    
    Usage:
        @log_performance("ollama_chat")
        def chat_with_ollama(...):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            name = func_name or f"{func.__name__}"
            
            start_time = time.time()
            logger.debug(f"Starting {name}", function=name)
            
            try:
                result = func(*args, **kwargs)
                duration_ms = round((time.time() - start_time) * 1000)
                logger.info(f"Completed {name}", function=name, duration_ms=duration_ms)
                return result
            except Exception as e:
                duration_ms = round((time.time() - start_time) * 1000)
                logger.error(f"Failed {name}: {str(e)}", function=name, duration_ms=duration_ms)
                raise
                
        return wrapper
    return decorator


def log_agent_step(step_type: str, content: str, **metadata):
    """Log agent step with structured data."""
    logger = get_logger("agent.steps")
    logger.info(f"Agent step: {step_type}", 
                step_type=step_type, 
                content=content[:100] + "..." if len(content) > 100 else content,
                **metadata)


def log_search_query(provider: str, query: str, results_count: int, duration_ms: int):
    """Log search query with results."""
    logger = get_logger("search")
    logger.info(f"Search completed", 
                provider=provider,
                query=query,
                results_count=results_count,
                duration_ms=duration_ms)


def log_ollama_request(model: str, message_count: int, temperature: float, **params):
    """Log Ollama API request."""
    logger = get_logger("ollama")
    logger.info(f"Ollama request", 
                model=model,
                message_count=message_count,
                temperature=temperature,
                **params)


def log_ollama_response(model: str, tokens_generated: int, duration_ms: int, **metrics):
    """Log Ollama API response with metrics."""
    logger = get_logger("ollama")
    logger.info(f"Ollama response", 
                model=model,
                tokens_generated=tokens_generated,
                duration_ms=duration_ms,
                **metrics)


def log_user_interaction(action: str, mode: str, **details):
    """Log user interactions."""
    logger = get_logger("user")
    logger.info(f"User interaction: {action}",
                action=action,
                mode=mode,
                **details)


# Initialize logging on module import
setup_logging()
