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
from typing import Dict, Any, Optional, Union, List
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
        
        # Create specialized log files for different components
        _setup_specialized_loggers(log_dir, log_level, max_file_size, backup_count, enable_json_logging)


def _setup_specialized_loggers(log_dir: str, log_level: int, max_file_size: int, 
                              backup_count: int, enable_json_logging: bool):
    """Setup specialized loggers for different components."""
    specialized_loggers = {
        "ollama.request": "llm-requests.log",
        "ollama.response": "llm-responses.log", 
        "llm.conversation": "llm-conversations.log",
        "agent.workflow": "agent-workflow.log",
        "tools.execution": "tool-execution.log",
        "request.tracer": "request-tracing.log",
        "request.steps": "request-steps.log",
        "request.complete": "request-completion.log",
        "system.metrics": "system-metrics.log",
        "errors": "errors.log",
        "user": "user-interactions.log"
    }
    
    for logger_name, filename in specialized_loggers.items():
        logger = logging.getLogger(logger_name)
        
        # Skip if logger already has handlers to prevent duplicates
        if logger.handlers:
            continue
            
        # Don't propagate to root logger to avoid duplication
        logger.propagate = False
        logger.setLevel(log_level)
        
        # Text file handler
        log_path = Path(log_dir) / filename
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        if enable_json_logging:
            # JSON file handler for structured data
            json_path = Path(log_dir) / f"{filename}.json"
            json_handler = logging.handlers.RotatingFileHandler(
                json_path,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            json_handler.setLevel(log_level)
            json_handler.setFormatter(StructuredFormatter())
            logger.addHandler(json_handler)


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


def log_ollama_request(model: str, message_count: int, temperature: float, messages: Optional[List[Dict[str, str]]] = None, **params):
    """Log Ollama API request with detailed message tracking."""
    logger = get_logger("ollama.request")
    
    # Calculate total input tokens (approximate)
    total_chars = 0
    message_details = []
    
    if messages:
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            char_count = len(content)
            total_chars += char_count
            
            message_details.append({
                "index": i,
                "role": role,
                "content_length": char_count,
                "content": content
            })
    
    # Approximate token count (rough estimate: 1 token ≈ 4 characters)
    estimated_input_tokens = total_chars // 4
    
    logger.info(f"Ollama API Request Initiated", 
                model=model,
                message_count=message_count,
                temperature=temperature,
                estimated_input_tokens=estimated_input_tokens,
                total_input_chars=total_chars,
                message_details=message_details,
                **params)


def log_ollama_response(model: str, tokens_generated: int, duration_ms: int, 
                       full_response: Optional[str] = None, **metrics):
    """Log Ollama API response with comprehensive metrics and content."""
    logger = get_logger("ollama.response")
    
    # Calculate response statistics
    response_stats = {}
    if full_response:
        response_stats = {
            "response_length": len(full_response),
            "response_words": len(full_response.split()),
            "response_lines": full_response.count('\n') + 1,
            "response_preview": full_response[:200] + "..." if len(full_response) > 200 else full_response
        }
    
    # Calculate performance metrics
    performance_metrics = {
        "tokens_per_second": round(tokens_generated / (duration_ms / 1000), 2) if duration_ms > 0 else 0,
        "chars_per_second": round(response_stats.get("response_length", 0) / (duration_ms / 1000), 2) if duration_ms > 0 else 0,
        "duration_seconds": round(duration_ms / 1000, 3)
    }
    
    logger.info(f"Ollama API Response Completed", 
                model=model,
                tokens_generated=tokens_generated,
                duration_ms=duration_ms,
                **response_stats,
                **performance_metrics,
                **metrics)


def log_user_interaction(action: str, mode: str, **details):
    """Log user interactions."""
    logger = get_logger("user")
    logger.info(f"User interaction: {action}",
                action=action,
                mode=mode,
                **details)


def log_llm_conversation(conversation_id: str, messages: List[Dict[str, str]], 
                        model: str, action: str = "conversation", **metadata):
    """Log complete LLM conversation with full context."""
    logger = get_logger("llm.conversation")
    
    conversation_stats = {
        "conversation_id": conversation_id,
        "model": model,
        "action": action,
        "message_count": len(messages),
        "total_chars": sum(len(msg.get('content', '')) for msg in messages),
        "roles": [msg.get('role') for msg in messages],
        "conversation_flow": " -> ".join([msg.get('role', 'unknown') for msg in messages])
    }
    
    # Log each message with context
    message_logs = []
    for i, msg in enumerate(messages):
        message_logs.append({
            "index": i,
            "role": msg.get('role'),
            "content_length": len(msg.get('content', '')),
            "content": msg.get('content', '')[:500] + "..." if len(msg.get('content', '')) > 500 else msg.get('content', ''),
            "timestamp": metadata.get('timestamp', time.time())
        })
    
    logger.info(f"LLM Conversation: {action}",
                **conversation_stats,
                messages=message_logs,
                **metadata)


def log_agent_workflow(workflow_id: str, step_name: str, step_type: str, 
                      step_data: Dict[str, Any], duration_ms: Optional[int] = None):
    """Log agent workflow steps with detailed state tracking."""
    logger = get_logger("agent.workflow")
    
    workflow_info = {
        "workflow_id": workflow_id,
        "step_name": step_name,
        "step_type": step_type,
        "duration_ms": duration_ms,
        "step_data_keys": list(step_data.keys()) if step_data else [],
        "step_data_size": len(str(step_data)) if step_data else 0
    }
    
    # Add specific step data based on type
    if step_type == "reasoning":
        workflow_info.update({
            "reasoning_context": step_data.get("context", "")[:200] + "..." if len(step_data.get("context", "")) > 200 else step_data.get("context", ""),
            "reasoning_output": step_data.get("output", "")[:200] + "..." if len(step_data.get("output", "")) > 200 else step_data.get("output", "")
        })
    elif step_type == "tool_use":
        workflow_info.update({
            "tool_name": step_data.get("tool_name"),
            "tool_query": step_data.get("query", "")[:100] + "..." if len(step_data.get("query", "")) > 100 else step_data.get("query", ""),
            "tool_success": step_data.get("success", False)
        })
    
    logger.info(f"Agent Workflow Step: {step_name}",
                **workflow_info)


def log_tool_execution(tool_name: str, action: str, query: str, result: Dict[str, Any], 
                      duration_ms: int, success: bool = True, **metadata):
    """Log detailed tool execution with input/output tracking."""
    logger = get_logger("tools.execution")
    
    execution_info = {
        "tool_name": tool_name,
        "action": action,
        "query_length": len(query),
        "query_preview": query[:100] + "..." if len(query) > 100 else query,
        "duration_ms": duration_ms,
        "success": success,
        "result_keys": list(result.keys()) if result else [],
        "result_size": len(str(result)) if result else 0
    }
    
    # Add result details
    if result:
        if "results" in result:
            execution_info["result_count"] = len(result["results"]) if isinstance(result["results"], list) else 1
        if "output" in result:
            output = str(result["output"])
            execution_info.update({
                "output_length": len(output),
                "output_preview": output[:200] + "..." if len(output) > 200 else output
            })
        if "error" in result:
            execution_info["error_message"] = str(result["error"])[:200]
    
    logger.info(f"Tool Execution: {tool_name}",
                **execution_info,
                **metadata)


def log_system_metrics(component: str, metrics: Dict[str, Any], **context):
    """Log system performance and health metrics."""
    logger = get_logger("system.metrics")
    
    metric_info = {
        "component": component,
        "timestamp": time.time(),
        "metric_count": len(metrics),
        "metrics": metrics
    }
    
    logger.info(f"System Metrics: {component}",
                **metric_info,
                **context)


def log_error_with_context(error: Exception, context: Dict[str, Any], 
                          operation: str, **metadata):
    """Log errors with comprehensive context and stack trace."""
    logger = get_logger("errors")
    
    error_info = {
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stack_trace": traceback.format_exc(),
        "context": context,
        "timestamp": time.time()
    }
    
    logger.error(f"Error in {operation}: {type(error).__name__}",
                **error_info,
                **metadata)


def create_request_tracer(user_query: str, model: str = None) -> str:
    """Create a unique request tracer for end-to-end tracking."""
    correlation_id = str(uuid.uuid4())[:8]
    logger = get_logger("request.tracer")
    
    logger.info(f"Request Started: {correlation_id}",
                correlation_id=correlation_id,
                user_query=user_query[:100] + "..." if len(user_query) > 100 else user_query,
                model=model,
                timestamp=time.time())
    
    return correlation_id


def log_request_step(correlation_id: str, step: str, data: Dict[str, Any], 
                    duration_ms: Optional[int] = None):
    """Log individual steps within a request for end-to-end tracing."""
    logger = get_logger("request.steps")
    
    step_info = {
        "correlation_id": correlation_id,
        "step": step,
        "duration_ms": duration_ms,
        "data_keys": list(data.keys()) if data else [],
        "timestamp": time.time()
    }
    
    logger.info(f"Request Step: {step}",
                **step_info,
                **data)


def log_request_complete(correlation_id: str, total_duration_ms: int, 
                        success: bool = True, **summary):
    """Log request completion with summary metrics."""
    logger = get_logger("request.complete")
    
    completion_info = {
        "correlation_id": correlation_id,
        "total_duration_ms": total_duration_ms,
        "success": success,
        "timestamp": time.time()
    }
    
    logger.info(f"Request Completed: {correlation_id}",
                **completion_info,
                **summary)


# Initialize logging on module import
setup_logging()
