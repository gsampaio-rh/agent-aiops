"""
Configuration settings for the Ollama Chatbot application.
"""

import os
from typing import Dict, Any
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings, Field

# Default model configuration
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TOP_P = 0.9

# Ollama API configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# UI Configuration
APP_TITLE = "Ollama Chatbot"
APP_ICON = "🤖"

# Model parameters configuration
MODEL_PARAMS: Dict[str, Any] = {
    "temperature": {
        "min": 0.0,
        "max": 2.0,
        "step": 0.1,
        "default": DEFAULT_TEMPERATURE,
        "help": "Controls randomness in responses. Lower values make output more focused and deterministic."
    },
    "max_tokens": {
        "min": 100,
        "max": 4000,
        "step": 100,
        "default": DEFAULT_MAX_TOKENS,
        "help": "Maximum number of tokens to generate in the response."
    },
    "top_p": {
        "min": 0.1,
        "max": 1.0,
        "step": 0.05,
        "default": DEFAULT_TOP_P,
        "help": "Nucleus sampling parameter. Lower values focus on more probable tokens."
    }
}

# Chat configuration
MAX_CHAT_HISTORY = 100
TYPING_ANIMATION_DELAY = 0.02

# Metrics display configuration
SHOW_DETAILED_METRICS = True  # Set to False for simplified metrics display

# Logging configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "DEBUG"),
    "log_dir": os.getenv("LOG_DIR", "logs"),
    "enable_file_logging": os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true",
    "enable_json_logging": os.getenv("ENABLE_JSON_LOGGING", "true").lower() == "true",
    "max_file_size": int(os.getenv("LOG_MAX_FILE_SIZE", str(10 * 1024 * 1024))),  # 10MB
    "backup_count": int(os.getenv("LOG_BACKUP_COUNT", "5"))
}


class AppConfig(BaseSettings):
    """Application configuration with validation."""
    
    # Ollama Configuration
    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    default_model: str = Field("llama3.2:3b", env="DEFAULT_MODEL")
    
    # Model Parameters
    default_temperature: float = Field(0.7, ge=0.0, le=2.0, env="DEFAULT_TEMPERATURE")
    default_max_tokens: int = Field(1000, ge=1, le=10000, env="DEFAULT_MAX_TOKENS")
    default_top_p: float = Field(0.9, ge=0.0, le=1.0, env="DEFAULT_TOP_P")
    
    # UI Configuration
    app_title: str = Field("Ollama Chatbot", env="APP_TITLE")
    app_icon: str = Field("🤖", env="APP_ICON")
    max_chat_history: int = Field(100, ge=1, le=1000, env="MAX_CHAT_HISTORY")
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_dir: str = Field("logs", env="LOG_DIR")
    enable_file_logging: bool = Field(True, env="ENABLE_FILE_LOGGING")
    enable_json_logging: bool = Field(True, env="ENABLE_JSON_LOGGING")
    log_max_file_size: int = Field(10 * 1024 * 1024, env="LOG_MAX_FILE_SIZE")  # 10MB
    log_backup_count: int = Field(5, env="LOG_BACKUP_COUNT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global config instance
config = AppConfig()

# Keep backward compatibility
DEFAULT_MODEL = config.default_model
DEFAULT_TEMPERATURE = config.default_temperature
DEFAULT_MAX_TOKENS = config.default_max_tokens
DEFAULT_TOP_P = config.default_top_p
OLLAMA_BASE_URL = config.ollama_base_url
APP_TITLE = config.app_title
APP_ICON = config.app_icon
MAX_CHAT_HISTORY = config.max_chat_history
SHOW_DETAILED_METRICS = True  # Keep as is for now
