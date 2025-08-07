"""
Configuration settings for the Ollama Chatbot application.
"""

import os
from typing import Dict, Any

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
