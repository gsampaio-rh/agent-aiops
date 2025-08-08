"""
UI constants and enums for Agent-AIOps.

This module contains all UI-related constants, enums, and configuration
that doesn't need to be environment-configurable.
"""

from enum import Enum


# UI Configuration
APP_TITLE = "Ollama Chatbot"
APP_ICON = "🤖"

# Chat configuration
MAX_CHAT_HISTORY = 100
TYPING_ANIMATION_DELAY = 0.02

# Metrics display configuration
SHOW_DETAILED_METRICS = True


class ChatMode(str, Enum):
    """Chat operation modes."""
    NORMAL = "normal"
    AGENT = "agent"


class UITheme(str, Enum):
    """UI theme options."""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


# Model parameters UI configuration
MODEL_PARAMS_CONFIG = {
    "temperature": {
        "min": 0.0,
        "max": 2.0,
        "step": 0.1,
        "help": "Controls randomness in responses. Lower values make output more focused and deterministic."
    },
    "max_tokens": {
        "min": 100,
        "max": 4000,
        "step": 100,
        "help": "Maximum number of tokens to generate in the response."
    },
    "top_p": {
        "min": 0.1,
        "max": 1.0,
        "step": 0.05,
        "help": "Nucleus sampling parameter. Lower values focus on more probable tokens."
    }
}


# Agent configuration
AGENT_CONFIG = {
    "max_iterations": 10,
    "default_tools": ["web_search"],
    "step_display_delay": 0.5  # seconds between step displays
}


# Search providers configuration
SEARCH_PROVIDERS = {
    "duckduckgo": {
        "name": "DuckDuckGo",
        "supports_instant": True,
        "timeout": 10
    },
    "searx": {
        "name": "SearX",
        "supports_instant": False,
        "timeout": 10
    },
    "brave": {
        "name": "Brave Search",
        "supports_instant": False,
        "timeout": 10
    },
    "startpage": {
        "name": "Startpage",
        "supports_instant": False,
        "timeout": 10
    }
}


# UI Messages and Labels
UI_MESSAGES = {
    "ollama_not_running": "❌ Ollama service is not running",
    "ollama_start_help": "Please start Ollama: `ollama serve`",
    "ollama_running": "✅ Ollama is running",
    "no_models": "No models available. Please install a model:",
    "agent_not_initialized": "⚠️ Agent not properly initialized",
    "normal_chat_placeholder": "Type your message here...",
    "agent_chat_placeholder": "Ask me anything... I can search the web if needed!",
    "chat_cleared": "🗑️ Chat history cleared",
    "export_success": "📄 Chat history exported successfully"
}


# Sidebar sections
SIDEBAR_SECTIONS = {
    "model": "🤖 Model",
    "mode": "🧠 Mode", 
    "tools": "🛠️ Available Tools",
    "thinking": "💭 Show thinking process",
    "parameters": "⚙️ Parameters",
    "session": "💾 Session",
    "metrics": "📊 Metrics"
}


# Chat mode descriptions
CHAT_MODE_DESCRIPTIONS = {
    ChatMode.NORMAL: "Direct conversation with the AI",
    ChatMode.AGENT: "Watch the AI reason step by step and use tools when needed"
}


# File size limits (in bytes)
FILE_SIZE_LIMITS = {
    "log_file": 10 * 1024 * 1024,  # 10MB
    "export_file": 5 * 1024 * 1024,  # 5MB
    "upload_file": 2 * 1024 * 1024   # 2MB
}
