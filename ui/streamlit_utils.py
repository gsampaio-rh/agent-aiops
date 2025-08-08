"""
Streamlit Utilities for the Ollama Chatbot Application.

This module serves as the main entry point for Streamlit utility functions,
providing backwards compatibility by re-exporting functions from specialized modules:

- session_management: Session state initialization and page setup
- chat_processing: Chat processing and response generation
- tool_execution: Tool execution workflow handling

All functions are designed to work seamlessly with Streamlit's reactive architecture.
"""

# Re-export all utility functions for backwards compatibility
from ui.session_management import setup_page_config, initialize_enhanced_session, display_app_header
from ui.chat_processing import (
    finalize_assistant_response_with_timeline,
    process_agent_query,
    process_normal_chat,
    handle_user_input
)
from ui.tool_execution import handle_tool_execution_workflow

# Export all functions to maintain API compatibility
__all__ = [
    # Session management
    'setup_page_config',
    'initialize_enhanced_session', 
    'display_app_header',
    
    # Chat processing
    'finalize_assistant_response_with_timeline',
    'process_agent_query',
    'process_normal_chat', 
    'handle_user_input',
    
    # Tool execution
    'handle_tool_execution_workflow'
]
