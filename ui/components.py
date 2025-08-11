"""
UI Components for the Streamlit Ollama Chatbot Application.

This module serves as the main entry point for all UI components, providing
backwards compatibility by re-exporting functions from specialized modules:

- sidebar: Model selection and parameters
- chat_display: Chat message rendering and interface
- agent_display: Agent step visualization
- tool_execution: Tool permission and execution workflow
- session_management: Page setup and session state

All components follow Jobs/Ive design principles with minimalist aesthetics.
"""

# Re-export all components for backwards compatibility
from ui.sidebar import render_sidebar
from ui.chat_display import render_chat_message, display_chat_interface
from ui.agent_display import (
    render_agent_step_with_state, 
    render_agent_step,
    render_sophisticated_thinking_indicator,
    render_agent_steps_with_timeline
)
from ui.tool_execution import (
    render_tool_permission_card, 
    render_tool_execution_progress,
    render_tool_execution_success, 
    render_tool_execution_failed,
    handle_tool_execution_workflow
)
from ui.session_management import setup_page_config, initialize_enhanced_session, display_app_header

# Export all functions to maintain API compatibility
__all__ = [
    # Sidebar components
    'render_sidebar',
    
    # Chat display components
    'render_chat_message',
    'display_chat_interface',
    
    # Agent display components
    'render_agent_step_with_state',
    'render_agent_step',
    'render_sophisticated_thinking_indicator',
    'render_agent_steps_with_timeline',
    
    # Tool execution components
    'render_tool_permission_card',
    'render_tool_execution_progress',
    'render_tool_execution_success',
    'render_tool_execution_failed',
    'handle_tool_execution_workflow',
    
    # Session management
    'setup_page_config',
    'initialize_enhanced_session',
    'display_app_header'
]