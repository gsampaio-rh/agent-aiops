"""
Session Management for the Streamlit Ollama Chatbot Application.

This module contains functions for managing Streamlit session state and application setup:
- Page configuration and styling
- Session state initialization
- App header display

All functions are designed to work seamlessly with Streamlit's reactive architecture.
"""

import streamlit as st
from typing import Dict, Any

from config.settings import APP_TITLE, APP_ICON, DEFAULT_MODEL
from services.agent_service import ReactAgent
from utils.chat_utils import initialize_session_state


def setup_page_config():
    """
    Configure the Streamlit page with title, icon, and layout settings.
    
    This should be called once at the beginning of the app.
    """
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )


def initialize_enhanced_session():
    """
    Initialize session state with agent functionality.
    
    Extends the basic session state initialization with agent-specific variables
    and ensures proper initialization order.
    """
    # Initialize basic session state first
    initialize_session_state()
    
    # Agent-specific session state
    if "agent_steps" not in st.session_state:
        st.session_state.agent_steps = []
    
    if "agent_mode" not in st.session_state:
        st.session_state.agent_mode = False
    
    if "show_agent_steps" not in st.session_state:
        st.session_state.show_agent_steps = True
    
    if "agent_processing" not in st.session_state:
        st.session_state.agent_processing = False
    
    # Tool execution state management
    if "current_request" not in st.session_state:
        st.session_state.current_request = None
    
    if "tool_context" not in st.session_state:
        st.session_state.tool_context = None
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    if "pending_finalization_query" not in st.session_state:
        st.session_state.pending_finalization_query = None
    
    # Initialize agent after we have the current model
    if "agent" not in st.session_state:
        try:
            model = st.session_state.get("current_model", DEFAULT_MODEL)
            st.session_state.agent = ReactAgent(model)
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            # Create a fallback agent
            st.session_state.agent = None


def display_app_header(config: Dict[str, Any]):
    """
    Display the main application header based on current mode.
    
    Args:
        config: Configuration dictionary containing agent_mode and other settings
    """
    if config["agent_mode"]:
        st.title("🤖 Agent Chat")
        st.markdown("*Watch the AI reason step by step and use tools when needed*")
    else:
        st.title("💬 Chat")
        st.markdown("*Direct conversation with the AI*")
