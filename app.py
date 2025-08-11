"""
Streamlit Ollama Chatbot Application

A minimalist, high-end chatbot interface built on Ollama with real-time metrics.
Inspired by Dieter Rams' design principles and Apple's user experience.

This is the main application entry point, orchestrating all UI components
and business logic in a clean, maintainable architecture.
"""

import streamlit as st

# Setup logging before importing other modules
from config.settings import LOGGING_CONFIG
from utils.logger import setup_logging, get_logger, log_user_interaction
from utils.system_metrics import start_system_monitoring, log_current_metrics

# Initialize logging with configuration
setup_logging(**LOGGING_CONFIG)

# Import our modular components
from ui.styles import apply_app_styles
from ui.components import render_sidebar, display_chat_interface
from ui.streamlit_utils import (
    setup_page_config, 
    initialize_enhanced_session,
    display_app_header,
    handle_user_input
)


def main():
    """
    Main application orchestration.
    
    This function coordinates all the major components of the application:
    1. Page setup and styling
    2. Session state initialization  
    3. Sidebar configuration
    4. Chat interface display
    5. User input handling
    """
    logger = get_logger(__name__)
    logger.info("Starting Agent-AIOps application")
    
    # 0. Start system monitoring (non-blocking)
    try:
        start_system_monitoring(interval=60.0)  # Collect metrics every minute
        log_current_metrics()  # Log initial metrics
        logger.info("System monitoring started")
    except Exception as e:
        logger.warning("Failed to start system monitoring", error=str(e))
    
    # 1. Setup page configuration
    setup_page_config()
    
    # 2. Apply custom CSS styles
    apply_app_styles()
    
    # 3. Initialize enhanced session state
    initialize_enhanced_session()
    
    # 4. Render sidebar and get configuration
    config = render_sidebar()
    
    if not config:
        logger.error("Configuration failed - Ollama not available or no models")
        st.error("Please ensure Ollama is running and models are available.")
        st.stop()
    
    # 5. Display main app header
    display_app_header(config)
    
    # 6. Create chat container and display interface
    chat_container = st.container()
    display_chat_interface(config, chat_container)
    
    # 7. Handle user input
    placeholder = ("Ask me anything... I can search the web if needed!" 
                  if config["agent_mode"] 
                  else "Type your message here...")
    
    if prompt := st.chat_input(placeholder, key="chat_input"):
        log_user_interaction("chat_input", config.get("agent_mode", "normal"), prompt_length=len(prompt))
        logger.info("User input received", mode=config.get("agent_mode", "normal"), prompt_length=len(prompt))
        handle_user_input(prompt, config, chat_container)


if __name__ == "__main__":
    main()
