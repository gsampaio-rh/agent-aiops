"""
Streamlit Ollama Chatbot Application

A minimalist, high-end chatbot interface built on Ollama with real-time metrics.
Inspired by Dieter Rams' design principles and Apple's user experience.

This is the main application entry point, orchestrating all UI components
and business logic in a clean, maintainable architecture.
"""

import streamlit as st

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
    # 1. Setup page configuration
    setup_page_config()
    
    # 2. Apply custom CSS styles
    apply_app_styles()
    
    # 3. Initialize enhanced session state
    initialize_enhanced_session()
    
    # 4. Render sidebar and get configuration
    config = render_sidebar()
    
    if not config:
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
        handle_user_input(prompt, config, chat_container)


if __name__ == "__main__":
    main()
