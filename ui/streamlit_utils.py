"""
Streamlit Utilities for the Ollama Chatbot Application.

This module contains Streamlit-specific utility functions including:
- Session state initialization and management
- App configuration and setup
- Agent processing logic
- Chat flow orchestration

All functions are designed to work seamlessly with Streamlit's reactive architecture.
"""

import streamlit as st
import time
from typing import Dict, Any

from config.settings import APP_TITLE, APP_ICON, DEFAULT_MODEL
from services.agent_service import ReactAgent, AgentStep, StepType
from utils.chat_utils import initialize_session_state, add_message
from ui.components import render_agent_step_with_state, render_chat_message


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


def process_agent_query(prompt: str, config: Dict[str, Any], chat_container):
    """
    Process a user query in agent mode with real-time step visualization.
    
    Args:
        prompt: User's input prompt
        config: Configuration dictionary from sidebar
        chat_container: Streamlit container for displaying chat
    """
    # Check if agent is properly initialized
    if not st.session_state.agent:
        st.error("Agent not initialized. Please check the configuration.")
        return
    
    # Agent thinking with real-time streaming
    agent_steps = []
    thinking_placeholder = st.empty()
    
    # Set processing flag to avoid duplicate display
    st.session_state.agent_processing = True
    
    try:
        # Show thinking indicator
        with thinking_placeholder.container():
            st.markdown("""
            <div class="thinking-indicator">
                <span>Agent is thinking</span>
                <span class="thinking-dots"></span>
            </div>
            """, unsafe_allow_html=True)
        
        # Stream agent steps in real-time
        for step in st.session_state.agent.process_query_stream(
            prompt,
            **config["params"]
        ):
            agent_steps.append(step)
            
            # Clear thinking indicator and show all steps
            with thinking_placeholder.container():
                st.markdown('<div class="agent-thinking">', unsafe_allow_html=True)
                for i, s in enumerate(agent_steps):
                    # Mark the current step and completed steps
                    if i == len(agent_steps) - 1:
                        # This is the current step
                        render_agent_step_with_state(s, "current")
                    else:
                        # This is a completed step
                        render_agent_step_with_state(s, "completed")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Small delay for better visual effect
            time.sleep(0.1)
        
        # Store all steps at once ONLY after completion
        st.session_state.agent_steps.extend(agent_steps)
        
        # Extract final answer for chat history
        final_answer = None
        final_metadata = {}
        for step in reversed(agent_steps):
            if step.step_type == StepType.FINAL_ANSWER:
                final_answer = step.content
                if step.metadata:
                    final_metadata = step.metadata
                break
        
        if final_answer:
            # Add agent response to chat history with enhanced metadata
            total_time = final_metadata.get("total_time_ms", 0)
            add_message("assistant", final_answer, {
                "agent_steps": len(agent_steps),
                "total_time_ms": total_time,
                "agent_mode": True
            })
            
            # Clear the thinking placeholder
            thinking_placeholder.empty()
            
            # Clear processing flag  
            st.session_state.agent_processing = False
            
            # Show the final message immediately without rerun
            with chat_container:
                render_chat_message(st.session_state.messages[-1])
        
    except Exception as e:
        st.error(f"Agent processing failed: {e}")
        # Clear processing flag on error
        st.session_state.agent_processing = False
        # Fall back to normal chat mode
        st.session_state.agent_mode = False


def process_normal_chat(prompt: str, config: Dict[str, Any], chat_container):
    """
    Process a user query in normal chat mode with streaming response.
    
    Args:
        prompt: User's input prompt
        config: Configuration dictionary from sidebar
        chat_container: Streamlit container for displaying chat
    """
    with st.spinner("Thinking..."):
        # Prepare messages for API
        api_messages = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in st.session_state.messages
        ]
        
        # Stream response
        response_placeholder = st.empty()
        full_response = ""
        final_metadata = {}
        
        try:
            for chunk in config["ollama_service"].chat_stream(
                model=config["model"],
                messages=api_messages,
                **config["params"]
            ):
                if "error" in chunk:
                    st.error(f"Error: {chunk['error']}")
                    break
                
                full_response = chunk.get("full_response", "")
                final_metadata = chunk.get("metadata", {})
                
                # Update the response in real-time
                with response_placeholder.container():
                    current_time = format_timestamp(time.time(), "time")
                    st.markdown(f"""
                    <div class="timestamp">
                        {current_time}
                    </div>
                    <div class="chat-message assistant-message">
                        {full_response}
                    </div>
                    """, unsafe_allow_html=True)
                
                if chunk.get("done"):
                    break
            
            # Add final response to chat history
            if full_response:
                add_message("assistant", full_response, final_metadata)
                
                # Clear the placeholder and show final message with metrics
                response_placeholder.empty()
                with chat_container:
                    render_chat_message(st.session_state.messages[-1])
            
        except Exception as e:
            st.error(f"An error occurred: {e}")


def handle_user_input(prompt: str, config: Dict[str, Any], chat_container):
    """
    Handle user input and route to appropriate processing function.
    
    Args:
        prompt: User's input prompt
        config: Configuration dictionary from sidebar
        chat_container: Streamlit container for displaying chat
    """
    # Add user message to chat history
    add_message("user", prompt)
    
    # Display user message immediately
    with chat_container:
        render_chat_message(st.session_state.messages[-1])
    
    # Route to appropriate processing function based on mode
    if config["agent_mode"]:
        process_agent_query(prompt, config, chat_container)
    else:
        process_normal_chat(prompt, config, chat_container)


# Import format_timestamp here to avoid circular imports
def format_timestamp(timestamp: float, format_type: str = "time") -> str:
    """
    Format timestamp for display.
    
    Args:
        timestamp: Unix timestamp
        format_type: Type of formatting ("time", "datetime", "date")
    
    Returns:
        Formatted timestamp string
    """
    from utils.chat_utils import format_timestamp as _format_timestamp
    return _format_timestamp(timestamp, format_type)
