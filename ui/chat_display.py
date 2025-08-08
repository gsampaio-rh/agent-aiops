"""
Chat Display Components for the Streamlit Ollama Chatbot Application.

This module contains components for rendering chat messages and the main chat interface:
- Chat message rendering with timestamps and metrics
- Chat interface display logic
- Integration with agent steps and tool execution

All components follow Jobs/Ive design principles with minimalist aesthetics.
"""

import streamlit as st
import time
from typing import Dict, Any

from config.settings import SHOW_DETAILED_METRICS
from utils.chat_utils import format_timestamp, format_metrics


def render_chat_message(message: Dict[str, Any]):
    """
    Render a single chat message with timestamp and metrics.
    
    Args:
        message: Message dictionary containing role, content, timestamp, and metadata
    """
    role = message["role"]
    content = message["content"]
    timestamp = message.get("timestamp", time.time())
    metadata = message.get("metadata", {})
    
    # Format timestamp
    formatted_time = format_timestamp(timestamp, "time")
    
    # Determine CSS class based on role
    message_class = "user-message" if role == "user" else "assistant-message"
    
    # Format metrics if available and detailed metrics are enabled
    metrics_text = ""
    if metadata and role == "assistant":
        detailed = st.session_state.get("show_detailed_metrics", SHOW_DETAILED_METRICS)
        metrics_text = f'<div class="metrics">{format_metrics(metadata, detailed)}</div>'
    
    # Render the message
    st.markdown(f"""
    <div class="timestamp">
        {formatted_time}
    </div>
    <div class="chat-message {message_class}">
        {content}
        {metrics_text}
    </div>
    """, unsafe_allow_html=True)


def display_chat_interface(config: Dict[str, Any], chat_container):
    """
    Display the main chat interface with messages and agent steps.
    
    Args:
        config: Configuration dictionary from render_sidebar()
        chat_container: Streamlit container for chat messages
    """
    with chat_container:
        if config["agent_mode"]:
            # In agent mode, show messages and optionally steps
            all_items = []
            
            # Add regular messages
            for msg in st.session_state.messages:
                all_items.append(("message", msg))
            
            # Add agent steps only if toggle is enabled and not currently processing
            if st.session_state.show_agent_steps and not st.session_state.agent_processing:
                for step in st.session_state.agent_steps:
                    all_items.append(("step", step))
            elif st.session_state.agent_steps and not st.session_state.agent_processing:
                # Show a subtle message when steps are hidden but exist
                st.markdown("""
                <div class="steps-hidden-message">
                    💭 Agent reasoning steps are hidden • Toggle "Show reasoning steps" in sidebar to view
                </div>
                """, unsafe_allow_html=True)
            
            # Sort by timestamp
            all_items.sort(key=lambda x: x[1].get("timestamp", 0) if x[0] == "message" else x[1].timestamp)
            
            # Render items
            for item_type, item in all_items:
                if item_type == "message":
                    render_chat_message(item)
                elif item_type == "step":
                    from ui.agent_display import render_agent_step
                    render_agent_step(item)
                    
            # Handle pending finalization first
            if st.session_state.processing and not st.session_state.current_request and st.session_state.pending_finalization_query:
                # Import the finalization handler here to avoid circular imports
                from ui.chat_processing import finalize_assistant_response_with_timeline
                
                query_to_finalize = st.session_state.pending_finalization_query
                st.session_state.pending_finalization_query = None  # Clear immediately
                finalize_assistant_response_with_timeline(query_to_finalize, config, st.container())
            
            # Handle active tool execution workflow if exists
            elif st.session_state.current_request:
                request = st.session_state.current_request
                
                # Import the workflow handler here to avoid circular imports
                from ui.tool_execution import handle_tool_execution_workflow
                
                handle_tool_execution_workflow(
                    request["tool_name"],
                    request["query"], 
                    request["description"],
                    st.session_state.agent,
                    config,
                    request.get("original_user_query", "")
                )
        else:
            # Normal mode - just show messages
            for message in st.session_state.messages:
                render_chat_message(message)
