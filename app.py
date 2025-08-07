"""
Streamlit Ollama Chatbot Application

A minimalist, high-end chatbot interface built on Ollama with real-time metrics.
Inspired by Dieter Rams' design principles and Apple's user experience.
"""

import streamlit as st
import time
from typing import Dict, Any

from config.settings import (
    APP_TITLE, APP_ICON, DEFAULT_MODEL, MODEL_PARAMS, SHOW_DETAILED_METRICS
)
from services.ollama_service import OllamaService
from utils.chat_utils import (
    initialize_session_state, add_message, clear_chat_history, 
    format_metrics, export_chat_history, format_timestamp
)


# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Apple-inspired design
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-radius: 18px;
        max-width: 80%;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        line-height: 1.4;
    }
    
    .user-message {
        background: #007AFF;
        color: white;
        margin-left: auto;
        text-align: right;
    }
    
    .assistant-message {
        background: #F2F2F7;
        color: #000000;
        margin-right: auto;
    }
    
    /* Metrics styling */
    .metrics {
        font-size: 0.75rem;
        color: #8E8E93;
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
        line-height: 1.3;
        max-width: 80%;
    }
    
    /* Timestamp styling */
    .timestamp {
        font-size: 0.7rem;
        color: #8E8E93;
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
        opacity: 0.8;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #F8F9FA;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 1px solid #D1D5DB;
        padding: 0.75rem 1rem;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 12px;
        border: none;
        background: #007AFF;
        color: white;
        font-weight: 500;
        padding: 0.5rem 1rem;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
    }
    
    .stButton > button:hover {
        background: #0056CC;
        border: none;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def render_sidebar() -> Dict[str, Any]:
    """Render the sidebar with model selection and parameters."""
    with st.sidebar:
        st.title(f"{APP_ICON} {APP_TITLE}")
        
        # Ollama service health check
        ollama_service = OllamaService()
        if not ollama_service.health_check():
            st.error("❌ Ollama service is not running")
            st.info("Please start Ollama: `ollama serve`")
            return {}
        
        st.success("✅ Ollama is running")
        
        # Model selection
        st.subheader("🤖 Model Selection")
        available_models = ollama_service.get_available_models()
        
        if not available_models:
            st.error("No models available. Please install a model:")
            st.code(f"ollama pull {DEFAULT_MODEL}")
            return {}
        
        current_model = st.selectbox(
            "Choose Model",
            options=available_models,
            index=available_models.index(st.session_state.current_model) 
                  if st.session_state.current_model in available_models 
                  else 0,
            help="Select the language model to use for conversations"
        )
        
        st.session_state.current_model = current_model
        
        # Model parameters
        st.subheader("⚙️ Parameters")
        
        params = {}
        for param_name, param_config in MODEL_PARAMS.items():
            params[param_name] = st.slider(
                label=param_name.replace('_', ' ').title(),
                min_value=param_config["min"],
                max_value=param_config["max"],
                value=st.session_state.model_params.get(param_name, param_config["default"]),
                step=param_config["step"],
                help=param_config["help"]
            )
        
        st.session_state.model_params = params
        
        # Display settings
        st.subheader("📊 Display")
        show_detailed_metrics = st.checkbox(
            "Show detailed metrics",
            value=st.session_state.get("show_detailed_metrics", SHOW_DETAILED_METRICS),
            help="Show comprehensive performance metrics including token counts, duration breakdowns, and throughput"
        )
        st.session_state.show_detailed_metrics = show_detailed_metrics
        
        # Session management
        st.subheader("💾 Session")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                clear_chat_history()
                st.rerun()
        
        with col2:
            if st.button("📥 Export", use_container_width=True):
                if st.session_state.messages:
                    export_text = export_chat_history()
                    st.download_button(
                        label="Download Chat",
                        data=export_text,
                        file_name=f"chat_export_{int(time.time())}.txt",
                        mime="text/plain"
                    )
        
        # Chat statistics
        if st.session_state.messages:
            st.subheader("📊 Statistics")
            total_messages = len(st.session_state.messages)
            user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
            assistant_messages = total_messages - user_messages
            
            st.metric("Total Messages", total_messages)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("User", user_messages)
            with col2:
                st.metric("Assistant", assistant_messages)
        
        return {
            "model": current_model,
            "params": params,
            "ollama_service": ollama_service
        }


def render_chat_message(message: Dict[str, Any]):
    """Render a single chat message with timestamp and metrics."""
    role = message["role"]
    content = message["content"]
    metadata = message.get("metadata", {})
    timestamp = message.get("timestamp", time.time())
    
    # Format timestamp
    formatted_time = format_timestamp(timestamp, "time")
    
    if role == "user":
        st.markdown(f"""
        <div class="timestamp" style="text-align: right;">
            {formatted_time}
        </div>
        <div class="chat-message user-message">
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="timestamp">
            {formatted_time}
        </div>
        <div class="chat-message assistant-message">
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        # Show metrics for assistant messages
        if metadata:
            detailed = st.session_state.get("show_detailed_metrics", True)
            metrics_text = format_metrics(metadata, detailed)
            if metrics_text:
                st.markdown(f"""
                <div class="metrics">
                    {metrics_text}
                </div>
                """, unsafe_allow_html=True)


def main():
    """Main application logic."""
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar and get configuration
    config = render_sidebar()
    
    if not config:
        st.error("Please ensure Ollama is running and models are available.")
        st.stop()
    
    # Main chat interface
    st.title("💬 Chat")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            render_chat_message(message)
    
    # Chat input
    if prompt := st.chat_input("Type your message here...", key="chat_input"):
        # Add user message
        add_message("user", prompt)
        
        # Display user message immediately
        with chat_container:
            render_chat_message(st.session_state.messages[-1])
        
        # Generate assistant response
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


if __name__ == "__main__":
    main()
