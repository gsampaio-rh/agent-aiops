"""
Sidebar Components for the Streamlit Ollama Chatbot Application.

This module contains all sidebar-related UI components including:
- Model selection and parameters
- Agent mode toggles
- Session management controls
- Chat statistics display

All components follow Jobs/Ive design principles with minimalist aesthetics.
"""

import streamlit as st
import time
from typing import Dict, Any

from config.settings import (
    APP_TITLE, APP_ICON, DEFAULT_MODEL, MODEL_PARAMS, SHOW_DETAILED_METRICS
)
from services.ollama_service import OllamaService
from services.agent_factory import create_agent, AgentFactory
from utils.chat_utils import (
    clear_chat_history, export_chat_history
)


def render_sidebar() -> Dict[str, Any]:
    """
    Render the sidebar with model selection and parameters.
    
    Returns:
        Dict[str, Any]: Configuration dictionary containing:
            - model: Selected model name
            - params: Model parameters
            - ollama_service: OllamaService instance
            - agent_mode: Whether agent mode is enabled
    """
    try:
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
            st.subheader("🤖 Model")
            available_models = ollama_service.get_available_models()
            
            if not available_models:
                st.error("No models available. Please install a model:")
                st.code(f"ollama pull {DEFAULT_MODEL}")
                return {}
            
            current_model = st.selectbox(
                "Language Model",
                options=available_models,
                index=available_models.index(st.session_state.current_model) 
                      if st.session_state.current_model in available_models 
                      else 0,
                help="Select the language model to use for conversations"
            )
            
            # Update agent if model changed
            if current_model != st.session_state.current_model:
                st.session_state.current_model = current_model
                st.session_state.agent = create_agent(model=current_model)
            
            # Agent mode toggle
            st.subheader("🧠 Mode")
            agent_mode = st.radio(
                "Chat Mode",
                options=["Normal Chat", "Agent Mode"],
                index=1 if st.session_state.agent_mode else 0,
                help="Normal: Direct chat | Agent: Shows reasoning and can use tools"
            )
            st.session_state.agent_mode = (agent_mode == "Agent Mode")
            
            if st.session_state.agent_mode:
                # Show available tools
                with st.expander("🛠️ Available Tools"):
                    if st.session_state.agent and hasattr(st.session_state.agent, 'tools'):
                        for tool_name, tool in st.session_state.agent.tools.items():
                            st.write(f"📡 **{tool_name}**: {tool.description[:80]}...")
                    else:
                        st.write("⚠️ Agent not properly initialized")
                
                # Agent steps visibility toggle  
                st.session_state.show_agent_steps = st.checkbox(
                    "💭 Show thinking process",
                    value=st.session_state.show_agent_steps,
                    help="Toggle visibility of the agent's step-by-step reasoning and tool usage"
                )
            
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
            st.subheader("📊 Metrics")
            show_detailed_metrics = st.checkbox(
                "Show detailed metrics",
                value=st.session_state.get("show_detailed_metrics", SHOW_DETAILED_METRICS),
                help="Show comprehensive performance metrics including timing breakdowns and throughput analysis"
            )
            st.session_state.show_detailed_metrics = show_detailed_metrics
            
            # Session management
            st.subheader("💾 Session")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    clear_chat_history()
                    st.session_state.agent_steps = []
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
            
            # Chat statistics and memory info
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
                
                # Memory status indicator
                st.subheader("🧠 Memory Status")
                from config.constants import AGENT_CONFIG
                max_memory = AGENT_CONFIG.get("max_conversation_history", 20)
                memory_usage = min(total_messages, max_memory)
                
                # Memory indicator with visual bar
                memory_percentage = (memory_usage / max_memory) * 100
                st.metric("Memory Usage", f"{memory_usage}/{max_memory}")
                st.progress(memory_percentage / 100, text=f"{memory_percentage:.0f}% of memory used")
                
                if st.session_state.agent_mode and total_messages > 0:
                    st.success("🔄 Conversation memory active")
                    st.caption("Agent remembers your conversation history")
                elif total_messages > 0:
                    st.info("💬 Chat history available")
                    st.caption("Switch to Agent mode for memory features")
            
            return {
                "model": current_model,
                "params": params,
                "ollama_service": ollama_service,
                "agent_mode": st.session_state.agent_mode
            }
    
    except Exception as e:
        st.error(f"Sidebar error: {e}")
        return {}
