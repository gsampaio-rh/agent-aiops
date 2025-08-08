"""
UI Components for the Streamlit Ollama Chatbot Application.

This module contains all reusable UI components including:
- Sidebar with model selection and parameters
- Chat message rendering  
- Agent step visualization
- Session management controls

All components follow Jobs/Ive design principles with minimalist aesthetics.
"""

import streamlit as st
import time
from typing import Dict, Any

from config.settings import (
    APP_TITLE, APP_ICON, DEFAULT_MODEL, MODEL_PARAMS, SHOW_DETAILED_METRICS
)
from services.ollama_service import OllamaService
from services.agent_service import ReactAgent, AgentStep, StepType
from utils.chat_utils import (
    clear_chat_history, format_metrics, export_chat_history, format_timestamp
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
                st.session_state.agent = ReactAgent(current_model)
            
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
                "ollama_service": ollama_service,
                "agent_mode": st.session_state.agent_mode
            }
    
    except Exception as e:
        st.error(f"Sidebar error: {e}")
        return {}


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


def render_agent_step_with_state(step: AgentStep, state: str = ""):
    """
    Render a single agent step with state-aware styling.
    
    Args:
        step: AgentStep object containing step information
        state: Current state of the step ("current", "completed", or "")
    """
    step_config = {
        StepType.THOUGHT: {
            "label": "Think",
            "class": "step-thought"
        },
        StepType.TOOL_SELECTION: {
            "label": "Select",
            "class": "step-tool-selection"
        },
        StepType.TOOL_USE: {
            "label": "Use",
            "class": "step-tool-use"
        },
        StepType.TOOL_RESULT: {
            "label": "Result",
            "class": "step-tool-result"
        },
        StepType.FINAL_ANSWER: {
            "label": "Answer",
            "class": "step-final-answer"
        },
        StepType.ERROR: {
            "label": "Error",
            "class": "step-error"
        }
    }
    
    config = step_config.get(step.step_type, {
        "label": "Info",
        "class": "step-thought"
    })
    
    # Format metadata if available
    metadata_text = ""
    if step.metadata:
        metadata_parts = []
        for key, value in step.metadata.items():
            if key not in ["status"]:  # Skip internal status
                if key == "search_time_ms":
                    metadata_parts.append(f"{value}ms")
                elif key == "total_results":
                    metadata_parts.append(f"{value} results")
                elif key == "provider":
                    metadata_parts.append(f"{value}")
                elif key == "tool":
                    metadata_parts.append(f"{value}")
                else:
                    metadata_parts.append(f"{key}: {value}")
        
        if metadata_parts:
            metadata_text = f"<div class='step-metadata'>{' • '.join(metadata_parts)}</div>"
    
    # Add state class for styling
    state_class = f" {state}" if state else ""
    
    # Render the step with minimal design and state
    st.markdown(f"""
    <div class="thinking-step {config['class']}{state_class} agent-step-stream">
        <span class="step-label">{config['label']}</span>
        <span class="step-content">{step.content}</span>
    </div>
    {metadata_text}
    """, unsafe_allow_html=True)


def render_agent_step(step: AgentStep):
    """
    Render a single agent step with ultra-minimalist design (legacy support).
    
    Args:
        step: AgentStep object containing step information
    """
    render_agent_step_with_state(step, "")


def render_tool_permission_card(tool_name: str, query: str, description: str):
    """
    Render a tool permission card with Allow/Save & Execute buttons.
    
    Args:
        tool_name: Name of the tool requesting permission
        query: The query/parameters for the tool
        description: Description of what the tool does
    """
    st.markdown(f"""
    <div class="tool-permission-card">
        <div class="permission-header">
            <span class="permission-icon">🔧</span>
            <span class="permission-title">Tool Permission Required</span>
        </div>
        <div class="permission-content">
            <div class="tool-info">
                <strong>{tool_name}</strong> wants to execute
            </div>
            <div class="tool-description">{description}</div>
            <div class="tool-query">
                <strong>Query:</strong> {query}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create input field for potentially modifying the query
    modified_query = st.text_input(
        "Modify query (optional):",
        value=query,
        key="tool_query_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🚫 Cancel", key="tool_cancel", use_container_width=True):
            st.session_state.current_request = None
            st.rerun()
    
    with col2:
        if st.button("✅ Allow", key="tool_allow", use_container_width=True):
            if st.session_state.current_request:
                st.session_state.current_request["permission_status"] = "approved"
                st.session_state.current_request["execution_status"] = "not_started"
                st.session_state.current_request["query"] = query  # Use original query
                st.rerun()
    
    with col3:
        if st.button("💾 Save & Execute", key="tool_save_execute", use_container_width=True):
            if st.session_state.current_request:
                st.session_state.current_request["permission_status"] = "modified"
                st.session_state.current_request["execution_status"] = "not_started"
                st.session_state.current_request["query"] = modified_query  # Use modified query
                st.rerun()


def render_tool_execution_progress(tool_name: str):
    """
    Render tool execution progress indicator.
    
    Args:
        tool_name: Name of the tool being executed
    """
    with st.spinner(f"🔄 Executing {tool_name}..."):
        st.markdown(f"""
        <div class="tool-execution-progress">
            <div class="execution-header">
                <span class="execution-icon">🔄</span>
                <span class="execution-title">Executing {tool_name}...</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_tool_execution_success(tool_name: str, provider: str, duration_ms: int, 
                                 total_results: int, results: str, config: Dict[str, Any] = None):
    """
    Render successful tool execution results card with approval actions.
    
    Args:
        tool_name: Name of the executed tool
        provider: Provider used (e.g., DuckDuckGo)
        duration_ms: Execution duration in milliseconds
        total_results: Number of results returned
        results: The actual results content
    """
    st.markdown(f"""
    <div class="tool-execution-success">
        <div class="execution-header">
            <span class="execution-icon">✅</span>
            <span class="execution-title">Tool Execution Successful</span>
        </div>
        <div class="execution-metadata">
            <span class="metadata-item">🔧 {tool_name}</span>
            <span class="metadata-item">🌐 {provider}</span>
            <span class="metadata-item">⏱️ {duration_ms}ms</span>
            <span class="metadata-item">📊 {total_results} results</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Scrollable results area
    with st.expander("📋 View Results", expanded=True):
        st.markdown(f"""
        <div class="tool-results-container">
            <pre class="tool-results">{results}</pre>
        </div>
        """, unsafe_allow_html=True)
    



def render_tool_execution_failed(tool_name: str, error: str):
    """
    Render failed tool execution error card.
    
    Args:
        tool_name: Name of the tool that failed
        error: Error message
    """
    st.markdown(f"""
    <div class="tool-execution-failed">
        <div class="execution-header">
            <span class="execution-icon">❌</span>
            <span class="execution-title">Tool Execution Failed</span>
        </div>
        <div class="error-content">
            <div class="error-tool">Tool: {tool_name}</div>
            <div class="error-message">{error}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col2:
        if st.button("🔄 Retry", key="tool_retry", use_container_width=True):
            if st.session_state.current_request:
                st.session_state.current_request["permission_status"] = "pending"
                st.session_state.current_request["execution_status"] = "not_started"
                st.session_state.current_request["error"] = None
                st.rerun()
    
    with col3:
        if st.button("❌ Cancel", key="tool_cancel_error", use_container_width=True):
            st.session_state.current_request = None
            st.rerun()


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
                    render_agent_step(item)
                    
            # Handle pending finalization first
            if st.session_state.processing and not st.session_state.current_request and st.session_state.pending_finalization_query:
                # Import the finalization handler here to avoid circular imports
                from ui.streamlit_utils import finalize_assistant_response_with_timeline
                
                query_to_finalize = st.session_state.pending_finalization_query
                st.session_state.pending_finalization_query = None  # Clear immediately
                finalize_assistant_response_with_timeline(query_to_finalize, config, st.container())
            
            # Handle active tool execution workflow if exists
            elif st.session_state.current_request:
                request = st.session_state.current_request
                
                # Import the workflow handler here to avoid circular imports
                from ui.streamlit_utils import handle_tool_execution_workflow
                
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
