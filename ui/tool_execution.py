"""
Tool Execution Components for the Streamlit Ollama Chatbot Application.

This module contains components for tool permission and execution workflow:
- Tool permission cards and approval interfaces
- Tool execution progress indicators
- Tool execution result displays (success/failure)
- Complete tool execution workflow handling

All components follow Jobs/Ive design principles with minimalist aesthetics.
"""

import streamlit as st
import time
from typing import Dict, Any
from services.agent_service import AgentStep, StepType


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


def handle_tool_execution_workflow(tool_name: str, query: str, description: str, agent, config: Dict[str, Any], original_user_query: str = "") -> bool:
    """
    Handle the complete tool execution workflow with permission and progress.
    
    Args:
        tool_name: Name of the tool to execute
        query: Query parameters for the tool
        description: Tool description
        agent: Agent instance to execute the tool
        config: Configuration dictionary
        original_user_query: The original user query that triggered this tool use
        
    Returns:
        bool: True if execution completed (success or failure), False if still in progress
    """
    # Create or update current request
    if not st.session_state.current_request:
        st.session_state.current_request = {
            "tool_name": tool_name,
            "query": query,
            "description": description,
            "original_user_query": original_user_query,
            "permission_status": "pending",
            "execution_status": "not_started",
            "result": None,
            "error": None
        }
    
    request = st.session_state.current_request
    
    # Handle permission flow
    if request["permission_status"] == "pending":
        render_tool_permission_card(tool_name, query, description)
        return False
    
    # Handle execution flow
    elif request["execution_status"] == "not_started" and request["permission_status"] in ["approved", "modified"]:
        # Start execution
        request["execution_status"] = "executing"
        
        # Show progress indicator
        render_tool_execution_progress(tool_name)
        
        # Execute the tool
        try:
            final_query = request["query"]  # This will be the original or modified query
            tool_result = agent.execute_tool(tool_name, final_query)
            
            if tool_result["success"]:
                request["execution_status"] = "completed"
                request["result"] = tool_result
                request["error"] = None
            else:
                request["execution_status"] = "failed"
                request["error"] = tool_result.get("error", "Unknown error")
                request["result"] = None
        except Exception as e:
            request["execution_status"] = "failed"
            request["error"] = str(e)
            request["result"] = None
        
        # Force rerun to show results
        st.rerun()
    
    # Handle execution results - show approval interface
    elif request["execution_status"] == "completed":
        result = request["result"]
        metadata = result.get("metadata", {})
        
        # Show approval interface instead of separate results card
        st.markdown("""
        <div class="tool-approval-interface">
            <div class="approval-header">
                <span class="approval-icon">✅</span>
                <span class="approval-title">Tool Execution Completed</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show brief summary
        st.info(f"🔧 **{tool_name}** retrieved {metadata.get('total_results', 0)} results from {metadata.get('provider', 'search')} in {metadata.get('search_time_ms', 0)}ms")
        
        # Show expandable results preview
        with st.expander("📋 Preview Results", expanded=False):
            st.markdown(f"""
            <div class="tool-results-container">
                <pre class="tool-results">{result.get("results", "No results")}</pre>
            </div>
            """, unsafe_allow_html=True)
        
        # Action buttons for result approval
        st.markdown("""
        <div class="results-approval-section">
            <div class="approval-question">
                <strong>Should these results inform the assistant's final answer?</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1.5, 1.5])
        
        with col1:
            if st.button("✅ Accept & Continue", key="tool_accept", use_container_width=True, type="primary"):
                if st.session_state.current_request:
                    # Accept: Add TOOL_RESULT step to timeline and trigger finalization
                    
                    # Create TOOL_RESULT step
                    tool_result_step = AgentStep(
                        step_type=StepType.TOOL_RESULT,
                        content=f"Retrieved {metadata.get('total_results', 0)} results from {metadata.get('provider', 'search')} in {metadata.get('search_time_ms', 0)}ms",
                        timestamp=time.time(),
                        metadata={
                            "tool_name": tool_name,
                            "provider": metadata.get("provider", "Unknown"),
                            "search_time_ms": metadata.get("search_time_ms", 0),
                            "total_results": metadata.get("total_results", 0),
                            "results_preview": result.get("results", "")[:200] + "..." if len(result.get("results", "")) > 200 else result.get("results", ""),
                            "full_results": result.get("results", ""),
                            "user_decision": "accepted"
                        }
                    )
                    
                    # Add to agent steps
                    st.session_state.agent_steps.append(tool_result_step)
                    
                    # Store tool context for finalization
                    st.session_state.tool_context = {
                        "result": result,
                        "query": st.session_state.current_request.get("original_user_query", ""),
                        "tool_name": tool_name,
                        "provider": metadata.get("provider", "Unknown")
                    }
                    
                    original_query = st.session_state.current_request.get("original_user_query", "")
                    st.session_state.current_request = None
                    st.session_state.processing = True
                    st.session_state.pending_finalization_query = original_query
                    
                    # Force immediate rerun to show timeline instead of approval UI
                    st.rerun()
        
        with col2:
            if st.button("🔄 Retry with Different Query", key="tool_retry_query", use_container_width=True):
                if st.session_state.current_request:
                    # Retry: Reset to pending state
                    st.session_state.current_request["permission_status"] = "pending"
                    st.session_state.current_request["execution_status"] = "not_started"
                    st.session_state.current_request["result"] = None
                    st.session_state.current_request["error"] = None
                    st.rerun()
        
        with col3:
            if st.button("❌ Reject & Answer Without Tool", key="tool_reject", use_container_width=True):
                if st.session_state.current_request:
                    # Reject: Add TOOL_RESULT step showing rejection and trigger finalization without tools
                    
                    # Create TOOL_RESULT step showing rejection
                    tool_result_step = AgentStep(
                        step_type=StepType.TOOL_RESULT,
                        content=f"Tool results rejected by user. Retrieved {metadata.get('total_results', 0)} results but proceeding without tool context.",
                        timestamp=time.time(),
                        metadata={
                            "tool_name": tool_name,
                            "provider": metadata.get("provider", "Unknown"),
                            "search_time_ms": metadata.get("search_time_ms", 0),
                            "total_results": metadata.get("total_results", 0),
                            "user_decision": "rejected"
                        }
                    )
                    
                    # Add to agent steps
                    st.session_state.agent_steps.append(tool_result_step)
                    
                    original_query = st.session_state.current_request.get("original_user_query", "")
                    st.session_state.tool_context = None
                    st.session_state.current_request = None
                    st.session_state.processing = True
                    st.session_state.pending_finalization_query = original_query
                    
                    # Force immediate rerun to show timeline instead of approval UI
                    st.rerun()
        
        # Don't clear request - wait for user approval action
        return False
    
    elif request["execution_status"] == "failed":
        render_tool_execution_failed(tool_name, request["error"])
        return False
    
    return False
