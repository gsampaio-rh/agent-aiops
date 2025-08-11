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
from core.models.agent import AgentStep, StepType
from utils.logger import (
    get_logger, log_user_interaction, log_tool_execution,
    create_request_tracer, log_request_step
)

# Initialize logger for this module
logger = get_logger(__name__)


def _extract_command_output(raw_output: str) -> str:
    """
    Extract the actual command output from Desktop Commander's response format.
    
    Desktop Commander returns output in format:
    "Process started with PID XXXXX (shell: /bin/sh)\nInitial output:\n[ACTUAL_OUTPUT]"
    
    Args:
        raw_output: Raw output from Desktop Commander
        
    Returns:
        str: Clean command output
    """
    if not raw_output:
        return ""
    
    # Look for "Initial output:" marker
    if "Initial output:\n" in raw_output:
        parts = raw_output.split("Initial output:\n", 1)
        if len(parts) > 1:
            return parts[1].rstrip('\n')
    
    # Alternative: if it starts with process info, skip that line
    if raw_output.startswith("Process started with PID"):
        lines = raw_output.split('\n')
        if len(lines) > 2:  # PID line, "Initial output:", actual output
            # Find the first line that's not process info or "Initial output:"
            for i, line in enumerate(lines):
                if line == "Initial output:":
                    # Return everything after this line
                    return '\n'.join(lines[i+1:]).rstrip('\n')
            # Fallback: skip first line (PID info)
            return '\n'.join(lines[1:]).rstrip('\n')
    
    # If no special format detected, return as-is
    return raw_output.rstrip('\n')


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
            <div class="permission-title">🔧 Allow {tool_name}?</div>
        </div>
        <div class="permission-content">
            <div class="tool-info">
                {description}
            </div>
            <div class="tool-query">
                📝 {query}
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

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("✅ Allow", key="tool_allow", use_container_width=True, type="primary"):
            if st.session_state.current_request:
                # Log user permission approval
                logger.info(f"User approved tool execution: {tool_name} with query: {query[:100]}...")
                log_user_interaction(
                    action="tool_permission_approved",
                    mode="tool_execution",
                    tool_name=tool_name,
                    query=query,
                    decision="approved"
                )
                
                st.session_state.current_request["permission_status"] = "approved"
                st.session_state.current_request["execution_status"] = "not_started"
                st.session_state.current_request["query"] = query  # Use original query
                st.rerun()

    with col2:
        if st.button("❌ Cancel", key="tool_cancel", use_container_width=True):
            # Log user permission denial
            logger.info(f"User cancelled tool execution: {tool_name}")
            log_user_interaction(
                action="tool_permission_denied",
                mode="tool_execution",
                tool_name=tool_name,
                query=query,
                decision="cancelled"
            )
            
            st.session_state.current_request = None
            st.rerun()

    with col3:
        if st.button("✏️ Allow Modified", key="tool_save_execute", use_container_width=True):
            if st.session_state.current_request:
                # Log user permission with modification
                logger.info(f"User approved modified tool execution: {tool_name}, original: {query[:50]}..., modified: {modified_query[:50]}...")
                log_user_interaction(
                    action="tool_permission_modified",
                    mode="tool_execution",
                    tool_name=tool_name,
                    original_query=query,
                    modified_query=modified_query,
                    decision="modified"
                )
                
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
                # Log retry from error
                logger.info(f"User retrying tool execution after error: {tool_name} - {error}")
                log_user_interaction(
                    action="tool_execution_retry_from_error",
                    mode="tool_execution",
                    tool_name=tool_name,
                    error_message=error,
                    decision="retry"
                )
                
                st.session_state.current_request["permission_status"] = "pending"
                st.session_state.current_request["execution_status"] = "not_started"
                st.session_state.current_request["error"] = None
                st.rerun()
    
    with col3:
        if st.button("❌ Cancel", key="tool_cancel_error", use_container_width=True):
            # Log cancellation from error
            logger.info(f"User cancelled tool execution after error: {tool_name} - {error}")
            log_user_interaction(
                action="tool_execution_cancelled_from_error",
                mode="tool_execution",
                tool_name=tool_name,
                error_message=error,
                decision="cancelled"
            )
            
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
        # Log initial tool execution request
        logger.info(f"Tool execution requested: {tool_name} for query: {original_user_query[:100]}...")
        log_user_interaction(
            action="tool_execution_requested",
            mode="tool_execution",
            tool_name=tool_name,
            query=query,
            description=description,
            original_user_query=original_user_query
        )
        
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
            execution_start_time = time.time()
            
            # Log tool execution start
            logger.info(f"Starting tool execution: {tool_name} with query: {final_query[:100]}...")
            log_user_interaction(
                action="tool_execution_started",
                mode="tool_execution",
                tool_name=tool_name,
                query=final_query,
                permission_type=request["permission_status"]
            )
            
            tool_result = agent.execute_tool(tool_name, final_query)
            execution_duration = round((time.time() - execution_start_time) * 1000)
            
            if tool_result["success"]:
                request["execution_status"] = "completed"
                request["result"] = tool_result
                request["error"] = None
                
                # Log successful tool execution
                logger.info(f"Tool execution completed successfully: {tool_name} in {execution_duration}ms")
                log_tool_execution(
                    tool_name=tool_name,
                    action="ui_execution",
                    query=final_query,
                    result=tool_result,
                    duration_ms=execution_duration,
                    success=True,
                    user_initiated=True,
                    permission_type=request["permission_status"]
                )
            else:
                request["execution_status"] = "failed"
                request["error"] = tool_result.get("error", "Unknown error")
                request["result"] = None
                
                # Log failed tool execution
                logger.error(f"Tool execution failed: {tool_name} - {tool_result.get('error', 'Unknown error')}")
                log_tool_execution(
                    tool_name=tool_name,
                    action="ui_execution",
                    query=final_query,
                    result=tool_result,
                    duration_ms=execution_duration,
                    success=False,
                    user_initiated=True,
                    permission_type=request["permission_status"],
                    error_reason=tool_result.get("error", "Unknown error")
                )
        except Exception as e:
            execution_duration = round((time.time() - execution_start_time) * 1000) if 'execution_start_time' in locals() else 0
            request["execution_status"] = "failed"
            request["error"] = str(e)
            request["result"] = None
            
            # Log exception in tool execution
            logger.exception(f"Tool execution exception: {tool_name} - {str(e)}")
            log_tool_execution(
                tool_name=tool_name,
                action="ui_execution",
                query=final_query,
                result={"success": False, "error": str(e)},
                duration_ms=execution_duration,
                success=False,
                user_initiated=True,
                permission_type=request["permission_status"],
                error_type=type(e).__name__,
                error_message=str(e)
            )
        
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
        
        # Show brief summary - handle different tool types
        if tool_name == "terminal":
            # Terminal tool specific summary
            exit_code = metadata.get('exit_code', 0)
            status = "✅ Success" if exit_code == 0 else f"❌ Exit code {exit_code}"
            output_length = len(result.get("output", ""))
            st.info(f"🔧 **{tool_name}** executed command with {status} - {output_length} characters output")
            
            # Show expandable results preview for terminal with options
            with st.expander("📋 Preview Command Output", expanded=False):
                # Toggle between clean and raw output
                show_raw = st.checkbox("Show full JSON response", key="show_raw_output")
                
                if show_raw:
                    # Show the complete result structure
                    import json
                    st.markdown("**Full Result JSON:**")
                    st.json(result)
                else:
                    # Clean up the output to remove MCP metadata
                    raw_output = result.get("output", "No output")
                    clean_output = _extract_command_output(raw_output)
                    st.markdown(f"""
                    <div class="tool-results-container">
                        <pre class="tool-results">{clean_output}</pre>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Web search and other tools
            st.info(f"🔧 **{tool_name}** retrieved {metadata.get('total_results', 0)} results from {metadata.get('provider', 'search')} in {metadata.get('search_time_ms', 0)}ms")
            
            # Show expandable results preview
            with st.expander("📋 Preview Results", expanded=False):
                # Check for results in both possible keys (ReactAgent uses "results", LangGraph uses "output")
                results_content = result.get("results") or result.get("output", "No results")
                st.markdown(f"""
                <div class="tool-results-container">
                    <pre class="tool-results">{results_content}</pre>
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
            if st.button("✅ Accept", key="tool_accept", use_container_width=True, type="primary"):
                if st.session_state.current_request:
                    # Log user acceptance of tool results
                    results_data = result.get("output", "") if tool_name == "terminal" else (result.get("results") or result.get("output", ""))
                    logger.info(f"User accepted tool results: {tool_name} - {len(str(results_data))} characters of results")
                    log_user_interaction(
                        action="tool_results_accepted",
                        mode="tool_execution",
                        tool_name=tool_name,
                        decision="accepted",
                        results_length=len(str(results_data)),
                        execution_metadata=metadata
                    )
                    
                    # Accept: Add TOOL_RESULT step to timeline and trigger finalization
                    
                    # Create TOOL_RESULT step
                    tool_result_step = AgentStep(
                        step_type=StepType.TOOL_RESULT,
                        content=f"Retrieved {metadata.get('total_results', 0)} results from {metadata.get('provider', 'search')} in {metadata.get('search_time_ms', 0)}ms" if tool_name != "terminal" else f"Executed command with exit code {metadata.get('exit_code', 0)}",
                        timestamp=time.time(),
                        metadata={
                            "tool_name": tool_name,
                            "provider": metadata.get("provider", "terminal" if tool_name == "terminal" else "Unknown"),
                            "search_time_ms": metadata.get("search_time_ms", 0),
                            "total_results": metadata.get("total_results", 0),
                            "results_preview": (result.get("output", "")[:200] + "..." if len(result.get("output", "")) > 200 else result.get("output", "")) if tool_name == "terminal" else ((result.get("results") or result.get("output", ""))[:200] + "..." if len(result.get("results") or result.get("output", "")) > 200 else (result.get("results") or result.get("output", ""))),
                            "full_results": result.get("output", "") if tool_name == "terminal" else (result.get("results") or result.get("output", "")),
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
            if st.button("🔄 Retry", key="tool_retry_query", use_container_width=True):
                if st.session_state.current_request:
                    # Log user retry decision
                    logger.info(f"User requested retry of tool execution: {tool_name}")
                    log_user_interaction(
                        action="tool_results_retry",
                        mode="tool_execution",
                        tool_name=tool_name,
                        decision="retry",
                        reason="user_requested_retry"
                    )
                    
                    # Retry: Reset to pending state
                    st.session_state.current_request["permission_status"] = "pending"
                    st.session_state.current_request["execution_status"] = "not_started"
                    st.session_state.current_request["result"] = None
                    st.session_state.current_request["error"] = None
                    st.rerun()
        
        with col3:
            if st.button("❌ Reject", key="tool_reject", use_container_width=True):
                if st.session_state.current_request:
                    # Log user rejection of tool results
                    results_data = result.get("output", "") if tool_name == "terminal" else (result.get("results") or result.get("output", ""))
                    logger.info(f"User rejected tool results: {tool_name} - declined {len(str(results_data))} characters of results")
                    log_user_interaction(
                        action="tool_results_rejected",
                        mode="tool_execution",
                        tool_name=tool_name,
                        decision="rejected",
                        results_length=len(str(results_data)),
                        execution_metadata=metadata,
                        reason="user_declined_results"
                    )
                    
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
