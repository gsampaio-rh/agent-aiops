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
from ui.components import (
    render_agent_step_with_state, render_chat_message, 
    render_tool_permission_card, render_tool_execution_progress,
    render_tool_execution_success, render_tool_execution_failed
)


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
                    from services.agent_service import AgentStep, StepType
                    
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
                    from services.agent_service import AgentStep, StepType
                    
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


def finalize_assistant_response(prompt: str, config: Dict[str, Any], chat_container):
    """
    Generate the final assistant response based on tool context or without tool context.
    
    Args:
        prompt: Original user query
        config: Configuration dictionary
        chat_container: Streamlit container for displaying chat
    """
    try:
        with st.spinner("Generating final answer..."):
            # Prepare messages based on whether we have tool context
            if st.session_state.tool_context:
                # Accept & Continue: Use tool results as hidden system context
                tool_context = st.session_state.tool_context
                result = tool_context["result"]
                
                messages = [
                    {
                        "role": "system", 
                        "content": f"""You are a helpful AI assistant. You have access to search results from {tool_context['provider']} for the query "{tool_context['query']}". Use this information to provide a comprehensive answer to the user's question. Do not mention that you performed a search or reference the search process - just provide a natural, helpful answer based on the information.

Search Results:
{result['results']}"""
                    },
                    {"role": "user", "content": prompt}
                ]
                
                # Metadata for the response
                metadata = result.get("metadata", {})
                response_metadata = {
                    "agent_mode": True,
                    "tool_used": tool_context["tool_name"],
                    "search_provider": tool_context["provider"],
                    "search_time_ms": metadata.get("search_time_ms", 0),
                    "total_results": metadata.get("total_results", 0)
                }
            else:
                # Reject & Answer Without Tool: Generate best-effort answer without tool context
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Provide the best answer you can based on your training data, without using any external tools or search capabilities."
                    },
                    {"role": "user", "content": prompt}
                ]
                
                response_metadata = {
                    "agent_mode": True,
                    "tool_context": "rejected"
                }
            
            # Stream the response
            response_placeholder = st.empty()
            full_response = ""
            final_metadata = {}
            
            for chunk in st.session_state.agent.ollama_service.chat_stream(
                model=config["model"],
                messages=messages,
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
                # Merge response metadata with tool metadata
                combined_metadata = {**response_metadata, **final_metadata}
                add_message("assistant", full_response, combined_metadata)
                
                # Clear the placeholder and show final message with metrics
                response_placeholder.empty()
                with chat_container:
                    render_chat_message(st.session_state.messages[-1])
        
        # Clear finalization state
        st.session_state.tool_context = None
        st.session_state.processing = False
        
    except Exception as e:
        st.error(f"Error generating response: {e}")
        # Clear state on error
        st.session_state.tool_context = None
        st.session_state.processing = False


def finalize_assistant_response_with_timeline(prompt: str, config: Dict[str, Any], chat_container):
    """
    Generate the final assistant response and add it as a FINAL_ANSWER step to the timeline.
    
    Args:
        prompt: Original user query
        config: Configuration dictionary
        chat_container: Streamlit container for displaying chat
    """
    try:
        # Prepare messages based on whether we have tool context
        if st.session_state.tool_context:
            # Accept & Continue: Use tool results as hidden system context
            tool_context = st.session_state.tool_context
            result = tool_context["result"]
            
            messages = [
                {
                    "role": "system", 
                    "content": f"""You are a helpful AI assistant. You have access to search results from {tool_context['provider']} for the query "{tool_context['query']}". Use this information to provide a comprehensive answer to the user's question. Do not mention that you performed a search or reference the search process - just provide a natural, helpful answer based on the information.

Search Results:
{result['results']}"""
                },
                {"role": "user", "content": prompt}
            ]
            
            # Metadata for the response
            metadata = result.get("metadata", {})
            response_metadata = {
                "agent_mode": True,
                "tool_used": tool_context["tool_name"],
                "search_provider": tool_context["provider"],
                "search_time_ms": metadata.get("search_time_ms", 0),
                "total_results": metadata.get("total_results", 0)
            }
        else:
            # Reject & Answer Without Tool: Generate best-effort answer without tool context
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Provide the best answer you can based on your training data, without using any external tools or search capabilities."
                },
                {"role": "user", "content": prompt}
            ]
            
            response_metadata = {
                "agent_mode": True,
                "tool_context": "rejected"
            }
        
        # Create a placeholder for streaming response
        response_placeholder = st.empty()
        full_response = ""
        final_metadata = {}
        
        # Stream the response with visual updates
        for chunk in st.session_state.agent.ollama_service.chat_stream(
            model=config["model"],
            messages=messages,
            **config["params"]
        ):
            if "error" in chunk:
                st.error(f"Error: {chunk['error']}")
                break
            
            full_response = chunk.get("full_response", "")
            final_metadata = chunk.get("metadata", {})
            
            # Update the streaming display
            if full_response:
                with response_placeholder.container():
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        {full_response}
                    </div>
                    """, unsafe_allow_html=True)
            
            if chunk.get("done"):
                break
        
        # Clear the streaming placeholder and add the final response to timeline and chat
        response_placeholder.empty()
        
        # Add FINAL_ANSWER step to timeline
        if full_response:
            from services.agent_service import AgentStep, StepType
            
            final_answer_step = AgentStep(
                step_type=StepType.FINAL_ANSWER,
                content=full_response,
                timestamp=time.time(),
                metadata={
                    **response_metadata,
                    **final_metadata,
                    "response_length": len(full_response)
                }
            )
            
            # Add to agent steps
            st.session_state.agent_steps.append(final_answer_step)
            
            # Merge response metadata with tool metadata
            combined_metadata = {**response_metadata, **final_metadata}
            add_message("assistant", full_response, combined_metadata)
            
            # Show the final message in chat
            with chat_container:
                render_chat_message(st.session_state.messages[-1])
        
        # Clear finalization state
        st.session_state.tool_context = None
        st.session_state.processing = False
        
    except Exception as e:
        st.error(f"Error generating response: {e}")
        # Clear state on error
        st.session_state.tool_context = None
        st.session_state.processing = False


def process_agent_query(prompt: str, config: Dict[str, Any], chat_container):
    """
    Process a user query in agent mode with real-time step visualization and tool execution workflow.
    
    Args:
        prompt: User's input prompt
        config: Configuration dictionary from sidebar
        chat_container: Streamlit container for displaying chat
    """
    # Check if agent is properly initialized
    if not st.session_state.agent:
        st.error("Agent not initialized. Please check the configuration.")
        return
    
    # Check for finalization state (processing = True and current_request = None)
    if st.session_state.processing and not st.session_state.current_request and st.session_state.pending_finalization_query:
        query_to_finalize = st.session_state.pending_finalization_query
        st.session_state.pending_finalization_query = None  # Clear immediately
        finalize_assistant_response_with_timeline(query_to_finalize, config, chat_container)
        return
    
    # Check if we have a pending tool execution workflow
    if st.session_state.current_request:
        request = st.session_state.current_request
        tool_name = request["tool_name"]
        query = request["query"]
        description = request["description"]
        original_user_query = request.get("original_user_query", prompt)
        
        # Handle the tool execution workflow
        handle_tool_execution_workflow(
            tool_name, query, description, st.session_state.agent, config, original_user_query
        )
        
        return  # Exit early if we're handling tool execution
    
    # Standard agent processing - but with tool interception
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
        
        # Stream agent steps in real-time, but intercept tool usage
        for step in st.session_state.agent.process_query_stream(
            prompt,
            **config["params"]
        ):
            # Check if this is a tool use step - if so, intercept it
            if step.step_type == StepType.TOOL_USE:
                agent_steps.append(step)
                
                # Parse tool usage
                content = step.content.strip()
                if ':' in content:
                    tool_part, query_part = content.split(':', 1)
                    tool_name = tool_part.strip()
                    tool_query = query_part.strip()
                    
                    # Clean up the query
                    tool_query = tool_query.replace('"', '').replace(' OR ', ' ')
                    if tool_query.startswith('(') and tool_query.endswith(')'):
                        tool_query = tool_query[1:-1]
                    
                    # Get tool description
                    tool_description = "Unknown tool"
                    if tool_name in st.session_state.agent.tools:
                        tool_description = st.session_state.agent.tools[tool_name].description
                    
                    # Clear thinking display and show current steps
                    with thinking_placeholder.container():
                        st.markdown('<div class="agent-thinking">', unsafe_allow_html=True)
                        for i, s in enumerate(agent_steps):
                            render_agent_step_with_state(s, "completed")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Clear processing flag
                    st.session_state.agent_processing = False
                    
                    # Store the steps we've collected so far
                    st.session_state.agent_steps.extend(agent_steps)
                    
                    # Start the tool execution workflow
                    handle_tool_execution_workflow(
                        tool_name, tool_query, tool_description, st.session_state.agent, config, prompt
                    )
                    
                    return  # Exit early to let the tool workflow take over
            
            # Regular step processing
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
