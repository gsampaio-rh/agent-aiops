"""
Chat Processing for the Streamlit Ollama Chatbot Application.

This module contains functions for processing user input and generating responses:
- Agent query processing with step visualization
- Normal chat processing with streaming
- Response finalization with tool context
- User input handling and routing

All functions are designed to work seamlessly with Streamlit's reactive architecture.
"""

import streamlit as st
import time
from typing import Dict, Any

from services.agent_service import AgentStep, StepType
from utils.chat_utils import add_message
from ui.agent_display import render_agent_step_with_state
from ui.chat_display import render_chat_message


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
            
            # Get the appropriate result content based on tool type
            if tool_context["tool_name"] == "terminal":
                tool_output = result.get('output', 'No output')
                content_type = "command output"
                system_content = f"""You are a helpful AI assistant. You have access to terminal command output from the query "{tool_context['query']}". Use this information to provide a comprehensive answer to the user's question. Do not mention that you performed a terminal command - just provide a natural, helpful answer based on the output.

Command Output:
{tool_output}"""
            else:
                search_results = result.get('results', 'No results')
                content_type = "search results"
                system_content = f"""You are a helpful AI assistant. You have access to search results from {tool_context['provider']} for the query "{tool_context['query']}". Use this information to provide a comprehensive answer to the user's question. Do not mention that you performed a search or reference the search process - just provide a natural, helpful answer based on the information.

Search Results:
{search_results}"""

            messages = [
                {
                    "role": "system", 
                    "content": system_content
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
        for chunk in st.session_state.agent.llm_service.chat_stream(
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
        from ui.tool_execution import handle_tool_execution_workflow
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
        # Show sophisticated thinking indicator
        with thinking_placeholder.container():
            from ui.agent_display import render_sophisticated_thinking_indicator
            render_sophisticated_thinking_indicator("Agent is thinking")
        
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
                    from ui.tool_execution import handle_tool_execution_workflow
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
    from utils.chat_utils import format_timestamp
    
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
            for chunk in st.session_state.agent.llm_service.chat_stream(
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
