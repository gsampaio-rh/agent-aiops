"""
Agent Display Components for the Streamlit Ollama Chatbot Application.

This module contains components for rendering agent steps and thinking processes:
- Agent step visualization with state-aware styling
- Step metadata formatting
- Thinking indicators and animations

All components follow Jobs/Ive design principles with minimalist aesthetics.
"""

import streamlit as st
from services.agent_service import AgentStep, StepType


def render_agent_step_with_state(step: AgentStep, state: str = ""):
    """
    Render a single agent step with state-aware styling and Jobs/Ive animations.
    
    Args:
        step: AgentStep object containing step information
        state: Current state of the step ("current", "completed", or "")
    """
    step_config = {
        StepType.THOUGHT: {
            "label": "Think",
            "class": "step-thought",
            "icon": "💭"
        },
        StepType.TOOL_SELECTION: {
            "label": "Select",
            "class": "step-tool-selection",
            "icon": "🔍"
        },
        StepType.TOOL_USE: {
            "label": "Use",
            "class": "step-tool-use",
            "icon": "⚡"
        },
        StepType.TOOL_RESULT: {
            "label": "Result",
            "class": "step-tool-result",
            "icon": "📋"
        },
        StepType.FINAL_ANSWER: {
            "label": "Answer",
            "class": "step-final-answer",
            "icon": "✨"
        },
        StepType.ERROR: {
            "label": "Error",
            "class": "step-error",
            "icon": "⚠️"
        }
    }
    
    config = step_config.get(step.step_type, {
        "label": "Info",
        "class": "step-thought",
        "icon": "ℹ️"
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
    
    # Escape content to prevent HTML issues
    escaped_content = step.content.replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
    
    # Add processing indicator for current steps (simplified)
    processing_class = ""
    if state == "current":
        processing_class = " processing"
    
    # Render the step with minimal design and enhanced state animations
    st.markdown(f"""
    <div class="thinking-step {config['class']}{state_class}{processing_class} agent-step-stream">
        <span class="step-label">{config['label']}</span>
        <span class="step-content">{escaped_content}</span>
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


def render_sophisticated_thinking_indicator(message: str = "Agent is thinking"):
    """
    Render a sophisticated thinking indicator with Jobs/Ive inspired animations.
    
    Args:
        message: Custom message to display during thinking
    """
    st.markdown(f"""
    <div class="thinking-indicator">
        <span>{message}</span>
        <span class="thinking-dots"></span>
    </div>
    """, unsafe_allow_html=True)


def render_agent_steps_with_timeline(steps: list, current_step_index: int = -1):
    """
    Render agent steps with a sophisticated timeline view.
    
    Args:
        steps: List of AgentStep objects
        current_step_index: Index of the currently active step (-1 for none)
    """
    if not steps:
        return
    
    st.markdown('<div class="agent-thinking">', unsafe_allow_html=True)
    
    for i, step in enumerate(steps):
        if i == current_step_index:
            state = "current"
        elif i < current_step_index:
            state = "completed"
        else:
            state = ""
        
        render_agent_step_with_state(step, state)
    
    st.markdown('</div>', unsafe_allow_html=True)
