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
