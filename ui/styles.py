"""
CSS Styles for the Streamlit Ollama Chatbot Application.

Inspired by Dieter Rams' design principles and Apple's user experience.
All styles follow Jobs/Ive minimalist aesthetic with careful attention to typography,
spacing, and visual hierarchy.
"""

def get_app_styles() -> str:
    """
    Get the complete CSS styles for the application.
    
    Returns:
        str: Complete CSS styles as a string ready for st.markdown()
    """
    return """
<style>
    /* Main container styling - Premium feel */
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
        background: radial-gradient(ellipse at top, rgba(255, 255, 255, 0.1) 0%, transparent 50%);
    }
    
    /* Chat area background */
    .main {
        background: linear-gradient(180deg, #FAFAFA 0%, #F5F5F7 100%);
    }
    
    /* Floating input container - seamless integration */
    .stChatFloatingInputContainer {
        background: transparent;
    }
    
    /* Make the floating container blend better */
    .stChatFloatingInputContainer::before {
        content: '';
        position: absolute;
        top: -20px;
        left: 0;
        right: 0;
        height: 20px;
        pointer-events: none;
    }
    
    /* Chat message styling - Jobs/Ive inspired */
    .chat-message {
        padding: 1.25rem 1.75rem;
        margin: 1rem 0;
        border-radius: 22px;
        max-width: 75%;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        line-height: 1.47;
        font-size: 1.05rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        backdrop-filter: blur(20px);
        transition: all 0.2s ease;
    }
    
    .chat-message:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .user-message {
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
        color: white;
        margin-left: auto;
        text-align: right;
        margin-bottom: 1.5rem;
    }
    
    .assistant-message {
        background: rgba(248, 249, 250, 0.9);
        color: #1D1D1F;
        margin-right: auto;
        border: 1px solid rgba(0, 0, 0, 0.06);
    }
    
    /* Metrics styling - Minimalist approach */
    .metrics {
        font-size: 0.7rem;
        color: #98989D;
        margin-top: 0.75rem;
        margin-bottom: 0.5rem;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
        line-height: 1.2;
        max-width: 75%;
        font-weight: 400;
        letter-spacing: 0.2px;
        opacity: 0.8;
        transition: opacity 0.2s ease;
    }
    
    .metrics:hover {
        opacity: 1;
    }
    
    /* Timestamp styling - More subtle */
    .timestamp {
        font-size: 0.65rem;
        color: #98989D;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
        opacity: 0.6;
        font-weight: 400;
        letter-spacing: 0.3px;
        transition: opacity 0.2s ease;
    }
    
    .timestamp:hover {
        opacity: 0.9;
    }
    
    /* Agent thinking - Ultra minimalist Jobs/Ive approach */
    .agent-thinking {
        margin: 1.5rem 0;
        padding: 0;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        max-width: 100%;
    }
    
    .thinking-step {
        margin: 0;
        padding: 0.4rem 0;
        border: none;
        background: transparent;
        opacity: 0.6;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        font-size: 0.9rem;
        line-height: 1.4;
        display: flex;
        align-items: baseline;
    }
    
    .thinking-step.current {
        opacity: 1;
        transform: translateX(2px);
    }
    
    .thinking-step.completed {
        opacity: 0.5;
    }
    
    .step-label {
        display: inline-block;
        font-weight: 700;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-right: 1rem;
        min-width: 90px;
        flex-shrink: 0;
    }
    
    .step-thought .step-label { color: #86868B; }
    .step-tool-selection .step-label { color: #007AFF; }
    .step-tool-use .step-label { color: #FF9500; }
    .step-tool-result .step-label { color: #34C759; }
    .step-final-answer .step-label { color: #1D1D1F; font-weight: 800; }
    
    .step-content {
        color: #1D1D1F;
        font-weight: 400;
        flex: 1;
        word-wrap: break-word;
    }
    
    .step-metadata {
        font-size: 0.65rem;
        color: #86868B;
        margin-left: 106px;
        margin-top: 0.2rem;
        font-family: 'SF Mono', Monaco, monospace;
        opacity: 0.8;
    }
    
    /* Thinking indicator - minimal animation */
    .thinking-indicator {
        display: flex;
        align-items: center;
        font-size: 0.85rem;
        color: #86868B;
        margin: 0.8rem 0;
        font-weight: 500;
    }
    
    .thinking-dots {
        margin-left: 0.5rem;
        width: 20px;
    }
    
    .thinking-dots::after {
        content: '';
        animation: thinking 2s ease-in-out infinite;
    }
    
    @keyframes thinking {
        0% { content: '.'; }
        33% { content: '..'; }
        66% { content: '...'; }
        100% { content: '.'; }
    }
    
    /* Streaming effect for agent steps */
    .agent-step-stream {
        animation: streamIn 0.3s ease-out;
    }
    
    @keyframes streamIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Agent steps visibility message */
    .steps-hidden-message {
        text-align: center;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #86868B;
        font-size: 0.85rem;
        font-style: italic;
        border-radius: 8px;
        background: rgba(0, 0, 0, 0.02);
    }
    
    /* Sidebar styling - Jobs/Ive aesthetic */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #FBFBFD 0%, #F5F5F7 100%);
        border-right: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* Clean sidebar sections */
    .sidebar h3 {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1D1D1F;
        margin-bottom: 0.75rem;
        margin-top: 1.5rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .sidebar .stSelectbox label,
    .sidebar .stSlider label,
    .sidebar .stCheckbox label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #424245;
    }
    
    /* Input styling - More refined */
    .stTextInput > div > div > input,
    .stChatInput > div > div > input {
        border-radius: 24px;
        border: 1.5px solid rgba(0, 0, 0, 0.08);
        padding: 1rem 1.5rem;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
        font-size: 1rem;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stChatInput > div > div > input:focus {
        border-color: #007AFF;
        box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.1);
        outline: none;
    }
    
    /* Button styling - More premium */
    .stButton > button {
        border-radius: 14px;
        border: none;
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.25rem;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
        font-size: 0.9rem;
        letter-spacing: 0.2px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(0, 122, 255, 0.15);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0056CC 0%, #4339B8 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(0, 122, 255, 0.25);
        border: none;
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(0, 122, 255, 0.15);
    }
    
    /* Tool execution cards - following Jobs/Ive design principles */
    .tool-permission-card,
    .tool-execution-success,
    .tool-execution-failed,
    .tool-execution-progress {
        margin: 1.5rem 0;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(0, 0, 0, 0.06);
        background: rgba(248, 249, 250, 0.95);
        backdrop-filter: blur(20px);
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
    }
    
    .tool-permission-card:hover,
    .tool-execution-success:hover,
    .tool-execution-failed:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    }
    
    .permission-header,
    .execution-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    }
    
    .permission-icon,
    .execution-icon {
        font-size: 1.25rem;
        margin-right: 0.75rem;
    }
    
    .permission-title,
    .execution-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1D1D1F;
        letter-spacing: 0.2px;
    }
    
    .permission-content {
        margin-bottom: 1rem;
    }
    
    .tool-info {
        font-size: 1rem;
        font-weight: 500;
        color: #1D1D1F;
        margin-bottom: 0.5rem;
    }
    
    .tool-description {
        font-size: 0.9rem;
        color: #424245;
        margin-bottom: 1rem;
        line-height: 1.4;
    }
    
    .tool-query {
        font-size: 0.85rem;
        color: #86868B;
        background: rgba(0, 0, 0, 0.02);
        padding: 0.75rem;
        border-radius: 8px;
        font-family: 'SF Mono', Monaco, monospace;
        border: 1px solid rgba(0, 0, 0, 0.04);
    }
    
    .execution-metadata {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .metadata-item {
        font-size: 0.8rem;
        color: #424245;
        background: rgba(0, 0, 0, 0.02);
        padding: 0.4rem 0.75rem;
        border-radius: 20px;
        font-weight: 500;
        border: 1px solid rgba(0, 0, 0, 0.04);
    }
    
    .error-content {
        margin-top: 1rem;
    }
    
    .error-tool {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1D1D1F;
        margin-bottom: 0.5rem;
    }
    
    .error-message {
        font-size: 0.85rem;
        color: #FF3B30;
        background: rgba(255, 59, 48, 0.05);
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 3px solid #FF3B30;
        font-family: 'SF Mono', Monaco, monospace;
    }
    
    .tool-results-container {
        margin-top: 1rem;
        max-height: 400px;
        overflow-y: auto;
        border-radius: 8px;
        background: rgba(0, 0, 0, 0.02);
        border: 1px solid rgba(0, 0, 0, 0.04);
    }
    
    .tool-results {
        font-family: 'SF Mono', Monaco, monospace;
        font-size: 0.8rem;
        line-height: 1.5;
        color: #1D1D1F;
        white-space: pre-wrap;
        word-wrap: break-word;
        padding: 1rem;
        margin: 0;
        background: transparent;
        border: none;
    }
    
    /* Tool execution specific styling */
    .tool-execution-success {
        border-left: 4px solid #34C759;
        background: rgba(52, 199, 89, 0.02);
    }
    
    .tool-execution-failed {
        border-left: 4px solid #FF3B30;
        background: rgba(255, 59, 48, 0.02);
    }
    
    .tool-execution-progress {
        border-left: 4px solid #007AFF;
        background: rgba(0, 122, 255, 0.02);
    }
    
    /* Tool approval interface */
    .tool-approval-interface {
        margin: 1.5rem 0;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.06);
        background: rgba(248, 249, 250, 0.95);
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
    }
    
    .approval-header {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .approval-icon {
        font-size: 1.1rem;
        margin-right: 0.5rem;
    }
    
    .approval-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1D1D1F;
    }
    
    /* Results approval section */
    .results-approval-section {
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(0, 0, 0, 0.06);
    }
    
    .approval-question {
        margin-bottom: 1rem;
        font-size: 0.95rem;
        color: #1D1D1F;
        text-align: center;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    # header {visibility: hidden;}
</style>
"""


def apply_app_styles():
    """
    Apply the application styles using Streamlit's markdown function.
    
    This function should be called once during app initialization to inject
    all CSS styles into the Streamlit application.
    """
    import streamlit as st
    st.markdown(get_app_styles(), unsafe_allow_html=True)
