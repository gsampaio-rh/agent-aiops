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
        position: relative;
    }
    
    .thinking-step.current {
        opacity: 1;
        transform: translateX(2px);
        animation: thinkingBreath 2.5s ease-in-out infinite;
    }
    
    /* Fallback for older browsers or if animations cause issues */
    @media (prefers-reduced-motion: reduce) {
        .thinking-step.current {
            animation: none;
            opacity: 1;
        }
        
        .thinking-step.current::before {
            animation: none;
        }
        
        .thinking-indicator {
            animation: none;
        }
        
        .thinking-indicator::before {
            animation: none;
        }
    }
    
    .thinking-step.current::before {
        content: '';
        position: absolute;
        left: -8px;
        top: 50%;
        transform: translateY(-50%);
        width: 3px;
        height: 3px;
        background: #007AFF;
        border-radius: 50%;
        animation: thinkingPulse 2s ease-in-out infinite;
        z-index: 1;
    }
    
    .thinking-step.completed {
        opacity: 0.5;
    }
    
    .thinking-step.completed::before {
        content: '';
        position: absolute;
        left: -8px;
        top: 50%;
        transform: translateY(-50%);
        width: 3px;
        height: 3px;
        background: #34C759;
        border-radius: 50%;
        opacity: 0.6;
        z-index: 1;
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
    .step-error .step-label { color: #FF3B30; }
    
    .step-content {
        color: #1D1D1F;
        font-weight: 400;
        flex: 1;
        word-wrap: break-word;
    }
    
    .step-processing-icon {
        margin-left: 0.5rem;
        font-size: 0.8rem;
        opacity: 0.7;
        animation: processingFloat 2s ease-in-out infinite;
    }
    
    /* Simplified processing state styling */
    .thinking-step.processing {
        position: relative;
    }
    
    .thinking-step.processing::after {
        content: '⚡';
        position: absolute;
        right: 0;
        top: 50%;
        transform: translateY(-50%);
        font-size: 0.8rem;
        opacity: 0.6;
        animation: processingFloat 2s ease-in-out infinite;
    }
    
    .step-metadata {
        font-size: 0.65rem;
        color: #86868B;
        margin-left: 106px;
        margin-top: 0.2rem;
        font-family: 'SF Mono', Monaco, monospace;
        opacity: 0.8;
    }
    
    /* Thinking indicator - Jobs/Ive minimal elegance */
    .thinking-indicator {
        display: flex;
        align-items: center;
        font-size: 0.85rem;
        color: #86868B;
        margin: 0.8rem 0;
        font-weight: 500;
        padding: 0.75rem 0;
        animation: indicatorBreath 3s ease-in-out infinite;
    }
    
    .thinking-indicator::before {
        content: '';
        width: 4px;
        height: 4px;
        background: #007AFF;
        border-radius: 50%;
        margin-right: 0.75rem;
        animation: indicatorPulse 2s ease-in-out infinite;
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
    
    /* Jobs/Ive inspired breathing animation for active thinking */
    @keyframes thinkingBreath {
        0%, 100% { 
            opacity: 1;
            transform: translateX(2px);
        }
        50% { 
            opacity: 0.85;
            transform: translateX(2px);
        }
    }
    
    /* Subtle pulsing indicator for current step */
    @keyframes thinkingPulse {
        0%, 100% { 
            opacity: 1;
            transform: translateY(-50%) scale(1);
        }
        50% { 
            opacity: 0.4;
            transform: translateY(-50%) scale(1.2);
        }
    }
    
    /* Breathing animation for the main thinking indicator */
    @keyframes indicatorBreath {
        0%, 100% { 
            opacity: 1;
            transform: scale(1);
        }
        50% { 
            opacity: 0.7;
            transform: scale(1.001);
        }
    }
    
    /* Pulse animation for indicator dot */
    @keyframes indicatorPulse {
        0%, 100% { 
            opacity: 1;
            transform: scale(1);
            background: #007AFF;
        }
        50% { 
            opacity: 0.3;
            transform: scale(1.5);
            background: #5AC8FA;
        }
    }
    
    /* Floating animation for processing icons */
    @keyframes processingFloat {
        0%, 100% { 
            opacity: 0.7;
            transform: translateY(0px);
        }
        50% { 
            opacity: 0.9;
            transform: translateY(-2px);
        }
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
    
    /* Input styling - Consistent with permission card */
    .stTextInput > div > div > input,
    .stChatInput > div > div > input {
        border-radius: 8px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        padding: 0.75rem 1rem;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif !important;
        font-size: 0.9rem;
        font-weight: 500;
        background: rgba(255, 255, 255, 0.95);
        transition: all 0.2s ease;
        letter-spacing: 0.1px;
    }
    
    .tool-permission-card .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        padding: 0.75rem 1rem;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif !important;
        font-size: 0.9rem;
        font-weight: 500;
        background: rgba(255, 255, 255, 0.95);
        color: #1D1D1F;
    }
    
    .stTextInput > div > div > input:focus,
    .stChatInput > div > div > input:focus,
    .tool-permission-card .stTextInput > div > div > input:focus {
        border-color: #1D1D1F;
        box-shadow: 0 0 0 2px rgba(29, 29, 31, 0.1);
        outline: none;
        background: rgba(255, 255, 255, 1);
    }
    
    /* Button styling - Jobs/Ive hierarchy */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        background: rgba(248, 249, 250, 0.95);
        color: #1D1D1F;
        font-weight: 500;
        padding: 0.75rem 1.25rem;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
        font-size: 0.9rem;
        letter-spacing: 0.1px;
        transition: all 0.2s ease;
        box-shadow: none;
    }
    
    /* Primary button - the most important action */
    .stButton > button[kind="primary"] {
        background: #1D1D1F;
        color: white;
        border: 1px solid #1D1D1F;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: rgba(0, 0, 0, 0.05);
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.15);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: #000;
        color: white;
        border: 1px solid #000;
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Tool permission cards - Distinct with subtle warmth */
    .tool-permission-card {
        margin: 2rem 0;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(251, 251, 253, 0.95) 0%, rgba(248, 249, 251, 0.95) 100%);
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        border: none;
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.08);
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
        backdrop-filter: blur(10px);
    }
    
    /* Tool execution cards - subtle distinction */
    .tool-execution-success,
    .tool-execution-failed,
    .tool-execution-progress {
        margin: 1.5rem 0;
        padding: 1.25rem;
        border-radius: 12px;
        border: none;
        background: rgba(248, 249, 250, 0.6);
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        transition: background 0.3s ease;
    }
    
    .tool-execution-success:hover,
    .tool-execution-failed:hover {
        background: rgba(248, 249, 250, 0.8);
    }
    
    /* Ultra-minimal permission header */
    .permission-header {
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .execution-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(0, 0, 0, 0.04);
    }
    
    .permission-icon {
        display: none; /* Remove visual clutter */
    }
    
    .execution-icon {
        font-size: 1rem;
        margin-right: 0.5rem;
        opacity: 0.6;
    }
    
    /* Clear hierarchy - consistent typography */
    .permission-title {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif !important;
        font-size: 1.3rem;
        font-weight: 700;
        color: #1D1D1F;
        letter-spacing: -0.02em;
        margin-bottom: 0.75rem;
        text-align: center;
    }
    
    .execution-title {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif !important;
        font-size: 0.95rem;
        font-weight: 600;
        color: #1D1D1F;
        letter-spacing: 0.1px;
    }
    
    /* Clean content hierarchy */
    .permission-content {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .tool-info {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif !important;
        font-size: 1rem;
        font-weight: 500;
        color: #1D1D1F;
        margin-bottom: 0.75rem;
        line-height: 1.4;
        max-width: 500px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .tool-description {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif !important;
        font-size: 0.9rem;
        color: #86868B;
        margin-bottom: 1.5rem;
        line-height: 1.4;
        max-width: 400px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* De-emphasize the query - it's not the hero */
    .tool-query {
        font-family: 'SF Mono', Monaco, 'Cascadia Code', Consolas, monospace !important;
        font-size: 0.8rem;
        color: #86868B;
        background: rgba(0, 0, 0, 0.02);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border: 1px solid rgba(0, 0, 0, 0.08);
        margin: 0 auto 1rem auto;
        max-width: 350px;
        text-align: left;
        opacity: 0.9;
        line-height: 1.3;
        font-weight: 500;
    }
    
    /* Tool permission button hierarchy - consistent styling */
    .stButton > button {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.25rem !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.1px !important;
        transition: all 0.2s ease !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        text-transform: none !important;
    }
    
    /* Primary Allow button - harmonized with user message blue */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%) !important;
        color: white !important;
        border: 1px solid #007AFF !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0056CC 0%, #4339B8 100%) !important;
        color: white !important;
        border: 1px solid #0056CC !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(0, 122, 255, 0.25) !important;
    }
    
    /* Cancel button - least prominent */
    .stButton > button:not([kind="primary"]):first-child {
        background: transparent !important;
        color: #86868B !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
    }
    
    .stButton > button:not([kind="primary"]):first-child:hover {
        background: rgba(0, 0, 0, 0.02) !important;
        color: #1D1D1F !important;
        border: 1px solid rgba(0, 0, 0, 0.15) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Allow Modified button - secondary */
    .stButton > button:not([kind="primary"]):last-child {
        background: rgba(248, 249, 250, 0.95) !important;
        color: #1D1D1F !important;
        border: 1px solid rgba(0, 0, 0, 0.15) !important;
    }
    
    .stButton > button:not([kind="primary"]):last-child:hover {
        background: rgba(235, 236, 238, 0.95) !important;
        color: #1D1D1F !important;
        border: 1px solid rgba(0, 0, 0, 0.2) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
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
