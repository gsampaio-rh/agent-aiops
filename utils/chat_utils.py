"""
Utility functions for chat functionality.
"""

import time
from typing import List, Dict, Any
import streamlit as st


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_model" not in st.session_state:
        from config.settings import DEFAULT_MODEL
        st.session_state.current_model = DEFAULT_MODEL
    
    if "model_params" not in st.session_state:
        from config.settings import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_TOP_P
        st.session_state.model_params = {
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "top_p": DEFAULT_TOP_P
        }


def add_message(role: str, content: str, metadata: Dict[str, Any] = None):
    """Add a message to the chat history."""
    message = {
        "role": role,
        "content": content,
        "timestamp": time.time(),
        "metadata": metadata or {}
    }
    st.session_state.messages.append(message)


def clear_chat_history():
    """Clear the chat history."""
    st.session_state.messages = []


def format_duration(nanoseconds: int) -> str:
    """Convert nanoseconds to human-readable format."""
    if nanoseconds == 0:
        return "0ms"
    
    # Convert nanoseconds to milliseconds
    ms = nanoseconds / 1_000_000
    
    if ms < 1000:
        return f"{ms:.1f}ms"
    else:
        seconds = ms / 1000
        return f"{seconds:.2f}s"


def calculate_tokens_per_second(token_count: int, duration_ns: int) -> str:
    """Calculate tokens per second from count and duration."""
    if duration_ns == 0 or token_count == 0:
        return "0 tok/s"
    
    seconds = duration_ns / 1_000_000_000  # Convert nanoseconds to seconds
    tokens_per_second = token_count / seconds
    
    if tokens_per_second >= 1000:
        return f"{tokens_per_second/1000:.1f}k tok/s"
    else:
        return f"{tokens_per_second:.1f} tok/s"


def format_metrics(metadata: Dict[str, Any], detailed: bool = True) -> str:
    """Format technical metrics for display with proper terminology."""
    if not metadata:
        return ""
    
    if not detailed:
        # Simple technical metrics display
        simple_metrics = []
        
        if "latency_ms" in metadata:
            simple_metrics.append(f"⏱️ {metadata['latency_ms']}ms")
        
        prompt_tokens = metadata.get('prompt_eval_count', 0)
        output_tokens = metadata.get('eval_count', 0)
        if prompt_tokens > 0 or output_tokens > 0:
            total_tokens = prompt_tokens + output_tokens
            simple_metrics.append(f"🔢 {total_tokens} tokens ({output_tokens} gen)")
        
        if "model" in metadata:
            simple_metrics.append(f"🤖 {metadata['model']}")
        
        return " • ".join(simple_metrics)
    
    # Detailed technical metrics display
    metrics_lines = []
    
    # Model and latency
    basic_info = []
    if "model" in metadata:
        basic_info.append(f"🤖 {metadata['model']}")
    
    if "latency_ms" in metadata:
        basic_info.append(f"⏱️ {metadata['latency_ms']}ms latency")
    
    if basic_info:
        metrics_lines.append(" • ".join(basic_info))
    
    # Token counts and throughput
    prompt_tokens = metadata.get('prompt_eval_count', 0)
    output_tokens = metadata.get('eval_count', 0)
    
    if prompt_tokens > 0 or output_tokens > 0:
        total_tokens = prompt_tokens + output_tokens
        # prompt_eval_count includes full context (conversation history + system prompts + current input)
        token_info = f"🔢 {total_tokens} tokens (context: {prompt_tokens}, generated: {output_tokens})"
        
        # Add token generation throughput
        eval_duration = metadata.get('eval_duration', 0)
        if eval_duration > 0 and output_tokens > 0:
            tokens_per_second = output_tokens / (eval_duration / 1_000_000_000)
            if tokens_per_second >= 1000:
                token_info += f" @ {tokens_per_second/1000:.1f}k tok/s"
            else:
                token_info += f" @ {tokens_per_second:.1f} tok/s"
        
        metrics_lines.append(token_info)
    
    # Duration breakdown
    timing_parts = []
    
    total_duration = metadata.get('total_duration', 0)
    if total_duration > 0:
        timing_parts.append(f"total: {format_duration(total_duration)}")
    
    load_duration = metadata.get('load_duration', 0)
    if load_duration > 0:
        timing_parts.append(f"load: {format_duration(load_duration)}")
    
    prompt_eval_duration = metadata.get('prompt_eval_duration', 0)
    if prompt_eval_duration > 0:
        timing_parts.append(f"prompt_eval: {format_duration(prompt_eval_duration)}")
    
    eval_duration = metadata.get('eval_duration', 0)
    if eval_duration > 0:
        timing_parts.append(f"eval: {format_duration(eval_duration)}")
    
    if timing_parts:
        metrics_lines.append("⏳ " + " • ".join(timing_parts))
    
    # Done reason
    done_reason = metadata.get('done_reason', '')
    if done_reason and done_reason != '':
        metrics_lines.append(f"✅ done_reason: {done_reason}")
    
    return "<br/>".join(metrics_lines)


def format_timestamp(timestamp: float, format_type: str = "time") -> str:
    """Format timestamp for display."""
    if format_type == "time":
        return time.strftime("%H:%M:%S", time.localtime(timestamp))
    elif format_type == "datetime":
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    elif format_type == "date":
        return time.strftime("%Y-%m-%d", time.localtime(timestamp))
    else:
        return time.strftime("%H:%M:%S", time.localtime(timestamp))


def export_chat_history() -> str:
    """Export chat history as text."""
    if not st.session_state.messages:
        return "No chat history to export."
    
    export_text = f"Chat Export - {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += "=" * 50 + "\n\n"
    
    for message in st.session_state.messages:
        role = message["role"].title()
        content = message["content"]
        timestamp = format_timestamp(message["timestamp"], "datetime")
        
        export_text += f"[{timestamp}] {role}: {content}\n"
        
        if message.get("metadata"):
            # For export, create technical metrics
            metadata = message["metadata"]
            metrics_parts = []
            
            if "model" in metadata:
                metrics_parts.append(f"model: {metadata['model']}")
            
            if "latency_ms" in metadata:
                metrics_parts.append(f"latency: {metadata['latency_ms']}ms")
            
            prompt_tokens = metadata.get('prompt_eval_count', 0)
            output_tokens = metadata.get('eval_count', 0)
            if prompt_tokens > 0 or output_tokens > 0:
                total_tokens = prompt_tokens + output_tokens
                metrics_parts.append(f"tokens: {total_tokens} (context: {prompt_tokens}, generated: {output_tokens})")
            
            total_duration = metadata.get('total_duration', 0)
            if total_duration > 0:
                metrics_parts.append(f"total_duration: {format_duration(total_duration)}")
            
            load_duration = metadata.get('load_duration', 0)
            if load_duration > 0:
                metrics_parts.append(f"load_duration: {format_duration(load_duration)}")
            
            prompt_eval_duration = metadata.get('prompt_eval_duration', 0)
            if prompt_eval_duration > 0:
                metrics_parts.append(f"prompt_eval_duration: {format_duration(prompt_eval_duration)}")
            
            eval_duration = metadata.get('eval_duration', 0)
            if eval_duration > 0:
                metrics_parts.append(f"eval_duration: {format_duration(eval_duration)}")
                
                # Add throughput calculation
                if output_tokens > 0:
                    tokens_per_second = output_tokens / (eval_duration / 1_000_000_000)
                    metrics_parts.append(f"throughput: {tokens_per_second:.1f} tok/s")
            
            done_reason = metadata.get('done_reason', '')
            if done_reason:
                metrics_parts.append(f"done_reason: {done_reason}")
            
            if metrics_parts:
                export_text += f"   Metrics: {' | '.join(metrics_parts)}\n"
        
        export_text += "\n"
    
    return export_text
