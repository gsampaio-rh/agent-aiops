"""
Sidebar Components for the Streamlit Ollama Chatbot Application.

This module contains all sidebar-related UI components including:
- Model selection and parameters
- Agent mode toggles
- Session management controls
- Chat statistics display

All components follow Jobs/Ive design principles with minimalist aesthetics.
"""

import streamlit as st
import time
from typing import Dict, Any

from config.settings import (
    APP_TITLE, APP_ICON, DEFAULT_MODEL, MODEL_PARAMS, SHOW_DETAILED_METRICS
)
from services.ollama_service import OllamaService
from services.agent_factory import create_agent, AgentFactory
from utils.chat_utils import (
    clear_chat_history, export_chat_history
)
import tempfile
import os
from pathlib import Path


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
                st.session_state.agent = create_agent(model=current_model)
            
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
            
            # Chat statistics and memory info
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
                
                # Memory status indicator
                st.subheader("🧠 Memory Status")
                from config.constants import AGENT_CONFIG
                max_memory = AGENT_CONFIG.get("max_conversation_history", 20)
                memory_usage = min(total_messages, max_memory)
                
                # Memory indicator with visual bar
                memory_percentage = (memory_usage / max_memory) * 100
                st.metric("Memory Usage", f"{memory_usage}/{max_memory}")
                st.progress(memory_percentage / 100, text=f"{memory_percentage:.0f}% of memory used")
                
                if st.session_state.agent_mode and total_messages > 0:
                    st.success("🔄 Conversation memory active")
                    st.caption("Agent remembers your conversation history")
                elif total_messages > 0:
                    st.info("💬 Chat history available")
                    st.caption("Switch to Agent mode for memory features")
            
            # RAG status indicator
            if st.session_state.agent_mode:
                st.subheader("📚 Knowledge Base")
                try:
                    # Get RAG tool status if available
                    if st.session_state.agent and hasattr(st.session_state.agent, 'tools'):
                        rag_tool = st.session_state.agent.tools.get('rag_search')
                        if rag_tool:
                            status = rag_tool.get_status()
                            
                            if status.get("initialized", False):
                                docs_count = status.get("documents_indexed", 0)
                                st.success(f"✅ {docs_count} documents indexed")
                                st.caption("Agent can search local knowledge base")
                                
                                # Show documents path and list
                                docs_path = status.get("documents_path", "./documents")
                                st.caption(f"📁 Path: {docs_path}")
                                
                                # Show document list in expander
                                with st.expander("📄 View Documents", expanded=False):
                                    _display_document_list(docs_path)
                                
                                # Document management section
                                st.markdown("---")
                                st.caption("📁 Document Management")
                                
                                # File upload interface
                                uploaded_files = st.file_uploader(
                                    "Upload documents",
                                    accept_multiple_files=True,
                                    type=['md', 'txt', 'pdf'],
                                    help="Upload .md, .txt, or .pdf files to add to your knowledge base (Max 10MB per file)",
                                    key="document_uploader"
                                )
                                
                                if uploaded_files:
                                    # Show upload preview
                                    with st.expander("📋 Upload Preview", expanded=True):
                                        total_size = sum(f.size for f in uploaded_files)
                                        st.write(f"**{len(uploaded_files)} files selected** (Total: {_format_file_size(total_size)})")
                                        
                                        for f in uploaded_files:
                                            col1, col2 = st.columns([3, 1])
                                            with col1:
                                                ext = Path(f.name).suffix.lower()
                                                icon = {"md": "📝", "txt": "📄", "pdf": "📑"}.get(ext[1:], "📄")
                                                st.write(f"{icon} {f.name}")
                                            with col2:
                                                st.caption(_format_file_size(f.size))
                                    
                                    # Upload button with confirmation
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button("📤 Add to Knowledge Base", help="Process and add uploaded files", type="primary", use_container_width=True):
                                            with st.spinner("Processing uploaded documents..."):
                                                upload_result = _handle_document_upload(uploaded_files, rag_tool)
                                                if upload_result.get("success"):
                                                    st.success(f"✅ Added {upload_result.get('files_processed', 0)} documents!")
                                                    if upload_result.get("files_failed", 0) > 0:
                                                        st.warning(f"⚠️ {upload_result.get('files_failed', 0)} files failed")
                                                    st.rerun()
                                                else:
                                                    st.error(f"❌ Upload failed: {upload_result.get('error', 'Unknown error')}")
                                    
                                    with col2:
                                        if st.button("🗑️ Clear Selection", help="Clear selected files", use_container_width=True):
                                            st.rerun()
                                
                                # Refresh button
                                if st.button("🔄 Refresh Index", help="Re-scan documents folder"):
                                    with st.spinner("Refreshing document index..."):
                                        refresh_result = rag_tool.refresh_index()
                                        if refresh_result.get("success"):
                                            st.success(f"✅ Index refreshed! {refresh_result.get('documents_indexed', 0)} documents")
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Refresh failed: {refresh_result.get('error', 'Unknown error')}")
                            else:
                                st.warning("⚠️ Knowledge base not initialized")
                                st.caption("Check documents folder and dependencies")
                        else:
                            st.info("📚 Knowledge base available")
                            st.caption("RAG tool will be loaded when needed")
                    else:
                        st.info("📚 Knowledge base available")
                        st.caption("Switch to Agent mode to use RAG search")
                        
                except Exception as e:
                    st.error("❌ RAG status check failed")
                    st.caption(f"Error: {str(e)}")
            else:
                # Show RAG info in normal mode
                st.subheader("📚 Knowledge Base")
                st.info("💡 Switch to Agent mode")
                st.caption("Enable Agent mode to search local documents")
            
            return {
                "model": current_model,
                "params": params,
                "ollama_service": ollama_service,
                "agent_mode": st.session_state.agent_mode
            }
    
    except Exception as e:
        st.error(f"Sidebar error: {e}")
        return {}


def _handle_document_upload(uploaded_files, rag_tool):
    """
    Handle document upload and processing.
    
    Args:
        uploaded_files: List of Streamlit uploaded file objects
        rag_tool: RAG tool instance
        
    Returns:
        Dict with success status and processing results
    """
    try:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        
        if not uploaded_files:
            return {"success": False, "error": "No files uploaded"}
        
        # Get documents path from RAG tool
        docs_path = Path(rag_tool.config.get("documents_path", "./documents"))
        docs_path.mkdir(parents=True, exist_ok=True)
        
        files_processed = 0
        files_failed = []
        
        for uploaded_file in uploaded_files:
            try:
                # Validate file type
                file_extension = Path(uploaded_file.name).suffix.lower()
                supported_extensions = rag_tool.config.get("supported_extensions", [".md", ".txt", ".pdf"])
                
                if file_extension not in supported_extensions:
                    files_failed.append(f"{uploaded_file.name}: Unsupported file type")
                    continue
                
                # Check file size (limit to 10MB)
                max_size = 10 * 1024 * 1024  # 10MB
                if uploaded_file.size > max_size:
                    files_failed.append(f"{uploaded_file.name}: File too large (max 10MB)")
                    continue
                
                # Save file to documents folder
                file_path = docs_path / uploaded_file.name
                
                # Handle file name conflicts
                counter = 1
                original_stem = file_path.stem
                while file_path.exists():
                    file_path = docs_path / f"{original_stem}_{counter}{file_extension}"
                    counter += 1
                
                # Write file content
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                logger.info(f"Uploaded document saved: {file_path}")
                files_processed += 1
                
            except Exception as e:
                logger.error(f"Failed to process uploaded file {uploaded_file.name}: {str(e)}")
                files_failed.append(f"{uploaded_file.name}: {str(e)}")
        
        # Refresh the RAG index to include new documents
        if files_processed > 0:
            try:
                refresh_result = rag_tool.refresh_index()
                if not refresh_result.get("success"):
                    logger.warning("Index refresh failed after upload", error=refresh_result.get("error"))
            except Exception as e:
                logger.error("Failed to refresh index after upload", error=str(e))
        
        # Prepare result
        result = {
            "success": files_processed > 0,
            "files_processed": files_processed,
            "files_failed": len(files_failed),
            "total_files": len(uploaded_files)
        }
        
        if files_failed:
            result["failed_files"] = files_failed
            result["error"] = f"Some files failed to process: {'; '.join(files_failed[:3])}"
            if len(files_failed) > 3:
                result["error"] += f" and {len(files_failed) - 3} more..."
        
        logger.info(f"Document upload completed", **result)
        return result
        
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        return {
            "success": False,
            "error": f"Upload processing failed: {str(e)}",
            "files_processed": 0
        }


def _display_document_list(docs_path):
    """
    Display list of documents in the knowledge base.
    
    Args:
        docs_path: Path to documents folder
    """
    try:
        docs_dir = Path(docs_path)
        if not docs_dir.exists():
            st.info("📁 Documents folder doesn't exist yet")
            return
        
        # Get all supported document files
        supported_extensions = [".md", ".txt", ".pdf"]
        all_files = []
        
        for ext in supported_extensions:
            files = list(docs_dir.glob(f"*{ext}"))
            all_files.extend(files)
        
        if not all_files:
            st.info("📄 No documents found")
            st.caption("Upload documents using the interface below")
            return
        
        # Sort files by modification time (newest first)
        all_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Header with bulk actions
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**Found {len(all_files)} documents:**")
        with col2:
            if len(all_files) > 1:
                if st.button("🗑️ Clear All", help="Delete all documents", key="delete_all_docs"):
                    if st.session_state.get("confirm_delete_all", False):
                        # Actually delete
                        deleted_count = 0
                        for file_path in all_files:
                            if _delete_document(file_path):
                                deleted_count += 1
                        st.success(f"Deleted {deleted_count} documents")
                        st.session_state.confirm_delete_all = False
                        st.rerun()
                    else:
                        # Ask for confirmation
                        st.session_state.confirm_delete_all = True
                        st.warning("Click again to confirm deletion of all documents")
        
        # Show files
        for file_path in all_files:
            # Get file info
            file_size = file_path.stat().st_size
            file_size_str = _format_file_size(file_size)
            
            # Get file type icon
            ext = file_path.suffix.lower()
            icon = {"md": "📝", "txt": "📄", "pdf": "📑"}.get(ext[1:], "📄")
            
            # Display file info
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{icon} **{file_path.name}**")
                st.caption(f"Size: {file_size_str}")
            
            with col2:
                # Add delete button
                if st.button("🗑️", key=f"delete_{file_path.name}", help=f"Delete {file_path.name}"):
                    if _delete_document(file_path):
                        st.success(f"Deleted {file_path.name}")
                        st.rerun()
                    else:
                        st.error(f"Failed to delete {file_path.name}")
        
    except Exception as e:
        st.error(f"Error displaying documents: {str(e)}")


def _format_file_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _delete_document(file_path):
    """
    Delete a document file.
    
    Args:
        file_path: Path to file to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        
        file_path.unlink()
        logger.info(f"Deleted document: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete document {file_path}: {str(e)}")
        return False
