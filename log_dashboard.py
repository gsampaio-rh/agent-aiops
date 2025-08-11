"""
Log Dashboard for Agent-AIOps

Provides real-time log monitoring and metrics visualization within the Streamlit interface.
"""

import streamlit as st
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque

from utils.enhanced_log_analyzer import LogEntry, LogStats, EnhancedLogAnalyzer
from utils.system_metrics import get_system_monitor


class LogDashboard:
    """Real-time log dashboard for Streamlit."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.analyzer = EnhancedLogAnalyzer(log_dir)
        self.max_recent_logs = 50
        
    def render_dashboard(self):
        """Render the complete log dashboard."""
        st.header("📊 Enhanced Logging Dashboard")
        
        # Refresh button
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        with col2:
            auto_refresh = st.checkbox("Auto-refresh", value=False)
        
        if auto_refresh:
            # Auto-refresh every 5 seconds
            time.sleep(5)
            st.rerun()
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview", 
            "🤖 LLM Communication", 
            "🔧 Tool Execution",
            "⚠️ Error Analysis",
            "💻 System Metrics"
        ])
        
        with tab1:
            self._render_overview_tab()
        
        with tab2:
            self._render_llm_tab()
        
        with tab3:
            self._render_tools_tab()
        
        with tab4:
            self._render_errors_tab()
        
        with tab5:
            self._render_system_metrics_tab()
    
    def _render_overview_tab(self):
        """Render the overview tab."""
        st.subheader("📊 Logging Overview")
        
        # Get recent statistics
        stats = self.analyzer.analyze_logs()
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Log Entries", f"{stats.total_entries:,}")
        
        with col2:
            st.metric("Active Requests", len(stats.correlation_ids))
        
        with col3:
            error_count = stats.level_counts.get('ERROR', 0)
            st.metric("Errors", error_count)
        
        with col4:
            llm_requests = len(stats.llm_requests)
            st.metric("LLM Requests", llm_requests)
        
        # Log level distribution
        if stats.level_counts:
            st.subheader("Log Level Distribution")
            level_data = dict(stats.level_counts)
            st.bar_chart(level_data)
        
        # Recent log entries
        st.subheader("Recent Log Entries")
        recent_logs = self._get_recent_logs()
        
        if recent_logs:
            for entry in recent_logs[-10:]:  # Show last 10
                level_color = {
                    'ERROR': '🔴',
                    'WARNING': '🟡', 
                    'INFO': '🔵',
                    'DEBUG': '⚪'
                }.get(entry.level, '⚪')
                
                timestamp = datetime.now().strftime("%H:%M:%S") if not entry.timestamp else entry.timestamp
                st.text(f"{level_color} [{timestamp}] {entry.logger}: {entry.message[:100]}")
        else:
            st.info("No recent log entries found")
    
    def _render_llm_tab(self):
        """Render the LLM communication tab."""
        st.subheader("🤖 LLM Communication Tracking")
        
        # Check for LLM-specific log files
        llm_request_logs = self._read_specialized_log("llm-requests.log")
        llm_response_logs = self._read_specialized_log("llm-responses.log")
        conversation_logs = self._read_specialized_log("llm-conversations.log")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Requests", len(llm_request_logs))
        
        with col2:
            st.metric("Responses", len(llm_response_logs))
        
        with col3:
            st.metric("Conversations", len(conversation_logs))
        
        # Recent LLM requests
        if llm_request_logs:
            st.subheader("Recent LLM Requests")
            for log_entry in llm_request_logs[-5:]:
                with st.expander(f"Request - {log_entry.get('model', 'unknown')} - {log_entry.get('message_count', 0)} messages"):
                    st.json(log_entry)
        
        # Token usage over time (if available)
        if llm_response_logs:
            st.subheader("Token Usage")
            token_data = []
            for log_entry in llm_response_logs:
                if 'tokens_generated' in log_entry:
                    token_data.append(log_entry['tokens_generated'])
            
            if token_data:
                st.line_chart(token_data[-20:])  # Last 20 responses
        
        # Performance metrics
        if llm_response_logs:
            st.subheader("Performance Metrics")
            durations = [log.get('duration_ms', 0) for log in llm_response_logs if log.get('duration_ms')]
            tokens_per_sec = [log.get('tokens_per_second', 0) for log in llm_response_logs if log.get('tokens_per_second')]
            
            if durations:
                col1, col2 = st.columns(2)
                with col1:
                    avg_duration = sum(durations) / len(durations)
                    st.metric("Avg Response Time", f"{avg_duration:.1f}ms")
                
                with col2:
                    if tokens_per_sec:
                        avg_tokens_per_sec = sum(tokens_per_sec) / len(tokens_per_sec)
                        st.metric("Avg Tokens/sec", f"{avg_tokens_per_sec:.1f}")
    
    def _render_tools_tab(self):
        """Render the tool execution tab."""
        st.subheader("🔧 Tool Execution Monitoring")
        
        tool_logs = self._read_specialized_log("tool-execution.log")
        
        if tool_logs:
            # Tool usage statistics
            tool_stats = defaultdict(lambda: {'count': 0, 'success': 0, 'total_duration': 0})
            
            for log_entry in tool_logs:
                tool_name = log_entry.get('tool_name', 'unknown')
                tool_stats[tool_name]['count'] += 1
                if log_entry.get('success', False):
                    tool_stats[tool_name]['success'] += 1
                tool_stats[tool_name]['total_duration'] += log_entry.get('duration_ms', 0)
            
            st.subheader("Tool Usage Summary")
            for tool_name, stats in tool_stats.items():
                success_rate = (stats['success'] / stats['count'] * 100) if stats['count'] > 0 else 0
                avg_duration = stats['total_duration'] / stats['count'] if stats['count'] > 0 else 0
                
                with st.expander(f"{tool_name} - {stats['count']} executions"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Executions", stats['count'])
                    with col2:
                        st.metric("Success Rate", f"{success_rate:.1f}%")
                    with col3:
                        st.metric("Avg Duration", f"{avg_duration:.1f}ms")
            
            # Recent tool executions
            st.subheader("Recent Tool Executions")
            for log_entry in tool_logs[-10:]:
                status = "✅" if log_entry.get('success', False) else "❌"
                tool_name = log_entry.get('tool_name', 'unknown')
                duration = log_entry.get('duration_ms', 0)
                st.text(f"{status} {tool_name} - {duration}ms")
        else:
            st.info("No tool execution logs found")
    
    def _render_errors_tab(self):
        """Render the error analysis tab."""
        st.subheader("⚠️ Error Analysis")
        
        error_logs = self._read_specialized_log("errors.log")
        
        if error_logs:
            # Error frequency by type
            error_types = defaultdict(int)
            recent_errors = []
            
            for log_entry in error_logs:
                error_type = log_entry.get('error_type', 'Unknown')
                error_types[error_type] += 1
                recent_errors.append(log_entry)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Error Types")
                if error_types:
                    st.bar_chart(dict(error_types))
            
            with col2:
                st.subheader("Error Frequency")
                st.metric("Total Errors", len(error_logs))
                st.metric("Unique Error Types", len(error_types))
            
            # Recent errors with details
            st.subheader("Recent Errors")
            for error in recent_errors[-5:]:
                with st.expander(f"❌ {error.get('error_type', 'Unknown')} - {error.get('operation', 'Unknown operation')}"):
                    st.text(f"Error: {error.get('error_message', 'No message')}")
                    st.text(f"Operation: {error.get('operation', 'Unknown')}")
                    if error.get('stack_trace'):
                        st.code(error['stack_trace'][:500] + "..." if len(error.get('stack_trace', '')) > 500 else error['stack_trace'])
        else:
            st.success("No errors found in recent logs")
    
    def _render_system_metrics_tab(self):
        """Render the system metrics tab."""
        st.subheader("💻 System Metrics")
        
        # Get system monitor
        monitor = get_system_monitor()
        metrics_summary = monitor.get_metrics_summary()
        
        if metrics_summary:
            latest = metrics_summary.get('latest_metrics', {})
            averages = metrics_summary.get('averages', {})
            
            # Current system status
            st.subheader("Current System Status")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cpu_percent = latest.get('cpu_percent', 0)
                st.metric("CPU Usage", f"{cpu_percent:.1f}%")
            
            with col2:
                memory_percent = latest.get('memory_percent', 0)
                st.metric("Memory Usage", f"{memory_percent:.1f}%")
            
            with col3:
                disk_percent = latest.get('disk_percent', 0)
                st.metric("Disk Usage", f"{disk_percent:.1f}%")
            
            with col4:
                ollama_running = latest.get('ollama_running', False)
                status = "🟢 Running" if ollama_running else "🔴 Stopped"
                st.metric("Ollama Status", status)
            
            # Performance trends
            if averages:
                st.subheader("Performance Trends (Last 10 measurements)")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avg CPU", f"{averages.get('cpu_percent', 0):.1f}%")
                
                with col2:
                    st.metric("Avg Memory", f"{averages.get('memory_percent', 0):.1f}%")
                
                with col3:
                    st.metric("Avg Disk", f"{averages.get('disk_percent', 0):.1f}%")
            
            # Application-specific metrics
            if latest:
                st.subheader("Application Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    log_size = latest.get('log_directory_size', 0)
                    st.metric("Log Directory Size", f"{log_size / (1024*1024):.1f} MB")
                
                with col2:
                    python_threads = latest.get('python_threads', 0)
                    st.metric("Python Threads", python_threads)
                
                with col3:
                    python_memory = latest.get('current_process_memory', 0)
                    st.metric("Process Memory", f"{python_memory / (1024*1024):.1f} MB")
        else:
            st.info("System monitoring not active. Metrics will appear once monitoring starts.")
    
    def _get_recent_logs(self) -> List[LogEntry]:
        """Get recent log entries."""
        try:
            main_log = self.log_dir / "agent-aiops.log"
            if not main_log.exists():
                return []
            
            entries = []
            with open(main_log, 'r', encoding='utf-8') as f:
                lines = deque(f, maxlen=self.max_recent_logs)
                for line in lines:
                    if line.strip():
                        entries.append(LogEntry(line))
            
            return entries
        except Exception:
            return []
    
    def _read_specialized_log(self, filename: str) -> List[Dict[str, Any]]:
        """Read and parse a specialized log file."""
        try:
            # Try JSON version first
            json_file = self.log_dir / f"{filename}.json"
            if json_file.exists():
                entries = []
                with open(json_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                entries.append(json.loads(line.strip()))
                            except json.JSONDecodeError:
                                continue
                return entries[-20:]  # Return last 20 entries
            
            # Fallback to text version
            text_file = self.log_dir / filename
            if text_file.exists():
                entries = []
                with open(text_file, 'r', encoding='utf-8') as f:
                    lines = list(f)[-20:]  # Last 20 lines
                    for line in lines:
                        if line.strip():
                            # Basic parsing for text logs
                            parts = line.strip().split(' - ', 2)
                            if len(parts) >= 3:
                                entries.append({
                                    'timestamp': parts[0],
                                    'level': parts[1],
                                    'message': parts[2]
                                })
                return entries
            
            return []
        except Exception:
            return []


def render_log_dashboard():
    """Render the log dashboard in Streamlit."""
    dashboard = LogDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    # For testing the dashboard standalone
    render_log_dashboard()
