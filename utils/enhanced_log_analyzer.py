#!/usr/bin/env python3
"""
Enhanced Log Analyzer for Agent-AIOps

Provides advanced log analysis with:
- Real-time log streaming with filtering
- LLM communication tracking
- Request tracing and correlation
- Performance analytics
- Error pattern analysis
- Interactive dashboard
"""

import json
import time
import argparse
import re
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from collections import defaultdict, deque


class LogEntry:
    """Represents a single log entry with parsed fields."""
    
    def __init__(self, raw_line: str):
        self.raw_line = raw_line
        self.timestamp = None
        self.level = None
        self.logger = None
        self.message = None
        self.correlation_id = None
        self.data = {}
        self._parse()
    
    def _parse(self):
        """Parse the log line into structured fields."""
        try:
            # Try to parse as JSON first
            if self.raw_line.strip().startswith('{'):
                log_data = json.loads(self.raw_line.strip())
                self.timestamp = log_data.get('timestamp')
                self.level = log_data.get('level')
                self.logger = log_data.get('logger')
                self.message = log_data.get('message')
                self.correlation_id = log_data.get('correlation_id')
                self.data = {k: v for k, v in log_data.items() 
                           if k not in ['timestamp', 'level', 'logger', 'message']}
            else:
                # Parse structured text format
                parts = self.raw_line.split(' - ', 2)
                if len(parts) >= 3:
                    self.level = parts[0].split()[-1] if parts[0] else 'INFO'
                    self.logger = parts[1] if parts[1] else 'unknown'
                    self.message = parts[2] if parts[2] else ''
                
                # Extract correlation ID if present
                correlation_match = re.search(r'correlation_id[=:](\w+)', self.raw_line)
                if correlation_match:
                    self.correlation_id = correlation_match.group(1)
                    
        except (json.JSONDecodeError, Exception):
            # Fallback to basic parsing
            self.message = self.raw_line.strip()
            self.level = 'INFO'
            self.logger = 'unknown'


class LogStats:
    """Statistics container for log analysis."""
    
    def __init__(self):
        self.total_entries = 0
        self.level_counts = defaultdict(int)
        self.logger_counts = defaultdict(int)
        self.error_patterns = defaultdict(int)
        self.correlation_ids = set()
        self.llm_requests = []
        self.llm_responses = []
        self.tool_executions = []
        self.performance_metrics = []
        self.request_traces = defaultdict(list)
        
    def add_entry(self, entry: LogEntry):
        """Add a log entry to statistics."""
        self.total_entries += 1
        self.level_counts[entry.level] += 1
        self.logger_counts[entry.logger] += 1
        
        if entry.correlation_id:
            self.correlation_ids.add(entry.correlation_id)
            self.request_traces[entry.correlation_id].append(entry)
        
        # Categorize specific log types
        if 'ollama.request' in entry.logger:
            self.llm_requests.append(entry)
        elif 'ollama.response' in entry.logger:
            self.llm_responses.append(entry)
        elif 'tools.execution' in entry.logger:
            self.tool_executions.append(entry)
        elif entry.level == 'ERROR':
            # Extract error patterns
            if entry.message:
                error_key = entry.message.split(':')[0]
                self.error_patterns[error_key] += 1


class EnhancedLogAnalyzer:
    """Advanced log analyzer with filtering and analysis capabilities."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.stats = LogStats()
        self.filters = {
            'level': None,
            'logger': None,
            'correlation_id': None,
            'time_range': None,
            'keywords': []
        }
        
    def set_filter(self, filter_type: str, value: Any):
        """Set a filter for log analysis."""
        if filter_type in self.filters:
            self.filters[filter_type] = value
    
    def clear_filters(self):
        """Clear all filters."""
        for key in self.filters:
            if key == 'keywords':
                self.filters[key] = []
            else:
                self.filters[key] = None
    
    def _matches_filters(self, entry: LogEntry) -> bool:
        """Check if entry matches current filters."""
        if self.filters['level'] and entry.level != self.filters['level']:
            return False
        
        if self.filters['logger'] and self.filters['logger'] not in entry.logger:
            return False
        
        if self.filters['correlation_id'] and entry.correlation_id != self.filters['correlation_id']:
            return False
        
        if self.filters['keywords']:
            message_lower = entry.message.lower() if entry.message else ''
            if not any(keyword.lower() in message_lower for keyword in self.filters['keywords']):
                return False
        
        return True
    
    def analyze_log_file(self, file_path: Path) -> LogStats:
        """Analyze a single log file."""
        stats = LogStats()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = LogEntry(line)
                        if self._matches_filters(entry):
                            stats.add_entry(entry)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        return stats
    
    def analyze_logs(self, file_pattern: str = "*.log") -> LogStats:
        """Analyze all log files matching pattern."""
        self.stats = LogStats()
        
        if not self.log_dir.exists():
            print(f"Log directory {self.log_dir} does not exist")
            return self.stats
        
        log_files = list(self.log_dir.glob(file_pattern))
        if not log_files:
            print(f"No log files found in {self.log_dir}")
            return self.stats
        
        print(f"Analyzing {len(log_files)} log files...")
        
        for log_file in sorted(log_files):
            file_stats = self.analyze_log_file(log_file)
            self._merge_stats(file_stats)
        
        return self.stats
    
    def _merge_stats(self, other_stats: LogStats):
        """Merge statistics from another LogStats object."""
        self.stats.total_entries += other_stats.total_entries
        
        for level, count in other_stats.level_counts.items():
            self.stats.level_counts[level] += count
        
        for logger, count in other_stats.logger_counts.items():
            self.stats.logger_counts[logger] += count
        
        for pattern, count in other_stats.error_patterns.items():
            self.stats.error_patterns[pattern] += count
        
        self.stats.correlation_ids.update(other_stats.correlation_ids)
        self.stats.llm_requests.extend(other_stats.llm_requests)
        self.stats.llm_responses.extend(other_stats.llm_responses)
        self.stats.tool_executions.extend(other_stats.tool_executions)
        self.stats.performance_metrics.extend(other_stats.performance_metrics)
        
        for cid, entries in other_stats.request_traces.items():
            self.stats.request_traces[cid].extend(entries)
    
    def print_summary(self):
        """Print a comprehensive log analysis summary."""
        print("\n" + "="*60)
        print("ENHANCED LOG ANALYSIS SUMMARY")
        print("="*60)
        
        # Basic statistics
        print(f"\n📊 BASIC STATISTICS")
        print(f"Total log entries: {self.stats.total_entries:,}")
        print(f"Unique correlation IDs: {len(self.stats.correlation_ids):,}")
        print(f"Time period: {self._get_time_range()}")
        
        # Log levels
        print(f"\n📈 LOG LEVELS")
        for level, count in sorted(self.stats.level_counts.items()):
            percentage = (count / self.stats.total_entries * 100) if self.stats.total_entries > 0 else 0
            print(f"  {level:8} {count:6,} ({percentage:5.1f}%)")
        
        # Top loggers
        print(f"\n🔍 TOP LOGGERS")
        top_loggers = sorted(self.stats.logger_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for logger, count in top_loggers:
            percentage = (count / self.stats.total_entries * 100) if self.stats.total_entries > 0 else 0
            print(f"  {logger:30} {count:6,} ({percentage:5.1f}%)")
        
        # LLM Communication
        self._print_llm_stats()
        
        # Tool Execution
        self._print_tool_stats()
        
        # Error Analysis
        self._print_error_stats()
        
        # Request Tracing
        self._print_request_tracing_stats()
    
    def _get_time_range(self) -> str:
        """Get the time range of logs."""
        return "Unable to determine time range"  # Simplified for now
    
    def _print_llm_stats(self):
        """Print LLM communication statistics."""
        print(f"\n🤖 LLM COMMUNICATION")
        print(f"Total LLM requests: {len(self.stats.llm_requests):,}")
        print(f"Total LLM responses: {len(self.stats.llm_responses):,}")
        
        if self.stats.llm_requests:
            # Calculate token statistics
            total_input_tokens = 0
            total_output_tokens = 0
            total_duration = 0
            
            for request in self.stats.llm_requests:
                total_input_tokens += request.data.get('estimated_input_tokens', 0)
            
            for response in self.stats.llm_responses:
                total_output_tokens += response.data.get('tokens_generated', 0)
                total_duration += response.data.get('duration_ms', 0)
            
            avg_duration = total_duration / len(self.stats.llm_responses) if self.stats.llm_responses else 0
            
            print(f"  Total input tokens: {total_input_tokens:,}")
            print(f"  Total output tokens: {total_output_tokens:,}")
            print(f"  Average response time: {avg_duration:.1f}ms")
            
            if total_output_tokens > 0 and total_duration > 0:
                tokens_per_second = total_output_tokens / (total_duration / 1000)
                print(f"  Average tokens/second: {tokens_per_second:.1f}")
    
    def _print_tool_stats(self):
        """Print tool execution statistics."""
        print(f"\n🔧 TOOL EXECUTION")
        print(f"Total tool executions: {len(self.stats.tool_executions):,}")
        
        if self.stats.tool_executions:
            tool_counts = defaultdict(int)
            tool_success_counts = defaultdict(int)
            tool_durations = defaultdict(list)
            
            for execution in self.stats.tool_executions:
                tool_name = execution.data.get('tool_name', 'unknown')
                tool_counts[tool_name] += 1
                
                if execution.data.get('success', False):
                    tool_success_counts[tool_name] += 1
                
                duration = execution.data.get('duration_ms', 0)
                if duration > 0:
                    tool_durations[tool_name].append(duration)
            
            print(f"  Tool usage:")
            for tool_name, count in sorted(tool_counts.items()):
                success_count = tool_success_counts[tool_name]
                success_rate = (success_count / count * 100) if count > 0 else 0
                avg_duration = sum(tool_durations[tool_name]) / len(tool_durations[tool_name]) if tool_durations[tool_name] else 0
                print(f"    {tool_name:15} {count:4} executions, {success_rate:5.1f}% success, {avg_duration:6.1f}ms avg")
    
    def _print_error_stats(self):
        """Print error analysis."""
        if self.stats.error_patterns:
            print(f"\n❌ ERROR ANALYSIS")
            print(f"Total errors: {self.stats.level_counts.get('ERROR', 0):,}")
            print(f"Top error patterns:")
            
            top_errors = sorted(self.stats.error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
            for pattern, count in top_errors:
                print(f"  {pattern:40} {count:4} occurrences")
    
    def _print_request_tracing_stats(self):
        """Print request tracing statistics."""
        if self.stats.request_traces:
            print(f"\n🔗 REQUEST TRACING")
            print(f"Traced requests: {len(self.stats.request_traces):,}")
            
            # Calculate request completion statistics
            completed_requests = 0
            total_duration = 0
            
            for cid, entries in self.stats.request_traces.items():
                has_start = any('Request Started' in entry.message for entry in entries if entry.message)
                has_complete = any('Request Completed' in entry.message for entry in entries if entry.message)
                
                if has_start and has_complete:
                    completed_requests += 1
                    # Extract duration from completion entry
                    for entry in entries:
                        if entry.message and 'Request Completed' in entry.message:
                            duration = entry.data.get('total_duration_ms', 0)
                            total_duration += duration
                            break
            
            avg_request_duration = total_duration / completed_requests if completed_requests > 0 else 0
            completion_rate = (completed_requests / len(self.stats.request_traces) * 100) if self.stats.request_traces else 0
            
            print(f"  Completion rate: {completion_rate:.1f}%")
            print(f"  Average request duration: {avg_request_duration:.1f}ms")
    
    def export_to_csv(self, output_file: str):
        """Export analysis results to CSV for further processing."""
        try:
            # Create summary data
            summary_data = {
                'metric': [],
                'value': [],
                'category': []
            }
            
            # Basic stats
            summary_data['metric'].extend(['total_entries', 'unique_correlations'])
            summary_data['value'].extend([self.stats.total_entries, len(self.stats.correlation_ids)])
            summary_data['category'].extend(['basic', 'basic'])
            
            # Level counts
            for level, count in self.stats.level_counts.items():
                summary_data['metric'].append(f'level_{level.lower()}')
                summary_data['value'].append(count)
                summary_data['category'].append('levels')
            
            # LLM stats
            summary_data['metric'].extend(['llm_requests', 'llm_responses'])
            summary_data['value'].extend([len(self.stats.llm_requests), len(self.stats.llm_responses)])
            summary_data['category'].extend(['llm', 'llm'])
            
            # Tool stats
            summary_data['metric'].append('tool_executions')
            summary_data['value'].append(len(self.stats.tool_executions))
            summary_data['category'].append('tools')
            
            df = pd.DataFrame(summary_data)
            df.to_csv(output_file, index=False)
            print(f"\n📊 Analysis exported to {output_file}")
            
        except ImportError:
            print("\n❌ pandas not available. Install with: pip install pandas")
        except Exception as e:
            print(f"\n❌ Export failed: {e}")
    
    def follow_logs(self, file_pattern: str = "*.log", refresh_interval: float = 1.0):
        """Follow logs in real-time with filtering."""
        print(f"Following logs in {self.log_dir} (pattern: {file_pattern})")
        print("Press Ctrl+C to stop")
        print("-" * 60)
        
        last_positions = {}
        
        try:
            while True:
                log_files = list(self.log_dir.glob(file_pattern))
                
                for log_file in log_files:
                    if not log_file.is_file():
                        continue
                    
                    # Track file position
                    if log_file not in last_positions:
                        last_positions[log_file] = 0
                    
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            f.seek(last_positions[log_file])
                            
                            for line in f:
                                if line.strip():
                                    entry = LogEntry(line)
                                    if self._matches_filters(entry):
                                        timestamp = datetime.now().strftime("%H:%M:%S")
                                        print(f"[{timestamp}] {entry.level:5} {entry.logger:20} | {entry.message}")
                            
                            last_positions[log_file] = f.tell()
                    
                    except Exception as e:
                        print(f"Error reading {log_file}: {e}")
                
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\nStopped following logs")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Enhanced Log Analyzer for Agent-AIOps")
    
    parser.add_argument("--log-dir", default="logs", help="Log directory path")
    parser.add_argument("--pattern", default="*.log", help="Log file pattern")
    parser.add_argument("--follow", action="store_true", help="Follow logs in real-time")
    parser.add_argument("--level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Filter by log level")
    parser.add_argument("--logger", help="Filter by logger name (partial match)")
    parser.add_argument("--correlation-id", help="Filter by correlation ID")
    parser.add_argument("--keywords", nargs="+", help="Filter by keywords in message")
    parser.add_argument("--export", help="Export analysis to CSV file")
    parser.add_argument("--refresh", type=float, default=1.0, help="Refresh interval for --follow (seconds)")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = EnhancedLogAnalyzer(args.log_dir)
    
    # Set filters
    if args.level:
        analyzer.set_filter('level', args.level)
    if args.logger:
        analyzer.set_filter('logger', args.logger)
    if args.correlation_id:
        analyzer.set_filter('correlation_id', args.correlation_id)
    if args.keywords:
        analyzer.set_filter('keywords', args.keywords)
    
    if args.follow:
        # Real-time following
        analyzer.follow_logs(args.pattern, args.refresh)
    else:
        # Static analysis
        stats = analyzer.analyze_logs(args.pattern)
        analyzer.print_summary()
        
        if args.export:
            analyzer.export_to_csv(args.export)


if __name__ == "__main__":
    main()
