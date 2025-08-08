"""
Log Analysis Utilities for Agent-AIOps

Provides tools to analyze and understand application logs:
- Performance analysis
- Error tracking and trending
- User interaction patterns
- Agent behavior insights
- Search query analytics
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd


class LogAnalyzer:
    """Comprehensive log analysis utility."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.json_log_file = self.log_dir / "agent-aiops.json"
        self.text_log_file = self.log_dir / "agent-aiops.log"
    
    def load_json_logs(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Load and parse JSON logs from the specified time period."""
        if not self.json_log_file.exists():
            return []
        
        logs = []
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        try:
            with open(self.json_log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        log_time = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))
                        
                        if log_time >= cutoff_time:
                            logs.append(log_entry)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except FileNotFoundError:
            pass
        
        return sorted(logs, key=lambda x: x['timestamp'])
    
    def performance_analysis(self, hours_back: int = 24) -> Dict[str, Any]:
        """Analyze performance metrics from logs."""
        logs = self.load_json_logs(hours_back)
        
        analysis = {
            "ollama_requests": [],
            "search_queries": [],
            "agent_sessions": [],
            "performance_summary": {},
            "error_rates": {}
        }
        
        for log in logs:
            if 'extra_fields' not in log:
                continue
                
            fields = log['extra_fields']
            
            # Ollama performance
            if 'model' in fields and 'duration_ms' in log:
                analysis["ollama_requests"].append({
                    "timestamp": log['timestamp'],
                    "model": fields['model'],
                    "duration_ms": log['duration_ms'],
                    "tokens_generated": fields.get('tokens_generated', 0),
                    "input_tokens": fields.get('input_tokens', 0)
                })
            
            # Search performance
            if 'provider' in fields and 'search_time_ms' in fields:
                analysis["search_queries"].append({
                    "timestamp": log['timestamp'],
                    "provider": fields['provider'],
                    "duration_ms": fields['search_time_ms'],
                    "results_count": fields.get('results_count', 0),
                    "query": fields.get('query', '')
                })
            
            # Agent sessions
            if 'correlation_id' in log and 'total_time_ms' in fields:
                analysis["agent_sessions"].append({
                    "timestamp": log['timestamp'],
                    "correlation_id": log['correlation_id'],
                    "total_time_ms": fields['total_time_ms'],
                    "model": fields.get('model', '')
                })
        
        # Calculate summaries
        if analysis["ollama_requests"]:
            ollama_times = [r['duration_ms'] for r in analysis["ollama_requests"]]
            analysis["performance_summary"]["ollama"] = {
                "avg_response_time_ms": sum(ollama_times) / len(ollama_times),
                "min_response_time_ms": min(ollama_times),
                "max_response_time_ms": max(ollama_times),
                "total_requests": len(ollama_times)
            }
        
        if analysis["search_queries"]:
            search_times = [q['duration_ms'] for q in analysis["search_queries"]]
            analysis["performance_summary"]["search"] = {
                "avg_response_time_ms": sum(search_times) / len(search_times),
                "min_response_time_ms": min(search_times),
                "max_response_time_ms": max(search_times),
                "total_queries": len(search_times)
            }
        
        # Error rates
        error_logs = [log for log in logs if log['level'] == 'ERROR']
        total_logs = len(logs)
        analysis["error_rates"] = {
            "total_errors": len(error_logs),
            "error_rate_percent": (len(error_logs) / total_logs * 100) if total_logs > 0 else 0,
            "errors_by_module": Counter([log['module'] for log in error_logs])
        }
        
        return analysis
    
    def user_interaction_analysis(self, hours_back: int = 24) -> Dict[str, Any]:
        """Analyze user interaction patterns."""
        logs = self.load_json_logs(hours_back)
        
        analysis = {
            "session_count": 0,
            "mode_usage": {"normal": 0, "agent": 0},
            "average_session_length": 0,
            "popular_queries": [],
            "hourly_activity": defaultdict(int)
        }
        
        user_sessions = defaultdict(list)
        
        for log in logs:
            if 'extra_fields' not in log:
                continue
                
            fields = log['extra_fields']
            
            # Track user interactions
            if 'action' in fields and fields['action'] == 'chat_input':
                correlation_id = log.get('correlation_id', 'unknown')
                user_sessions[correlation_id].append({
                    "timestamp": log['timestamp'],
                    "mode": fields.get('mode', 'normal'),
                    "prompt_length": fields.get('prompt_length', 0)
                })
                
                # Count mode usage
                mode = fields.get('mode', 'normal')
                if mode in analysis["mode_usage"]:
                    analysis["mode_usage"][mode] += 1
                
                # Hourly activity
                hour = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00')).hour
                analysis["hourly_activity"][hour] += 1
        
        analysis["session_count"] = len(user_sessions)
        
        # Calculate average session length (in interactions)
        if user_sessions:
            session_lengths = [len(interactions) for interactions in user_sessions.values()]
            analysis["average_session_length"] = sum(session_lengths) / len(session_lengths)
        
        return analysis
    
    def search_analytics(self, hours_back: int = 24) -> Dict[str, Any]:
        """Analyze search query patterns and performance."""
        logs = self.load_json_logs(hours_back)
        
        analysis = {
            "total_queries": 0,
            "provider_usage": defaultdict(int),
            "provider_performance": defaultdict(list),
            "query_patterns": [],
            "success_rates": defaultdict(lambda: {"success": 0, "total": 0})
        }
        
        for log in logs:
            if 'extra_fields' not in log:
                continue
                
            fields = log['extra_fields']
            
            if 'provider' in fields and 'query' in fields:
                analysis["total_queries"] += 1
                provider = fields['provider']
                
                # Track provider usage
                analysis["provider_usage"][provider] += 1
                
                # Track performance
                if 'search_time_ms' in fields:
                    analysis["provider_performance"][provider].append(fields['search_time_ms'])
                
                # Track success rates
                analysis["success_rates"][provider]["total"] += 1
                if fields.get('results_count', 0) > 0:
                    analysis["success_rates"][provider]["success"] += 1
                
                # Collect query patterns
                query = fields.get('query', '')
                if query:
                    analysis["query_patterns"].append({
                        "query": query[:50] + "..." if len(query) > 50 else query,
                        "provider": provider,
                        "timestamp": log['timestamp'],
                        "results_count": fields.get('results_count', 0)
                    })
        
        # Calculate success rates
        for provider, stats in analysis["success_rates"].items():
            if stats["total"] > 0:
                stats["success_rate"] = stats["success"] / stats["total"] * 100
            else:
                stats["success_rate"] = 0
        
        return analysis
    
    def error_analysis(self, hours_back: int = 24) -> Dict[str, Any]:
        """Analyze error patterns and trends."""
        logs = self.load_json_logs(hours_back)
        
        error_logs = [log for log in logs if log['level'] in ['ERROR', 'CRITICAL']]
        
        analysis = {
            "total_errors": len(error_logs),
            "errors_by_level": Counter([log['level'] for log in error_logs]),
            "errors_by_module": Counter([log['module'] for log in error_logs]),
            "errors_by_function": Counter([log['function'] for log in error_logs]),
            "recent_errors": [],
            "error_timeline": defaultdict(int)
        }
        
        for error_log in error_logs[-10:]:  # Last 10 errors
            analysis["recent_errors"].append({
                "timestamp": error_log['timestamp'],
                "level": error_log['level'],
                "module": error_log['module'],
                "function": error_log['function'],
                "message": error_log['message'],
                "correlation_id": error_log.get('correlation_id', 'N/A')
            })
            
            # Error timeline (by hour)
            hour = datetime.fromisoformat(error_log['timestamp'].replace('Z', '+00:00')).hour
            analysis["error_timeline"][hour] += 1
        
        return analysis
    
    def generate_report(self, hours_back: int = 24) -> str:
        """Generate a comprehensive text report."""
        performance = self.performance_analysis(hours_back)
        user_interactions = self.user_interaction_analysis(hours_back)
        search_analytics = self.search_analytics(hours_back)
        error_analysis = self.error_analysis(hours_back)
        
        report = []
        report.append("="*60)
        report.append(f"AGENT-AIOPS LOG ANALYSIS REPORT")
        report.append(f"Time Period: Last {hours_back} hours")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*60)
        
        # Performance Summary
        report.append("\n📊 PERFORMANCE SUMMARY")
        report.append("-" * 30)
        
        if performance["performance_summary"].get("ollama"):
            ollama = performance["performance_summary"]["ollama"]
            report.append(f"Ollama API:")
            report.append(f"  • Total Requests: {ollama['total_requests']}")
            report.append(f"  • Avg Response Time: {ollama['avg_response_time_ms']:.1f}ms")
            report.append(f"  • Min/Max Response: {ollama['min_response_time_ms']:.1f}ms / {ollama['max_response_time_ms']:.1f}ms")
        
        if performance["performance_summary"].get("search"):
            search = performance["performance_summary"]["search"]
            report.append(f"Search Performance:")
            report.append(f"  • Total Queries: {search['total_queries']}")
            report.append(f"  • Avg Response Time: {search['avg_response_time_ms']:.1f}ms")
            report.append(f"  • Min/Max Response: {search['min_response_time_ms']:.1f}ms / {search['max_response_time_ms']:.1f}ms")
        
        # User Activity
        report.append(f"\n👥 USER ACTIVITY")
        report.append("-" * 30)
        report.append(f"Sessions: {user_interactions['session_count']}")
        report.append(f"Mode Usage: Normal ({user_interactions['mode_usage']['normal']}) | Agent ({user_interactions['mode_usage']['agent']})")
        report.append(f"Avg Session Length: {user_interactions['average_session_length']:.1f} interactions")
        
        # Search Analytics
        if search_analytics["total_queries"] > 0:
            report.append(f"\n🔍 SEARCH ANALYTICS")
            report.append("-" * 30)
            report.append(f"Total Queries: {search_analytics['total_queries']}")
            
            for provider, count in search_analytics["provider_usage"].items():
                success_rate = search_analytics["success_rates"][provider]["success_rate"]
                report.append(f"  • {provider}: {count} queries ({success_rate:.1f}% success)")
        
        # Error Summary
        report.append(f"\n🚨 ERROR SUMMARY")
        report.append("-" * 30)
        report.append(f"Total Errors: {error_analysis['total_errors']}")
        report.append(f"Error Rate: {performance['error_rates']['error_rate_percent']:.2f}%")
        
        if error_analysis["errors_by_module"]:
            report.append("Errors by Module:")
            for module, count in error_analysis["errors_by_module"].most_common(5):
                report.append(f"  • {module}: {count}")
        
        # Recent Errors
        if error_analysis["recent_errors"]:
            report.append(f"\nRecent Errors:")
            for error in error_analysis["recent_errors"][-3:]:
                timestamp = error['timestamp'][:19]  # Remove microseconds
                report.append(f"  • [{timestamp}] {error['module']}.{error['function']}: {error['message'][:80]}...")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
    
    def export_performance_csv(self, output_file: str = "performance_data.csv"):
        """Export performance data to CSV for further analysis."""
        performance = self.performance_analysis()
        
        # Combine all performance data
        all_data = []
        
        for req in performance["ollama_requests"]:
            all_data.append({
                "timestamp": req["timestamp"],
                "type": "ollama",
                "duration_ms": req["duration_ms"],
                "model": req["model"],
                "tokens_generated": req["tokens_generated"],
                "input_tokens": req["input_tokens"]
            })
        
        for query in performance["search_queries"]:
            all_data.append({
                "timestamp": query["timestamp"],
                "type": "search",
                "duration_ms": query["duration_ms"],
                "provider": query["provider"],
                "results_count": query["results_count"],
                "query": query["query"][:100]  # Truncate long queries
            })
        
        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv(output_file, index=False)
            return f"Performance data exported to {output_file}"
        else:
            return "No performance data available to export"


def analyze_logs_cli():
    """Command-line interface for log analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Agent-AIOps logs")
    parser.add_argument("--hours", type=int, default=24, help="Hours of logs to analyze")
    parser.add_argument("--log-dir", default="logs", help="Directory containing log files")
    parser.add_argument("--export-csv", help="Export performance data to CSV file")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    
    args = parser.parse_args()
    
    analyzer = LogAnalyzer(args.log_dir)
    
    if args.export_csv:
        result = analyzer.export_performance_csv(args.export_csv)
        print(result)
        return
    
    if args.format == "json":
        # JSON output for programmatic use
        results = {
            "performance": analyzer.performance_analysis(args.hours),
            "user_interactions": analyzer.user_interaction_analysis(args.hours),
            "search_analytics": analyzer.search_analytics(args.hours),
            "error_analysis": analyzer.error_analysis(args.hours)
        }
        print(json.dumps(results, indent=2, default=str))
    else:
        # Human-readable text report
        report = analyzer.generate_report(args.hours)
        print(report)


if __name__ == "__main__":
    analyze_logs_cli()
