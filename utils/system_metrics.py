"""
System Metrics Collection for Agent-AIOps

Collects and logs comprehensive system performance metrics including:
- Memory usage
- CPU utilization
- Disk I/O
- Network statistics
- Application-specific metrics
"""

import psutil
import time
import threading
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from utils.logger import get_logger, log_system_metrics


@dataclass
class SystemMetrics:
    """Container for system performance metrics."""
    
    # CPU Metrics
    cpu_percent: float
    cpu_count: int
    load_average: tuple
    
    # Memory Metrics
    memory_total: int
    memory_available: int
    memory_percent: float
    memory_used: int
    
    # Disk Metrics
    disk_total: int
    disk_used: int
    disk_free: int
    disk_percent: float
    
    # Network Metrics
    network_bytes_sent: int
    network_bytes_recv: int
    network_packets_sent: int
    network_packets_recv: int
    
    # Process Metrics
    process_count: int
    current_process_memory: int
    current_process_cpu: float
    
    # Application Metrics
    active_connections: int = 0
    cache_size: int = 0
    session_count: int = 0
    
    # Timestamp
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class SystemMonitor:
    """System metrics collector and monitor."""
    
    def __init__(self, collection_interval: float = 30.0):
        self.collection_interval = collection_interval
        self.logger = get_logger(__name__)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_network_counters = None
        self.metrics_history = []
        self.max_history = 100  # Keep last 100 metrics snapshots
        
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics (for root filesystem)
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            process_count = len(psutil.pids())
            current_process = psutil.Process()
            current_process_memory = current_process.memory_info().rss
            current_process_cpu = current_process.cpu_percent()
            
            metrics = SystemMetrics(
                # CPU
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                load_average=load_avg,
                
                # Memory
                memory_total=memory.total,
                memory_available=memory.available,
                memory_percent=memory.percent,
                memory_used=memory.used,
                
                # Disk
                disk_total=disk.total,
                disk_used=disk.used,
                disk_free=disk.free,
                disk_percent=disk.percent,
                
                # Network
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                network_packets_sent=network.packets_sent,
                network_packets_recv=network.packets_recv,
                
                # Process
                process_count=process_count,
                current_process_memory=current_process_memory,
                current_process_cpu=current_process_cpu
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error("Failed to collect system metrics", error=str(e))
            # Return minimal metrics on error
            return SystemMetrics(
                cpu_percent=0, cpu_count=1, load_average=(0, 0, 0),
                memory_total=0, memory_available=0, memory_percent=0, memory_used=0,
                disk_total=0, disk_used=0, disk_free=0, disk_percent=0,
                network_bytes_sent=0, network_bytes_recv=0,
                network_packets_sent=0, network_packets_recv=0,
                process_count=0, current_process_memory=0, current_process_cpu=0
            )
    
    def collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics."""
        app_metrics = {}
        
        try:
            # Check log directory size
            log_dir = Path("logs")
            if log_dir.exists():
                log_size = sum(f.stat().st_size for f in log_dir.rglob('*') if f.is_file())
                app_metrics["log_directory_size"] = log_size
                app_metrics["log_file_count"] = len(list(log_dir.rglob('*.log')))
            
            # Check for Ollama process
            ollama_running = False
            ollama_memory = 0
            ollama_cpu = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
                try:
                    if 'ollama' in proc.info['name'].lower():
                        ollama_running = True
                        ollama_memory += proc.info['memory_info'].rss
                        ollama_cpu += proc.info['cpu_percent']
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            app_metrics.update({
                "ollama_running": ollama_running,
                "ollama_memory_mb": ollama_memory // (1024 * 1024),
                "ollama_cpu_percent": ollama_cpu
            })
            
            # Add Python process specifics (with error handling)
            try:
                current_process = psutil.Process()
                app_metrics.update({
                    "python_threads": current_process.num_threads(),
                    "python_connections": len(current_process.connections()),
                    "python_open_files": len(current_process.open_files())
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                # Handle cases where we can't access process info
                app_metrics.update({
                    "python_threads": 0,
                    "python_connections": 0,
                    "python_open_files": 0
                })
            
        except Exception as e:
            self.logger.error("Failed to collect application metrics", error=str(e))
        
        return app_metrics
    
    def log_metrics(self):
        """Collect and log current metrics."""
        metrics = self.collect_metrics()
        app_metrics = self.collect_application_metrics()
        
        # Combine all metrics
        all_metrics = metrics.to_dict()
        all_metrics.update(app_metrics)
        
        # Add to history
        self.metrics_history.append(all_metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
        
        # Log system metrics
        log_system_metrics("system_monitor", all_metrics)
        
        # Log performance alerts if needed
        self._check_performance_alerts(metrics, app_metrics)
    
    def _check_performance_alerts(self, metrics: SystemMetrics, app_metrics: Dict[str, Any]):
        """Check for performance issues and log alerts."""
        alerts = []
        
        # High memory usage
        if metrics.memory_percent > 85:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        # High CPU usage
        if metrics.cpu_percent > 80:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # High disk usage
        if metrics.disk_percent > 90:
            alerts.append(f"High disk usage: {metrics.disk_percent:.1f}%")
        
        # Ollama not running
        if not app_metrics.get("ollama_running", False):
            alerts.append("Ollama service not detected")
        
        # Log alerts
        for alert in alerts:
            self.logger.warning("Performance Alert", alert=alert, **metrics.to_dict())
    
    def start_monitoring(self):
        """Start continuous metrics collection."""
        if self.running:
            self.logger.warning("System monitor already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        self.logger.info("System monitoring started", interval=self.collection_interval)
    
    def stop_monitoring(self):
        """Stop continuous metrics collection."""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self.log_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error("Error in monitoring loop", error=str(e))
                time.sleep(self.collection_interval)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recent metrics."""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        
        # Calculate averages over last 10 measurements
        recent_metrics = self.metrics_history[-10:]
        
        avg_cpu = sum(m.get('cpu_percent', 0) for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.get('memory_percent', 0) for m in recent_metrics) / len(recent_metrics)
        avg_disk = sum(m.get('disk_percent', 0) for m in recent_metrics) / len(recent_metrics)
        
        return {
            "latest_metrics": latest,
            "averages": {
                "cpu_percent": round(avg_cpu, 1),
                "memory_percent": round(avg_memory, 1),
                "disk_percent": round(avg_disk, 1)
            },
            "history_count": len(self.metrics_history),
            "monitoring_active": self.running
        }


# Global system monitor instance
_system_monitor = None


def get_system_monitor() -> SystemMonitor:
    """Get the global system monitor instance."""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    return _system_monitor


def start_system_monitoring(interval: float = 30.0):
    """Start system monitoring with specified interval."""
    monitor = get_system_monitor()
    monitor.collection_interval = interval
    monitor.start_monitoring()


def stop_system_monitoring():
    """Stop system monitoring."""
    monitor = get_system_monitor()
    monitor.stop_monitoring()


def collect_current_metrics() -> Dict[str, Any]:
    """Collect current system metrics synchronously."""
    monitor = get_system_monitor()
    metrics = monitor.collect_metrics()
    app_metrics = monitor.collect_application_metrics()
    
    all_metrics = metrics.to_dict()
    all_metrics.update(app_metrics)
    
    return all_metrics


# Convenience function for one-time metric collection
def log_current_metrics():
    """Log current system metrics once."""
    monitor = get_system_monitor()
    monitor.log_metrics()


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="System Metrics Monitor")
    parser.add_argument("--interval", type=float, default=5.0, help="Collection interval (seconds)")
    parser.add_argument("--duration", type=int, default=60, help="Monitor duration (seconds)")
    parser.add_argument("--once", action="store_true", help="Collect metrics once and exit")
    
    args = parser.parse_args()
    
    if args.once:
        metrics = collect_current_metrics()
        print("Current system metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    else:
        print(f"Starting system monitoring for {args.duration} seconds...")
        start_system_monitoring(args.interval)
        
        try:
            time.sleep(args.duration)
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            stop_system_monitoring()
