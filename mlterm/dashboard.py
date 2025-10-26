"""
Textual-based dashboard for real-time ML experiment monitoring.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    DataTable, Header, Footer, Static, TabbedContent, TabPane,
    ProgressBar, Sparkline, Label
)
from textual import events

from .system_monitor import SystemMonitor


class MetricsTable(DataTable):
    """A data table for displaying experiment metrics."""
    
    def __init__(self, title: str = "Metrics"):
        super().__init__()
        self.title = title
        self.cursor_type = "row"
    
    def on_mount(self) -> None:
        """Initialize the table when mounted."""
        self.add_columns("Metric", "Value", "Timestamp")
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update the table with new metrics."""
        self.clear()
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        for key, value in metrics.items():
            if key not in ["timestamp", "datetime", "type", "run_id", "project"]:
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        value = f"{value:.4f}"
                    else:
                        value = str(value)
                else:
                    value = str(value)
                self.add_row(key, value, timestamp)


class SystemStats(Static):
    """Display system statistics."""
    
    def __init__(self):
        super().__init__()
        self.system_monitor = SystemMonitor()
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Label("System Statistics", classes="title")
        yield Static("Loading...", id="cpu-info")
        yield Static("Loading...", id="memory-info")
        yield Static("Loading...", id="gpu-info")
    
    def on_mount(self) -> None:
        """Start updating system stats."""
        self.set_interval(2.0, self.update_stats)
        self.update_stats()
    
    def update_stats(self) -> None:
        """Update system statistics display."""
        try:
            system_info = self.system_monitor.get_system_info()
            
            # CPU info
            cpu_info = system_info["cpu"]
            cpu_text = f"CPU: {cpu_info['cpu_percent']:.1f}% "
            cpu_text += f"({cpu_info['cpu_count']} cores, {cpu_info['cpu_freq_current']:.0f} MHz)"
            
            # Memory info
            memory_info = system_info["memory"]
            memory_gb_used = memory_info["memory_used"] / (1024**3)
            memory_gb_total = memory_info["memory_total"] / (1024**3)
            memory_text = f"System Memory: {memory_gb_used:.1f}GB / {memory_gb_total:.1f}GB "
            memory_text += f"({memory_info['memory_percent']:.1f}%)"
            
            # GPU info
            gpu_info = system_info["gpu"]
            if gpu_info:
                gpu_text = "GPU: "
                for i, gpu in enumerate(gpu_info):
                    gpu_text += f"GPU{i}: {gpu['gpu_utilization']:.1f}% "
                    gpu_text += f"({gpu['gpu_memory_used']}MB/{gpu['gpu_memory_total']}MB) "
            else:
                gpu_text = "GPU: Not available"
            
            self.query_one("#cpu-info", Static).update(cpu_text)
            self.query_one("#memory-info", Static).update(memory_text)
            self.query_one("#gpu-info", Static).update(gpu_text)
            
        except Exception as e:
            error_text = f"Error updating stats: {str(e)}"
            self.query_one("#cpu-info", Static).update(error_text)


class LogViewer(Static):
    """Display recent log entries."""
    
    def __init__(self):
        super().__init__()
        self.max_entries = 20
        self.entries: List[str] = []
    
    def add_entry(self, entry: str) -> None:
        """Add a new log entry."""
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
        
        # Update display
        content = "\n".join(self.entries[-10:])  # Show last 10 entries
        self.update(content)
    
    def clear_entries(self) -> None:
        """Clear all log entries."""
        self.entries.clear()
        self.update("No log entries yet...")


class MLTermDashboard(App):
    """Main dashboard application."""
    
    CSS = """
    .title {
        text-style: bold;
        color: $accent;
    }
    
    .metric-value {
        text-style: bold;
        color: $success;
    }
    
    .error {
        color: $error;
    }
    
    .warning {
        color: $warning;
    }
    
    #system-stats {
        height: 20%;
        border: solid $primary;
    }
    
    #metrics-table {
        height: 40%;
        border: solid $primary;
    }
    
    #log-viewer {
        height: 40%;
        border: solid $primary;
    }
    """
    
    def __init__(self, log_file: Optional[Path] = None, refresh_rate: float = 0.5):
        super().__init__()
        self.log_file = log_file
        self.refresh_rate = refresh_rate
        self.last_position = 0
        self.metrics: Dict[str, Any] = {}
        self.system_monitor = SystemMonitor()
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        with TabbedContent():
            with TabPane("Overview", id="overview-tab"):
                yield Container(
                    SystemStats(),
                    MetricsTable(),
                    classes="overview"
                )
            with TabPane("System", id="system-tab"):
                yield Container(
                    SystemStats(),
                    classes="system"
                )
            with TabPane("Logs", id="logs-tab"):
                yield Container(
                    LogViewer(),
                    classes="logs"
                )
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the dashboard."""
        self.title = "mlterm Dashboard"
        self.sub_title = f"Monitoring: {self.log_file.name if self.log_file else 'No log file'}"
        
        # Start monitoring
        if self.log_file:
            self.set_interval(self.refresh_rate, self.update_from_log)
        
        # Tabs are now composed directly in the compose method
    
    def update_from_log(self) -> None:
        """Update dashboard from log file."""
        if not self.log_file or not self.log_file.exists():
            return
        
        try:
            with open(self.log_file, 'r') as f:
                f.seek(self.last_position)
                new_lines = f.readlines()
                self.last_position = f.tell()
            
            # Process new entries immediately
            for line in new_lines:
                try:
                    entry = json.loads(line.strip())
                    self.process_log_entry(entry)
                except json.JSONDecodeError:
                    continue
            
            # Force immediate refresh if we processed new data
            if new_lines:
                self.refresh()
                    
        except Exception as e:
            self.log(f"Error reading log file: {e}")
    
    def process_log_entry(self, entry: Dict[str, Any]) -> None:
        """Process a log entry and update the dashboard."""
        entry_type = entry.get("type", "unknown")
        
        if entry_type == "metric":
            # Update metrics table
            self.metrics.update(entry)
            try:
                metrics_table = self.query_one(MetricsTable)
                metrics_table.update_metrics(entry)
            except:
                pass  # Metrics table might not be ready yet
            
        elif entry_type in ["metadata", "hyperparameters", "artifact", "finish"]:
            # Add to log viewer
            timestamp = entry.get("datetime", datetime.now().isoformat())
            log_text = f"[{timestamp}] {entry_type.upper()}: {entry.get('run_id', 'unknown')}"
            
            if entry_type == "finish":
                log_text += f" - Status: {entry.get('status', 'unknown')}"
            
            try:
                log_viewer = self.query_one(LogViewer)
                log_viewer.add_entry(log_text)
            except:
                pass  # Log viewer might not be ready yet
    
    def on_key(self, event: events.Key) -> None:
        """Handle key presses."""
        if event.key == "q":
            self.exit()
        elif event.key == "r":
            # Refresh manually
            if self.log_file:
                self.update_from_log()
