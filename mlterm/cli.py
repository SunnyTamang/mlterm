"""
Command-line interface for mlterm.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .dashboard import MLTermDashboard
from .tracker import Tracker

app = typer.Typer(
    name="mlterm",
    help="A Terminal-Based ML Experiment Tracker",
    add_completion=False
)
console = Console()


@app.command()
def dashboard(
    project: Optional[str] = typer.Option(
        None,
    "--project",
    "-p",
    help="Project name to monitor"
    ),
    log_dir: Path = typer.Option(
        "./logs",
        "--log-dir",
        "-d",
        help="Directory containing log files"
    ),
    refresh: float = typer.Option(
        0.5,
        "--refresh",
        "-r",
        help="Refresh rate in seconds"
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Specific run ID to monitor"
    )
):
    """Launch the real-time dashboard."""
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        console.print(f"[red]Error: Log directory '{log_dir}' does not exist[/red]")
        raise typer.Exit(1)
    
    # Find log file
    log_file = None
    if project and run_id:
        log_file = log_dir / f"{project}_{run_id}.jsonl"
    elif project:
        # Find the most recent log file for the project
        pattern = f"{project}_*.jsonl"
        log_files = list(log_dir.glob(pattern))
        if log_files:
            log_file = max(log_files, key=lambda f: f.stat().st_mtime)
    else:
        # Find the most recent log file
        log_files = list(log_dir.glob("*.jsonl"))
        if log_files:
            log_file = max(log_files, key=lambda f: f.stat().st_mtime)
    
    if not log_file or not log_file.exists():
        console.print(f"[red]Error: No log file found for project '{project}'[/red]")
        if project:
            console.print(f"Looking for: {log_dir / f'{project}_*.jsonl'}")
        raise typer.Exit(1)
    
    console.print(f"[green]Starting dashboard for: {log_file.name}[/green]")
    console.print(f"[dim]Refresh rate: {refresh}s[/dim]")
    console.print("[dim]Press 'q' to quit, 'r' to refresh[/dim]")
    
    try:
        dashboard_app = MLTermDashboard(log_file=log_file, refresh_rate=refresh)
        dashboard_app.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error running dashboard: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_runs(
    log_dir: Path = typer.Option(
        "./logs",
        "--log-dir",
        "-d",
        help="Directory containing log files"
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project",
        "-p",
        help="Filter by project name"
    )
):
    """List available runs."""
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        console.print(f"[red]Error: Log directory '{log_dir}' does not exist[/red]")
        raise typer.Exit(1)
    
    # Find all log files
    pattern = "*.jsonl"
    if project:
        pattern = f"{project}_*.jsonl"
    
    log_files = list(log_dir.glob(pattern))
    
    if not log_files:
        console.print(f"[yellow]No runs found in '{log_dir}'[/yellow]")
        return
    
    # Create table
    table = Table(title="Available Runs")
    table.add_column("Project", style="cyan")
    table.add_column("Run ID", style="magenta")
    table.add_column("File", style="green")
    table.add_column("Size", style="dim")
    table.add_column("Modified", style="dim")
    
    for log_file in sorted(log_files, key=lambda f: f.stat().st_mtime, reverse=True):
        # Parse filename to extract project and run_id
        name = log_file.stem
        if "_" in name:
            project_name, run_id = name.split("_", 1)
        else:
            project_name = "unknown"
            run_id = name
        
        # Get file stats
        stat = log_file.stat()
        size = f"{stat.st_size / 1024:.1f} KB"
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        
        table.add_row(project_name, run_id, log_file.name, size, modified)
    
    console.print(table)


@app.command()
def info(
    log_file: Path = typer.Argument(..., help="Path to log file")
):
    """Show information about a specific run."""
    if not log_file.exists():
        console.print(f"[red]Error: Log file '{log_file}' does not exist[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]Analyzing log file: {log_file.name}[/green]")
    
    # Read and analyze log file
    entries = []
    metadata = {}
    metrics = []
    hyperparameters = {}
    artifacts = []
    
    try:
        with open(log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                    
                    entry_type = entry.get("type")
                    if entry_type == "metadata":
                        metadata = entry
                    elif entry_type == "metric":
                        metrics.append(entry)
                    elif entry_type == "hyperparameters":
                        hyperparameters = entry.get("hyperparameters", {})
                    elif entry_type == "artifact":
                        artifacts.append(entry.get("artifact", {}))
                        
                except json.JSONDecodeError as e:
                    console.print(f"[yellow]Warning: Invalid JSON on line {line_num}: {e}[/yellow]")
                    continue
    except Exception as e:
        console.print(f"[red]Error reading log file: {e}[/red]")
        raise typer.Exit(1)
    
    # Display information
    console.print(f"\n[bold]Run Information[/bold]")
    console.print(f"Project: {metadata.get('project', 'Unknown')}")
    console.print(f"Run ID: {metadata.get('run_id', 'Unknown')}")
    console.print(f"Start Time: {metadata.get('start_time', 'Unknown')}")
    console.print(f"Total Entries: {len(entries)}")
    console.print(f"Metrics Entries: {len(metrics)}")
    
    if hyperparameters:
        console.print(f"\n[bold]Hyperparameters[/bold]")
        for key, value in hyperparameters.items():
            console.print(f"  {key}: {value}")
    
    if artifacts:
        console.print(f"\n[bold]Artifacts[/bold]")
        for artifact in artifacts:
            console.print(f"  {artifact.get('name', 'Unknown')}: {artifact.get('path', 'Unknown')}")
    
    if metrics:
        console.print(f"\n[bold]Recent Metrics[/bold]")
        recent_metrics = metrics[-5:]  # Show last 5 metrics
        for metric in recent_metrics:
            timestamp = metric.get('datetime', metric.get('timestamp', 'Unknown'))
            console.print(f"  [{timestamp}]")
            for key, value in metric.items():
                if key not in ["type", "run_id", "project", "timestamp", "datetime"]:
                    console.print(f"    {key}: {value}")


@app.command()
def init(
    project: str = typer.Argument(..., help="Project name"),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Run ID (auto-generated if not provided)"
    ),
    log_dir: Path = typer.Option(
        "./logs",
        "--log-dir",
        "-d",
        help="Directory to store log files"
    )
):
    """Initialize a new experiment run."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    tracker = Tracker(
        project=project,
        run_id=run_id,
        log_dir=log_dir
    )
    
    console.print(f"[green]Initialized tracker for project '{project}'[/green]")
    console.print(f"Run ID: {tracker.run_id}")
    console.print(f"Log file: {tracker.log_file}")
    console.print(f"\n[dim]Use this in your training script:[/dim]")
    console.print(f"[dim]from mlterm import Tracker[/dim]")
    console.print(f"[dim]tracker = Tracker('{project}', '{tracker.run_id}')[/dim]")


if __name__ == "__main__":
    app()
