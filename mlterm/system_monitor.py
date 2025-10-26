"""
System monitoring utilities for CPU, GPU, and memory usage.
"""

import time
from typing import Dict, List, Optional, Tuple
import psutil

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class SystemMonitor:
    """
    Monitor system resources including CPU, memory, and GPU usage.
    """
    
    def __init__(self):
        """Initialize the system monitor."""
        self.gpu_available = GPU_AVAILABLE
        self._last_cpu_times = None
        self._last_cpu_times_timestamp = None
    
    def get_cpu_info(self) -> Dict[str, float]:
        """
        Get CPU usage information.
        
        Returns:
            Dictionary with CPU usage statistics
        """
        # Get CPU percentage (averaged over all cores)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get per-core CPU usage
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # Get CPU frequency
        cpu_freq = psutil.cpu_freq()
        cpu_freq_current = cpu_freq.current if cpu_freq else 0
        cpu_freq_max = cpu_freq.max if cpu_freq else 0
        
        # Get CPU count
        cpu_count = psutil.cpu_count()
        cpu_count_logical = psutil.cpu_count(logical=True)
        
        return {
            "cpu_percent": cpu_percent,
            "cpu_per_core": cpu_per_core,
            "cpu_freq_current": cpu_freq_current,
            "cpu_freq_max": cpu_freq_max,
            "cpu_count": cpu_count,
            "cpu_count_logical": cpu_count_logical
        }
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get memory usage information.
        
        Returns:
            Dictionary with memory usage statistics
        """
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            "memory_total": memory.total,
            "memory_available": memory.available,
            "memory_used": memory.used,
            "memory_percent": memory.percent,
            "memory_free": memory.free,
            "swap_total": swap.total,
            "swap_used": swap.used,
            "swap_percent": swap.percent,
            "swap_free": swap.free
        }
    
    def get_disk_info(self) -> Dict[str, float]:
        """
        Get disk usage information.
        
        Returns:
            Dictionary with disk usage statistics
        """
        disk_usage = psutil.disk_usage('/')
        
        return {
            "disk_total": disk_usage.total,
            "disk_used": disk_usage.used,
            "disk_free": disk_usage.free,
            "disk_percent": (disk_usage.used / disk_usage.total) * 100
        }
    
    def get_gpu_info(self) -> List[Dict[str, float]]:
        """
        Get GPU usage information.
        
        Returns:
            List of dictionaries with GPU statistics for each GPU
        """
        if not self.gpu_available:
            return []
        
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return []
                
            gpu_info = []
            
            for gpu in gpus:
                gpu_info.append({
                    "gpu_id": gpu.id,
                    "gpu_name": gpu.name,
                    "gpu_memory_total": gpu.memoryTotal,
                    "gpu_memory_used": gpu.memoryUsed,
                    "gpu_memory_free": gpu.memoryFree,
                    "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    "gpu_utilization": gpu.load * 100,
                    "gpu_temperature": gpu.temperature
                })
            
            return gpu_info
            
        except Exception:
            # GPU monitoring failed, return empty list
            return []
    
    def get_network_info(self) -> Dict[str, float]:
        """
        Get network usage information.
        
        Returns:
            Dictionary with network statistics
        """
        net_io = psutil.net_io_counters()
        
        return {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
            "errin": net_io.errin,
            "errout": net_io.errout,
            "dropin": net_io.dropin,
            "dropout": net_io.dropout
        }
    
    def get_system_info(self) -> Dict[str, any]:
        """
        Get comprehensive system information.
        
        Returns:
            Dictionary with all system statistics
        """
        return {
            "timestamp": time.time(),
            "cpu": self.get_cpu_info(),
            "memory": self.get_memory_info(),
            "disk": self.get_disk_info(),
            "gpu": self.get_gpu_info(),
            "network": self.get_network_info(),
            "gpu_available": self.gpu_available
        }
    
    def get_process_info(self, pid: Optional[int] = None) -> Dict[str, any]:
        """
        Get process-specific information.
        
        Args:
            pid: Process ID to monitor (current process if None)
            
        Returns:
            Dictionary with process statistics
        """
        if pid is None:
            process = psutil.Process()
        else:
            process = psutil.Process(pid)
        
        try:
            return {
                "pid": process.pid,
                "name": process.name(),
                "cpu_percent": process.cpu_percent(),
                "memory_info": process.memory_info()._asdict(),
                "memory_percent": process.memory_percent(),
                "num_threads": process.num_threads(),
                "create_time": process.create_time(),
                "status": process.status()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {"error": "Process not accessible"}
