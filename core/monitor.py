# core/monitor.py

import psutil
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)

class Monitor:
    """System performance monitor with smoothed metrics."""
    
    def __init__(self, window_size=30):
        """Initialize monitor.
        
        Args:
            window_size: Number of frames for averaging
        """
        self.prev = time.time()
        self.window_size = window_size
        self.fps_history = deque(maxlen=window_size)
        self.latency_history = deque(maxlen=window_size)

    def get_metrics(self, latency):
        """Get system metrics with smoothing.
        
        Args:
            latency: Inference latency in seconds
            
        Returns:
            Dict with system metrics
        """
        now = time.time()
        delta = now - self.prev
        self.prev = now
        
        # Calculate FPS with smoothing
        if delta > 0:
            fps = 1.0 / delta
            self.fps_history.append(fps)
        
        # Store latency
        self.latency_history.append(latency * 1000)  # Convert to ms
        
        # Calculate averages
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        avg_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0
        
        ram = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.01)

        return {
            "cpu": round(cpu, 1),
            "ram": round(ram.percent, 1),
            "fps": round(avg_fps, 1),
            "latency": round(avg_latency, 1)
        }