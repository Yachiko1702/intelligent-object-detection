#!/usr/bin/env python3
"""
Real-time performance monitoring for camera detector.
Shows FPS, inference time, and resource usage.
"""

import psutil
import time
from datetime import datetime

def monitor_performance():
    print("\n" + "="*60)
    print("PERFORMANCE MONITOR")
    print("="*60)
    
    while True:
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        gpu_memory = None
        
        # Try to get GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        except:
            pass
        
        print(f"\r{datetime.now().strftime('%H:%M:%S')} | "
              f"CPU: {cpu_percent:5.1f}% | "
              f"RAM: {memory.percent:5.1f}% | "
              f"GPU: {'N/A' if gpu_memory is None else f'{gpu_memory:.2f}GB'}", 
              end="", flush=True)
        
        time.sleep(1)

if __name__ == "__main__":
    try:
        monitor_performance()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
