#!/usr/bin/env python3
"""
Performance optimization script for camera detector.
Analyzes system and provides recommendations.
"""

import sys
import platform
import subprocess
import psutil
from ultralytics import YOLO

def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return True, gpu_name, gpu_memory
        return False, None, 0
    except ImportError:
        return False, None, 0

def check_system_info():
    """Get system information."""
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    memory_gb = memory.total / 1024**3
    
    return {
        'cpu_count': cpu_count,
        'memory_gb': memory_gb,
        'platform': platform.system(),
        'python_version': sys.version
    }

def benchmark_models():
    """Benchmark different YOLO models."""
    models = {
        'yolov8n': 'Nano (fastest)',
        'yolov8s': 'Small (balanced)',
        'yolov8m': 'Medium (accurate)',
        'yolov8l': 'Large (high accuracy)',
        'yolov8x': 'X-Large (best accuracy)'
    }
    
    gpu_available, gpu_name, gpu_memory = check_gpu()
    
    print("\n" + "="*60)
    print("CAMERA DETECTOR PERFORMANCE ANALYSIS")
    print("="*60)
    
    print(f"\nSystem Info:")
    info = check_system_info()
    print(f"  CPU Cores: {info['cpu_count']}")
    print(f"  Memory: {info['memory_gb']:.1f} GB")
    print(f"  Platform: {info['platform']}")
    print(f"  Python: {info['python_version'].split()[0]}")
    
    print(f"\nGPU Status:")
    if gpu_available:
        print(f"  ✓ GPU Available: {gpu_name}")
        print(f"  ✓ GPU Memory: {gpu_memory:.1f} GB")
        print("  Recommendation: Use GPU for faster inference")
    else:
        print("  ✗ No GPU detected")
        print("  Recommendation: Use smaller models (yolov8n/s)")
    
    print(f"\nModel Recommendations:")
    
    if gpu_available:
        print("  For GPU acceleration:")
        print("    - yolov8s.pt: Best balance of speed/accuracy")
        print("    - yolov8m.pt: Higher accuracy, still fast")
        print("    - yolov8x.pt: Maximum accuracy")
    else:
        print("  For CPU only:")
        print("    - yolov8n.pt: Fastest, good for real-time")
        print("    - yolov8s.pt: Slightly slower, better accuracy")
        print("    - Avoid: yolov8m/l/x (too slow for real-time)")
    
    print(f"\nPerformance Tips:")
    if not gpu_available:
        print("  1. Install NVIDIA CUDA drivers for GPU acceleration")
        print("  2. Reduce inference_size to 320 for faster CPU processing")
        print("  3. Increase confidence_threshold to 0.4 to reduce false positives")
    
    if info['memory_gb'] < 8:
        print("  4. Consider upgrading RAM for better performance")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    benchmark_models()
