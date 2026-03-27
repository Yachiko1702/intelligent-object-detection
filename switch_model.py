#!/usr/bin/env python3
"""
Model switching utility for camera detector.
Allows switching between different YOLO models.
"""

import sys
import os

def switch_model(model_name):
    """Switch to a different YOLO model."""
    models = {
        'nano': 'yolov8n.pt',      # Fastest, 6MB
        'small': 'yolov8s.pt',     # Fast, 22MB  
        'medium': 'yolov8m.pt',    # Balanced, 51MB
        'large': 'yolov8l.pt',     # Accurate, 84MB
        'xlarge': 'yolov8x.pt',    # Most accurate, 130MB
    }
    
    if model_name not in models:
        print(f"Available models: {', '.join(models.keys())}")
        return False
    
    model_path = models[model_name]
    
    # Update app.py
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Find and replace model_path
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'model_path: str =' in line:
            lines[i] = f"    model_path: str = \"{model_path}\"  # {model_name.upper()} model"
            break
    
    with open('app.py', 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Switched to {model_name.upper()} model ({model_path})")
    print(f"   - Size: {os.path.getsize(model_path) / 1024**2:.1f}MB")
    
    # Download model if not exists
    try:
        from ultralytics import YOLO
        YOLO(model_path)
        print(f"   - Model ready")
    except:
        print(f"   - Downloading model...")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python switch_model.py [nano|small|medium|large|xlarge]")
        sys.exit(1)
    
    switch_model(sys.argv[1])
