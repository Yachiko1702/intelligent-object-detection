#!/usr/bin/env python3
"""
YOLOv6 integration for camera detector.
YOLOv6 offers better accuracy and speed than YOLOv8.
"""

from ultralytics import YOLO

class YOLOv6Model:
    """YOLOv6 model wrapper for better performance."""
    
    def __init__(self, model_path="yolov6s.pt"):
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        
    def predict(self, frame):
        """Run inference with YOLOv6."""
        results = self.model(frame, verbose=False)
        return results

# Model options based on performance data:
YOLOV6_MODELS = {
    'nano': {
        'path': 'yolov6n.pt',
        'mAP': 40.9,
        'speed_cpu': 1.7,  # ms
        'speed_gpu': 2.4,  # ms
        'params': 5.4,     # M
        'description': 'Fastest, good for real-time'
    },
    'small': {
        'path': 'yolov6s.pt', 
        'mAP': 48.6,
        'speed_cpu': 87.2,   # ms
        'speed_gpu': 2.5,    # ms
        'params': 9.5,       # M
        'description': 'Best balance of speed and accuracy'
    },
    'medium': {
        'path': 'yolov6m.pt',
        'mAP': 53.1,
        'speed_cpu': 220.0,  # ms
        'speed_gpu': 4.7,    # ms
        'params': 20.4,      # M
        'description': 'High accuracy, still fast'
    },
    'large': {
        'path': 'yolov6l.pt',
        'mAP': 55.0,
        'speed_cpu': 286.2,  # ms
        'speed_gpu': 6.2,    # ms
        'params': 24.8,      # M
        'description': 'Very accurate'
    },
    'xlarge': {
        'path': 'yolov6x.pt',
        'mAP': 57.5,
        'speed_cpu': 525.8,  # ms
        'speed_gpu': 11.8,   # ms
        'params': 55.7,      # M
        'description': 'Maximum accuracy'
    }
}

def get_best_model_for_fps(target_fps=30):
    """Get the best YOLOv6 model for target FPS."""
    if target_fps >= 30:
        return YOLOV6_MODELS['small']  # 2.5ms GPU = 400 FPS potential
    elif target_fps >= 20:
        return YOLOV6_MODELS['medium']  # 4.7ms GPU = 200 FPS potential
    else:
        return YOLOV6_MODELS['large']   # 6.2ms GPU = 160 FPS potential
