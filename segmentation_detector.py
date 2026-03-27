#!/usr/bin/env python3
"""
Segmentation integration for precise object masking.
Uses YOLOv6-seg for pixel-perfect object detection.
"""

import cv2
import numpy as np
from ultralytics import YOLO

class SegmentationDetector:
    """Object segmentation using YOLOv6-seg."""
    
    def __init__(self, model_path="yolov6s-seg.pt"):
        self.model = YOLO(model_path)
        
    def detect_segments(self, frame):
        """Detect objects with segmentation masks."""
        results = self.model(frame, verbose=False)
        segments = []
        
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
                classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else None
                confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else None
                
                for i, mask in enumerate(masks):
                    segment_data = {
                        'mask': mask,
                        'bbox': boxes[i].tolist() if boxes is not None else None,
                        'class_id': int(classes[i]) if classes is not None else None,
                        'confidence': float(confidences[i]) if confidences is not None else None
                    }
                    segments.append(segment_data)
        
        return segments
    
    def draw_segments(self, frame, segments, alpha=0.4):
        """Draw segmentation masks on frame."""
        overlay = frame.copy()
        
        for segment in segments:
            mask = segment['mask']
            confidence = segment['confidence']
            
            if confidence > 0.5:
                # Resize mask to frame size
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                
                # Create colored overlay
                colored_mask = np.zeros_like(frame)
                color = np.random.randint(0, 255, 3)
                colored_mask[mask_resized > 0.5] = color
                
                # Blend with original frame
                cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0, overlay)
        
        return overlay
    
    def get_object_pixels(self, frame, segments):
        """Get pixel coordinates for each detected object."""
        objects = []
        
        for segment in segments:
            mask = segment['mask']
            class_id = segment['class_id']
            
            # Find contours in mask
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
            
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            object_data = {
                'class_id': class_id,
                'contours': contours,
                'pixel_count': np.sum(mask_resized > 0.5),
                'centroid': self._get_centroid(mask_resized)
            }
            objects.append(object_data)
        
        return objects
    
    def _get_centroid(self, mask):
        """Calculate centroid of mask."""
        y_coords, x_coords = np.where(mask > 0.5)
        if len(x_coords) > 0:
            return (int(np.mean(x_coords)), int(np.mean(y_coords)))
        return (0, 0)
