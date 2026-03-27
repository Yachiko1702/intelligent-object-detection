#!/usr/bin/env python3
"""Lightweight Camera Object Detector - No Heavy Dependencies"""

import os
import sys
import json
import base64
import logging
import threading
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try imports with fallback
try:
    logger.info("Importing libraries...")
    import cv2
    import numpy as np
    from flask import Flask, render_template, request, jsonify
    from flask_cors import CORS
    from flask_socketio import SocketIO, emit
    logger.info("Basic libraries imported")
except Exception as e:
    logger.error(f"Basic import error: {e}")
    sys.exit(1)

# Flask setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'camera-detector-secret-key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

@dataclass
class FrameStats:
    fps: float
    inference_time_ms: float
    total_detections: int
    detected_classes: Dict[str, int]
    timestamp: str
    
    def to_dict(self):
        return {
            'fps': round(self.fps, 1),
            'inference_time_ms': round(self.inference_time_ms, 1),
            'total_detections': self.total_detections,
            'detected_classes': self.detected_classes,
            'timestamp': self.timestamp
        }

class SimpleDetector:
    """Simple detector with motion detection"""
    
    def __init__(self):
        self.running = False
        self.frame_count = 0
        self.fps_history = deque(maxlen=10)
        self.target_frame_time = 1.0 / 30.0
        self.prev_frame = None
        self.motion_boxes = []
        
    def predict(self, frame):
        """Simple motion detection"""
        start_time = time.perf_counter()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return [], 0.0
        
        # Calculate difference
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create detection boxes for motion
        detections = []
        self.motion_boxes = []
        
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            
            (x, y, w, h) = cv2.boundingRect(contour)
            
            # Create a simple detection result
            detection = {
                'bbox': (x, y, x + w, y + h),
                'label': 'Motion',
                'color': (0, 255, 0),
                'confidence': 0.8
            }
            detections.append(detection)
            self.motion_boxes.append((x, y, x + w, y + h))
        
        self.prev_frame = gray
        inference_time = (time.perf_counter() - start_time) * 1000
        
        return detections, inference_time

class CameraManager:
    """Optimized camera manager"""
    
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        
    def start(self):
        """Start camera with multiple attempts"""
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY, cv2.CAP_MSMF]
        
        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(self.camera_index, backend)
                if self.cap.isOpened():
                    time.sleep(0.5)  # Warmup
                    
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        logger.info(f"Camera started with backend {backend}")
                        self.running = True
                        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
                        self.thread.start()
                        return True
                    else:
                        self.cap.release()
            except Exception as e:
                logger.warning(f"Camera attempt failed: {e}")
                if self.cap:
                    self.cap.release()
        
        logger.error("Could not start camera")
        return False
    
    def _capture_loop(self):
        """Simple capture loop"""
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        break
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    consecutive_failures = 0
                    with self.lock:
                        self.frame = frame
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        break
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Capture error: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                time.sleep(0.01)
    
    def get_frame(self):
        """Get latest frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        """Stop camera"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        logger.info("Camera stopped")

class DetectorApp:
    """Main application without YOLO"""
    
    def __init__(self):
        self.camera = None
        self.detector = SimpleDetector()
        self.running = False
        self.detection_thread = None
        self.frame_stats = None
        self.stats_history = deque(maxlen=100)
        self.frame_count = 0
        
    def initialize(self):
        """Initialize components"""
        try:
            # Start camera
            self.camera = CameraManager(0)
            if not self.camera.start():
                return False
            
            # Start detection loop
            self.running = True
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
            logger.info("App initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Init error: {e}")
            return False
    
    def _detection_loop(self):
        """Simple detection loop"""
        last_frame_time = time.perf_counter()
        
        while self.running:
            try:
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                self.frame_count += 1
                
                # Flip frame
                frame = cv2.flip(frame, 1)
                
                # Mock inference (motion detection)
                start_time = time.perf_counter()
                detections, inference_time = self.detector.predict(frame)
                inference_time = (time.perf_counter() - start_time) * 1000
                
                # Draw motion boxes
                annotated_frame = self._draw_detections(frame, detections)
                
                # Calculate FPS
                current_time = time.perf_counter()
                frame_time = current_time - last_frame_time
                
                if frame_time > 0:
                    instant_fps = 1.0 / frame_time
                    self.detector.fps_history.append(instant_fps)
                    if len(self.detector.fps_history) > 0:
                        smoothed_fps = sum(self.detector.fps_history) / len(self.detector.fps_history)
                    else:
                        smoothed_fps = 30.0
                else:
                    smoothed_fps = 30.0
                
                # FPS control
                if frame_time < self.detector.target_frame_time:
                    sleep_time = self.detector.target_frame_time - frame_time
                    if sleep_time > 0.001:
                        time.sleep(sleep_time)
                
                last_frame_time = time.perf_counter()
                
                # Update stats
                self.frame_stats = FrameStats(
                    fps=smoothed_fps,
                    inference_time_ms=inference_time,
                    total_detections=len(detections),
                    detected_classes={},
                    timestamp=datetime.now().isoformat()
                )
                self.stats_history.append(self.frame_stats)
                
                # Emit frame
                try:
                    frame_data = {
                        'frame': self._encode_frame(annotated_frame),
                        'stats': self.frame_stats.to_dict(),
                        'detections': [{'class_name': d['label'], 'confidence': d['confidence'], 'bbox': d['bbox']} for d in detections]
                    }
                    socketio.emit('frame', frame_data)
                except Exception as e:
                    logger.error(f"Socket error: {e}")
                
                # Log less frequently
                if self.frame_count % 120 == 0:
                    logger.info(f"FPS: {smoothed_fps:.1f} | Camera Active")
                    
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                time.sleep(0.01)
                continue
    
    def _draw_detections(self, frame, detections):
        """Draw motion detection boxes"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = det['color']
            label = det['label']
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_y = max(y1 - 10, label_size[1] + 5)
            
            cv2.rectangle(
                annotated,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0] + 10, label_y + 5),
                color,
                -1
            )
            
            cv2.putText(
                annotated, label, (x1 + 5, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                cv2.LINE_AA
            )
        
        # Add status text
        status_text = f"Motion Detection Active - {len(detections)} regions"
        cv2.putText(
            annotated, status_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            cv2.LINE_AA
        )
        
        return annotated
    
    def _encode_frame(self, frame):
        """Encode frame to base64"""
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Frame encoding error: {e}")
            return ""
    
    def get_stats_summary(self):
        """Get stats"""
        if not self.stats_history:
            return {}
        
        recent = list(self.stats_history)[-30:]
        avg_fps = sum(s.fps for s in recent) / len(recent)
        
        return {
            'average_fps': round(avg_fps, 1),
            'average_inference_ms': 0.0,
            'total_detections_by_class': {},
            'frames_processed': len(self.stats_history)
        }
    
    def shutdown(self):
        """Shutdown"""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        if self.camera:
            self.camera.stop()
        logger.info("App shutdown")

# Global app
detector_app = DetectorApp()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    return jsonify(detector_app.get_stats_summary())

@app.route('/api/current_model')
def get_current_model():
    return jsonify({
        'model_type': 'simple',
        'model_path': 'none',
        'display_name': 'Camera Test Mode (No AI)'
    })

@app.route('/api/config', methods=['POST'])
def update_config():
    return jsonify({'success': True})

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")
    emit('connected', {'message': 'Connected to detector'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("CAMERA DETECTOR SERVER STARTING")
    print("="*60)
    print("✅ Mode: Camera Test (No AI Dependencies)")
    print("✅ Open in browser:")
    print("   http://127.0.0.1:5000")
    print("   http://localhost:5000")
    print("="*60 + "\n")
    
    # Initialize
    if not detector_app.initialize():
        logger.error("Failed to initialize")
        sys.exit(1)
    
    try:
        logger.info("Starting server...")
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        detector_app.shutdown()
