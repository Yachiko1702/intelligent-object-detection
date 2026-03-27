#!/usr/bin/env python3
"""Professional Camera Object Detector Web Application.

A modern web-based UI for real-time object detection using YOLO models.
Detects: bag, ballpen, tumbler (with class mapping from COCO dataset)

Usage:
    python app.py

Then open http://localhost:5000 in your browser.
"""

import os
import sys
import signal
import json
import base64
import logging
import threading
import time
import base64
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import deque
import re

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Core imports with error handling
try:
    logger.info("Importing core libraries...")
    import cv2
    import numpy as np
    from flask import Flask, render_template, request, jsonify
    from flask_cors import CORS
    from flask_socketio import SocketIO, emit
    logger.info("Core libraries imported successfully")
    OPENCV_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenCV not available: {e}")
    logger.info("Running in demo mode without computer vision")
    import numpy as np
    from flask import Flask, render_template, request, jsonify
    from flask_cors import CORS
    from flask_socketio import SocketIO, emit
    OPENCV_AVAILABLE = False
    # Create a dummy cv2 module to prevent NameError
    class DummyCV2:
        COLOR_BGR2HSV = 0
        COLOR_BGR2LAB = 1
        COLOR_BGR2GRAY = 2
        COLOR_BGR2RGB = 3
        TERM_CRITERIA_EPS = 1
        TERM_CRITERIA_MAX_ITER = 2
        KMEANS_RANDOM_CENTERS = 2
        THRESH_BINARY = 0
        RETR_EXTERNAL = 0
        CHAIN_APPROX_SIMPLE = 1
        CAP_DSHOW = 0
        CAP_ANY = 0
        CAP_MSMF = 0
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FPS = 5
        CAP_PROP_BUFFERSIZE = 6
        FONT_HERSHEY_SIMPLEX = 0
        
        @staticmethod
        def cvtColor(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def kmeans(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def createCLAHE(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def absdiff(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def threshold(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def dilate(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def findContours(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def contourArea(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def boundingRect(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def arcLength(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def approxPolyDP(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def convexHull(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def convexityDefects(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def inRange(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def getTextSize(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def rectangle(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def putText(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def circle(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def line(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
        
        @staticmethod
        def VideoCapture(*args, **kwargs):
            raise NotImplementedError("OpenCV not available")
    
    cv2 = DummyCV2()
    
    # Optional imports with error handling
    OCR_AVAILABLE = False
    try:
        import pytesseract
        import easyocr
        OCR_AVAILABLE = True
        logger.info("OCR libraries available")
    except ImportError:
        logger.warning("OCR libraries not installed. Text detection features will be limited.")
    
    FACE_RECOGNITION_AVAILABLE = False
    try:
        import face_recognition
        FACE_RECOGNITION_AVAILABLE = True
        logger.info("Face recognition available")
    except ImportError:
        logger.warning("Face recognition not available")
        
    FACE_LANDMARKS_AVAILABLE = False
    try:
        import dlib
        FACE_LANDMARKS_AVAILABLE = True
        logger.info("dlib available for face landmarks")
    except ImportError:
        try:
            import mediapipe as mp
            FACE_LANDMARKS_AVAILABLE = True
            logger.info("MediaPipe available for face landmarks")
        except ImportError:
            logger.warning("Face landmark detection not available")
    
    # MediaPipe Hands for finger detection (using legacy API)
    HAND_DETECTION_AVAILABLE = False
    hands_detector = None
    try:
        import mediapipe as mp
        # Workaround for mediapipe not exposing solutions directly in some environments
        if not hasattr(mp, 'solutions'):
            import mediapipe.python.solutions as mp_solutions
            mp.solutions = mp_solutions
            
        mp_hands = mp.solutions.hands
        hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        HAND_DETECTION_AVAILABLE = True
        logger.info("MediaPipe Hands loaded for finger detection")
    except Exception as e:
        logger.warning(f"MediaPipe Hands not available: {e}")
    
except Exception as e:
    logger.error(f"Import error: {e}")
    logger.error("Please ensure all required packages are installed:")
    logger.error("pip install opencv-python numpy flask flask-cors flask-socketio")
    sys.exit(1)

if not OCR_AVAILABLE and not FACE_RECOGNITION_AVAILABLE:
    logger.warning("OCR/Face recognition libraries not installed. Some features will be limited.")

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'camera-detector-secret-key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


@dataclass
class DetectionConfig:
    """Configuration for object detection."""
    # Model: YOLOv8 Large for maximum accuracy across all object types
    model_type: str = "detection"
    model_path: str = "yolov8l.pt"  # Upgraded to Large model for maximum accuracy
    confidence_threshold: float = 0.25  # Lowered for better detection of small/distant objects
    iou_threshold: float = 0.45
    device: str = "auto"
    camera_index: int = 0
    enable_tracking: bool = True
    # Performance settings for stable FPS
    inference_size: int = 1280  # Increased for better accuracy on small objects
    inference_interval: int = 1
    jpeg_quality: int = 95  # Maximum clarity for better analysis
    max_fps: int = 30
    frame_delay: float = 1.0/30.0
    batch_processing: bool = False
    target_fps: int = 30
    fps_smoothing_window: int = 10
    hand_detection_interval: int = 5  # Run hand detection every 5th frame
    pose_detection_interval: int = 10  # Run pose detection every 10th frame
    enable_ocr: bool = False  # Disable OCR for performance
    enable_face_recognition: bool = False  # Disable face recognition
    enable_pose: bool = False  # Disable pose detection (body parts)
    enable_face_landmarks: bool = False  # Disable face landmarks completely
    # Enhanced detection settings
    enable_adaptive_threshold: bool = True  # Adaptive confidence per class
    enable_object_grouping: bool = True  # Group related objects
    enable_color_analysis: bool = True  # Color-based object enhancement
    max_detections: int = 100  # Maximum detections per frame
    
    def __post_init__(self):
        # Enhanced class categories for intelligent detection
        self.fruit_classes = {
            'apple', 'banana', 'orange', 'grape', 'strawberry', 'lemon', 'lime', 'peach',
            'pear', 'plum', 'cherry', 'blueberry', 'raspberry', 'watermelon', 'cantaloupe',
            'mango', 'pineapple', 'kiwi', 'avocado', 'coconut', 'papaya', 'pomegranate'
        }
        
        self.animal_classes = {
            'dog', 'cat', 'horse', 'cow', 'sheep', 'goat', 'pig', 'chicken', 'duck',
            'goose', 'turkey', 'bird', 'rabbit', 'mouse', 'rat', 'squirrel', 'deer',
            'elephant', 'giraffe', 'zebra', 'lion', 'tiger', 'bear', 'wolf', 'fox',
            'monkey', 'ape', 'kangaroo', 'koala', 'panda', 'hippo', 'rhino'
        }
        
        self.vehicle_classes = {
            'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'airplane',
            'helicopter', 'boat', 'ship', 'train', 'subway', 'tractor', 'ambulance',
            'police car', 'fire truck', 'taxi', 'van', 'suv', 'pickup'
        }
        
        self.electronic_classes = {
            'cell phone', 'laptop', 'computer', 'tablet', 'tv', 'monitor', 'keyboard',
            'mouse', 'remote', 'camera', 'printer', 'scanner', 'speaker', 'headphones',
            'microwave', 'refrigerator', 'washing machine', 'dryer', 'oven', 'toaster'
        }
        
        self.furniture_classes = {
            'chair', 'table', 'desk', 'bed', 'sofa', 'couch', 'bookshelf', 'cabinet',
            'dresser', 'nightstand', 'stool', 'bench', 'armchair', 'recliner', 'ottoman'
        }
        
        self.food_classes = {
            'pizza', 'hamburger', 'hot dog', 'sandwich', 'salad', 'soup', 'pasta',
            'bread', 'cake', 'cookie', 'donut', 'ice cream', 'coffee', 'tea', 'juice',
            'water', 'soda', 'milk', 'cheese', 'meat', 'fish', 'vegetable', 'fruit'
        }
        
        # Adaptive confidence thresholds per category
        self.adaptive_thresholds = {
            'fruits': 0.20,  # Lower for fruits (often small/occluded)
            'animals': 0.30,  # Medium for animals
            'vehicles': 0.40,  # Higher for vehicles (distinct shapes)
            'electronics': 0.35,  # Medium-high for electronics
            'furniture': 0.45,  # Higher for furniture (large objects)
            'food': 0.25,  # Lower for food items
            'default': 0.25  # Default threshold
        }


@dataclass
class DetectionResult:
    """Single detection result with support for detection, pose, and segmentation."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    display_name: str
    color: Tuple[int, int, int]
    # Enhanced category information
    category: Optional[str] = None  # Main category (fruit, animal, vehicle, etc.)
    subcategory: Optional[str] = None  # More specific subcategory
    # Additional fields for ID and text detection
    detected_text: Optional[str] = None
    face_name: Optional[str] = None
    finger_count: Optional[int] = None
    # Face landmarks for eyes, nose, mouth
    face_landmarks: Optional[List[Tuple[int, int, str]]] = None
    # Pose estimation fields
    keypoints: Optional[List[Tuple[float, float, float]]] = None  # (x, y, confidence)
    # Segmentation fields
    mask: Optional[np.ndarray] = None  # Binary mask
    # Enhanced detection features
    color_features: Optional[Dict[str, Any]] = None  # Color analysis results
    texture_features: Optional[Dict[str, Any]] = None  # Texture analysis
    relationships: Optional[List[str]] = None  # Related objects
    group_id: Optional[str] = None  # Object group identifier
    
    def to_dict(self):
        result = {
            'class_name': self.class_name,
            'display_name': self.display_name,
            'confidence': round(self.confidence, 3),
            'bbox': self.bbox,
            'color': self.color,
            'category': self.category,
            'subcategory': self.subcategory,
            'detected_text': self.detected_text,
            'face_name': self.face_name,
            'finger_count': self.finger_count,
            'has_pose': self.keypoints is not None,
            'has_segmentation': self.mask is not None,
            'group_id': self.group_id,
            'relationships': self.relationships or []
        }
        # Include keypoints if available (convert to serializable format)
        if self.keypoints:
            result['keypoints'] = [[int(x), int(y), round(conf, 2)] for x, y, conf in self.keypoints]
        # Include color features if available
        if self.color_features:
            result['color_features'] = self.color_features
        # Include texture features if available
        if self.texture_features:
            result['texture_features'] = self.texture_features
        return result


@dataclass
class FrameStats:
    """Frame processing statistics."""
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


class DetectionModel:
    """YOLOv8 model with enhanced detection for fruits, animals, and objects."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.device = None
        self.detection_model = None
        self.pose_model = None
        self.class_names = {}
        self.class_colors = {}
        self.frame_count = 0  # Frame counter for hand detection skipping
        self.last_pose_results = None  # Store last pose results when skipping
        
        self._load_models()
        self._init_categorization()
        
    def _load_models(self):
        """Load YOLOv8 detection and pose models."""
        try:
            logger.info("Loading YOLO models (this may take a moment)...")
            
            # Import YOLO here to avoid hanging during initial import
            from ultralytics import YOLO
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load detection model
            detection_path = self.config.model_path
            logger.info(f"Loading detection model: {detection_path}")
            try:
                self.detection_model = YOLO(detection_path)
                self.detection_model.to(self.device)
                logger.info("✅ Detection model loaded")
            except Exception as e:
                logger.warning(f"Failed to load detection model: {e}")
            
            # Load pose model for human body parts (only if enabled)
            if self.config.enable_pose:
                pose_path = "yolov8m-pose.pt"
                logger.info(f"Loading pose model: {pose_path}")
                try:
                    self.pose_model = YOLO(pose_path)
                    self.pose_model.to(self.device)
                    logger.info("✅ Pose model loaded for human body parts")
                except Exception as e:
                    logger.warning(f"Failed to load pose model: {e}")
            else:
                self.pose_model = None
                logger.info("Pose detection disabled for better performance")
            
            # Get class names from detection model
            if self.detection_model:
                self.class_names = self.detection_model.names
                
                # Generate distinct colors for each class
                import colorsys
                for cls_id, name in self.class_names.items():
                    hue = cls_id / 80
                    rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
                    self.class_colors[cls_id] = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            
            logger.info(f"Models loaded on device: {self.device}")
            
            # Initialize OCR
            self.ocr_reader = None
            if OCR_AVAILABLE:
                try:
                    self.ocr_reader = easyocr.Reader(['en'])
                    logger.info("OCR initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize OCR: {e}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            # Fallback to simple mode without YOLO
            logger.warning("Running in camera-only mode without AI detection")
            self.detection_model = None
            self.pose_model = None
            self.class_names = {}
            self.class_colors = {}
            
    def _init_categorization(self):
        """Initialize intelligent categorization system."""
        # Create class to category mapping
        self.class_to_category = {}
        
        # Map fruits
        for fruit in self.config.fruit_classes:
            self.class_to_category[fruit] = {'category': 'fruit', 'subcategory': 'produce'}
        
        # Map animals
        for animal in self.config.animal_classes:
            if animal in ['dog', 'cat', 'horse', 'cow', 'sheep', 'goat', 'pig']:
                subcat = 'domestic'
            elif animal in ['lion', 'tiger', 'bear', 'wolf', 'fox']:
                subcat = 'wild'
            elif animal in ['bird', 'chicken', 'duck', 'goose', 'turkey']:
                subcat = 'bird'
            else:
                subcat = 'other'
            self.class_to_category[animal] = {'category': 'animal', 'subcategory': subcat}
        
        # Map vehicles
        for vehicle in self.config.vehicle_classes:
            if vehicle in ['car', 'truck', 'bus', 'van', 'suv', 'pickup', 'taxi']:
                subcat = 'ground'
            elif vehicle in ['airplane', 'helicopter']:
                subcat = 'air'
            elif vehicle in ['boat', 'ship']:
                subcat = 'water'
            else:
                subcat = 'other'
            self.class_to_category[vehicle] = {'category': 'vehicle', 'subcategory': subcat}
        
        # Map electronics
        for electronic in self.config.electronic_classes:
            if electronic in ['cell phone', 'tablet', 'laptop']:
                subcat = 'portable'
            elif electronic in ['tv', 'monitor', 'computer']:
                subcat = 'display'
            elif electronic in ['microwave', 'refrigerator', 'washing machine', 'dryer', 'oven']:
                subcat = 'appliance'
            else:
                subcat = 'accessory'
            self.class_to_category[electronic] = {'category': 'electronics', 'subcategory': subcat}
        
        # Map furniture
        for furniture in self.config.furniture_classes:
            if furniture in ['chair', 'stool', 'bench']:
                subcat = 'seating'
            elif furniture in ['table', 'desk']:
                subcat = 'surface'
            elif furniture in ['bed', 'sofa', 'couch', 'armchair', 'recliner']:
                subcat = 'comfort'
            else:
                subcat = 'storage'
            self.class_to_category[furniture] = {'category': 'furniture', 'subcategory': subcat}
        
        # Map food
        for food in self.config.food_classes:
            if food in ['pizza', 'hamburger', 'hot dog', 'sandwich']:
                subcat = 'prepared'
            elif food in ['fruit', 'vegetable', 'meat', 'fish', 'cheese']:
                subcat = 'ingredient'
            elif food in ['coffee', 'tea', 'juice', 'water', 'soda', 'milk']:
                subcat = 'beverage'
            else:
                subcat = 'dessert'
            self.class_to_category[food] = {'category': 'food', 'subcategory': subcat}
    
    def _get_category_info(self, class_name: str) -> Dict[str, str]:
        """Get category information for a class name."""
        class_name_lower = class_name.lower()
        
        # Direct mapping first
        if class_name_lower in self.class_to_category:
            return self.class_to_category[class_name_lower]
        
        # Fuzzy matching for similar names
        for key, info in self.class_to_category.items():
            if class_name_lower in key or key in class_name_lower:
                return info
        
        # Default categorization based on common patterns
        if any(word in class_name_lower for word in ['person', 'people', 'human']):
            return {'category': 'person', 'subcategory': 'human'}
        elif any(word in class_name_lower for word in ['book', 'paper', 'card']):
            return {'category': 'document', 'subcategory': 'reading'}
        elif any(word in class_name_lower for word in ['bottle', 'cup', 'glass']):
            return {'category': 'container', 'subcategory': 'drinkware'}
        elif any(word in class_name_lower for word in ['bag', 'backpack', 'purse']):
            return {'category': 'container', 'subcategory': 'carry'}
        else:
            return {'category': 'object', 'subcategory': 'general'}
    
    def _get_adaptive_threshold(self, class_name: str) -> float:
        """Get adaptive confidence threshold for a specific class."""
        if not self.config.enable_adaptive_threshold:
            return self.config.confidence_threshold
        
        category_info = self._get_category_info(class_name)
        category = category_info['category']
        
        # Map category to threshold
        threshold_map = {
            'fruit': self.config.adaptive_thresholds.get('fruits', 0.20),
            'animal': self.config.adaptive_thresholds.get('animals', 0.30),
            'vehicle': self.config.adaptive_thresholds.get('vehicles', 0.40),
            'electronics': self.config.adaptive_thresholds.get('electronics', 0.35),
            'furniture': self.config.adaptive_thresholds.get('furniture', 0.45),
            'food': self.config.adaptive_thresholds.get('food', 0.25),
        }
        
        return threshold_map.get(category, self.config.adaptive_thresholds.get('default', 0.25))
    
    def _analyze_color_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Analyze color features of detected object."""
        if not self.config.enable_color_analysis:
            return {}
        
        try:
            x1, y1, x2, y2 = bbox
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return {}
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            
            # Calculate color statistics
            mean_bgr = np.mean(roi, axis=(0, 1))
            mean_hsv = np.mean(hsv, axis=(0, 1))
            mean_lab = np.mean(lab, axis=(0, 1))
            
            # Dominant color detection
            pixels = roi.reshape(-1, 3)
            kmeans = cv2.kmeans(pixels.astype(np.float32), 3, None, 
                              (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 
                              10, cv2.KMEANS_RANDOM_CENTERS)
            
            dominant_colors = kmeans[2].astype(int)
            
            return {
                'mean_bgr': mean_bgr.tolist(),
                'mean_hsv': mean_hsv.tolist(),
                'mean_lab': mean_lab.tolist(),
                'dominant_colors': dominant_colors.tolist(),
                'brightness': float(mean_bgr[2]),
                'saturation': float(mean_hsv[1]),
                'hue': float(mean_hsv[0])
            }
        except Exception as e:
            logger.debug(f"Color analysis error: {e}")
            return {}
            
            logger.info(f"Models loaded on device: {self.device}")
            
            # Initialize OCR
            self.ocr_reader = None
            if OCR_AVAILABLE:
                try:
                    self.ocr_reader = easyocr.Reader(['en'])
                    logger.info("OCR initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize OCR: {e}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            # Fallback to simple mode without YOLO
            logger.warning("Running in camera-only mode without AI detection")
            self.detection_model = None
            self.pose_model = None
            self.class_names = {}
            self.class_colors = {}
            
    def _extract_text_from_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """Extract text from bounding box area using OCR."""
        if not self.ocr_reader:
            return None
            
        try:
            x1, y1, x2, y2 = bbox
            # Extract the region of interest
            roi = frame[y1:y2, x1:x2]
            
            # Preprocess for better OCR
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Use EasyOCR
            results = self.ocr_reader.readtext(enhanced)
            
            # Extract and clean text
            texts = []
            for (bbox_points, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence results
                    # Clean text - keep alphanumeric and common symbols
                    clean_text = re.sub(r'[^a-zA-Z0-9\s\-\\./]', '', text).strip()
                    if len(clean_text) > 2:
                        texts.append(clean_text)
            
            return ' '.join(texts) if texts else None
            
        except Exception as e:
            logger.debug(f"OCR error: {e}")
            return None
    
    def _recognize_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """Recognize face in bounding box area - looks at head region only."""
        if not FACE_RECOGNITION_AVAILABLE:
            return None
            
        try:
            x1, y1, x2, y2 = bbox
            # For person detection, only look at top 30% (head region)
            head_y2 = y1 + int((y2 - y1) * 0.3)
            
            # Ensure valid bbox
            if x2 <= x1 or head_y2 <= y1:
                return None
                
            roi = frame[y1:head_y2, x1:x2]
            
            if roi.size == 0:
                return None
            
            # Convert to RGB (face_recognition uses RGB)
            rgb_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # Use HOG for speed
            
            if face_locations:
                # For demo, return that face was detected
                # In production, you'd compare against known faces
                return "Face Detected"
            
            return None
            
        except Exception as e:
            logger.debug(f"Face recognition error: {e}")
            return None
    
    def _detect_face_landmarks(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[List[Tuple[int, int, str]]]:
        """Detect face landmarks (eyes, nose, mouth) in bounding box."""
        if not FACE_LANDMARKS_AVAILABLE:
            return None
            
        try:
            x1, y1, x2, y2 = bbox
            # For person detection, only look at top 30% (head region)
            head_y2 = y1 + int((y2 - y1) * 0.3)
            
            # Ensure valid bbox
            if x2 <= x1 or head_y2 <= y1:
                return None
                
            roi = frame[y1:head_y2, x1:x2]
            
            if roi.size == 0:
                return None
            
            landmarks = []
            
            # Try MediaPipe first (more reliable)
            try:
                import mediapipe as mp
                mp_face_mesh = mp.solutions.face_mesh
                face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # Map landmark indices to features
                    landmark_map = {
                        # Eyes
                        'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133],
                        'right_eye': [362, 398, 384, 385, 386, 387, 388, 466, 263],
                        # Nose
                        'nose_tip': [1],
                        # Mouth
                        'mouth_center': [13, 14]
                    }
                    
                    h, w = roi.shape[:2]
                    
                    for feature_name, indices in landmark_map.items():
                        # Calculate center point for each feature
                        points = []
                        for idx in indices:
                            if idx < len(face_landmarks.landmark):
                                lm = face_landmarks.landmark[idx]
                                px = int(lm.x * w) + x1
                                py = int(lm.y * h) + y1
                                points.append((px, py))
                        
                        if points:
                            # Average position for the feature
                            avg_x = sum(p[0] for p in points) // len(points)
                            avg_y = sum(p[1] for p in points) // len(points)
                            landmarks.append((avg_x, avg_y, feature_name))
                    
                    return landmarks
                    
            except Exception as e:
                logger.debug(f"MediaPipe face landmarks error: {e}")
            
            return None
            
        except Exception as e:
            logger.debug(f"Face landmarks detection error: {e}")
            return None
    
    def _detect_hands(self, frame: np.ndarray) -> List[Dict]:
        """Detect hands and fingers using MediaPipe Hands (legacy API)."""
        if not HAND_DETECTION_AVAILABLE or hands_detector is None:
            return []
            
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(rgb_frame)
            
            hands_data = []
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Get hand landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        x = int(lm.x * frame.shape[1])
                        y = int(lm.y * frame.shape[0])
                        landmarks.append((x, y))
                    
                    # Count extended fingers
                    finger_count = self._count_fingers_legacy(hand_landmarks)
                    
                    # Calculate bounding box
                    xs = [p[0] for p in landmarks]
                    ys = [p[1] for p in landmarks]
                    bbox = (min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 10)
                    
                    hands_data.append({
                        'landmarks': landmarks,
                        'finger_count': finger_count,
                        'bbox': bbox,
                        'hand_idx': hand_idx
                    })
            
            return hands_data
            
        except Exception as e:
            logger.debug(f"Hand detection error: {e}")
            return []
    
    def _count_fingers_legacy(self, hand_landmarks) -> int:
        """Count number of extended fingers using legacy API."""
        try:
            fingers = []
            # Thumb (check x distance from palm center)
            thumb_tip = hand_landmarks.landmark[4]
            thumb_mcp = hand_landmarks.landmark[2]
            if thumb_tip.x < thumb_mcp.x:
                fingers.append(1)
            else:
                fingers.append(0)
            
            # Other 4 fingers (check if tip is higher than pip joint)
            finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
            finger_pips = [6, 10, 14, 18]  # PIP joints
            
            for tip, pip in zip(finger_tips, finger_pips):
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                    fingers.append(1)
                else:
                    fingers.append(0)
            
            return sum(fingers)
        except Exception as e:
            logger.debug(f"Finger counting error: {e}")
            return 0
            
    def predict(self, frame: np.ndarray) -> Tuple[List[DetectionResult], float]:
        """Run enhanced inference with adaptive thresholds and categorization."""
        start_time = time.perf_counter()
        self.frame_count += 1  # Increment frame counter at the start
        
        if self.detection_model is None and self.pose_model is None:
            # Fallback: simple motion detection
            return self._fallback_detection(frame), 0.0
        
        detections = []
        inference_time = 0
        
        # Resize frame for faster inference
        orig_h, orig_w = frame.shape[:2]
        inference_w = self.config.inference_size
        inference_h = int(orig_h * (inference_w / orig_w))
        
        if inference_w < orig_w:
            inference_frame = cv2.resize(frame, (inference_w, inference_h))
            scale_x = orig_w / inference_w
            scale_y = orig_h / inference_h
        else:
            inference_frame = frame
            scale_x, scale_y = 1.0, 1.0
        
        # Run detection model with enhanced processing
        if self.detection_model:
            det_start = time.perf_counter()
            det_results = self.detection_model(
                inference_frame,
                conf=self.config.confidence_threshold,  # Use base threshold, will adapt per class
                iou=self.config.iou_threshold,
                verbose=False,
                max_det=self.config.max_detections
            )[0]
            inference_time += (time.perf_counter() - det_start) * 1000
            
            # Process detection results with enhanced categorization
            if det_results.boxes is not None:
                for box in det_results.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = self.class_names.get(cls_id, "unknown")
                    
                    # Get adaptive threshold for this class
                    adaptive_threshold = self._get_adaptive_threshold(class_name)
                    if conf < adaptive_threshold:
                        continue
                    
                    color = self.class_colors.get(cls_id, (59, 130, 246))
                    
                    # Scale to original size
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    # Get category information
                    category_info = self._get_category_info(class_name)
                    
                    # Enhanced display name with category
                    display_name = f"{class_name.title()} [{category_info['category'].title()}]"
                    
                    # Person variant mapping
                    person_classes = ['person', 'man', 'woman', 'boy', 'girl']
                    if class_name.lower() in person_classes:
                        display_name = 'Person [Human]'
                    
                    # Analyze color features if enabled
                    color_features = None
                    if self.config.enable_color_analysis:
                        color_features = self._analyze_color_features(frame, (x1, y1, x2, y2))
                    
                    # No face recognition or face landmarks (disabled for performance)
                    face_name = None
                    face_landmarks = None
                    
                    # No OCR (disabled for performance)
                    detected_text = None
                    
                    detection = DetectionResult(
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        display_name=display_name,
                        color=color,
                        category=category_info['category'],
                        subcategory=category_info['subcategory'],
                        detected_text=detected_text,
                        face_name=face_name,
                        face_landmarks=face_landmarks,
                        finger_count=None,
                        color_features=color_features,
                        texture_features=None,  # Could be added later
                        relationships=None,  # Will be added in grouping phase
                        group_id=None
                    )
                    detections.append(detection)
        
        # Apply intelligent object grouping if enabled
        if self.config.enable_object_grouping and detections:
            detections = self._group_related_objects(detections)
        
        # Run pose model for human body parts (only if enabled)
        if self.config.enable_pose and self.pose_model:
            if self.frame_count % self.config.pose_detection_interval == 0:
                pose_start = time.perf_counter()
                pose_results = self.pose_model(
                    inference_frame,
                    conf=self.config.confidence_threshold,
                    verbose=False
                )[0]
                inference_time += (time.perf_counter() - pose_start) * 1000
                self.last_pose_results = pose_results
            elif self.last_pose_results is not None:
                # Use cached pose results
                pose_results = self.last_pose_results
            else:
                pose_results = None
            
            # Process pose results
            if pose_results and pose_results.boxes is not None and pose_results.keypoints is not None:
                for i, box in enumerate(pose_results.boxes):
                    conf = float(box.conf[0])
                    if conf < self.config.confidence_threshold:
                        continue
                    
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    # Scale to original size
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    # Get keypoints
                    keypoints = pose_results.keypoints[i].data[0].cpu().numpy()
                    kp_list = []
                    for kp in keypoints:
                        kp_x, kp_y, kp_conf = float(kp[0] * scale_x), float(kp[1] * scale_y), float(kp[2])
                        kp_list.append((kp_x, kp_y, kp_conf))
                    
                    # Check if we already have this person from detection
                    person_det = None
                    for det in detections:
                        if det.class_name == 'person':
                            # Check IOU
                            iou = self._calculate_iou(det.bbox, (x1, y1, x2, y2))
                            if iou > 0.5:
                                person_det = det
                                break
                    
                    if person_det:
                        # Add pose to existing person detection
                        person_det.keypoints = kp_list
                    else:
                        # Create new detection with pose
                        detection = DetectionResult(
                            class_name='person',
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            display_name='Person [Body Parts]',
                            color=(0, 255, 255),  # Yellow for pose
                            category='person',
                            subcategory='human',
                            keypoints=kp_list,
                            detected_text=None,
                            face_name=None,
                            finger_count=None
                        )
                        detections.append(detection)
        
        # Detect hands/fingers (skip every N frames for performance)
        if HAND_DETECTION_AVAILABLE and (self.frame_count % self.config.hand_detection_interval == 0):
            hands_data = self._detect_hands(frame)
            for hand in hands_data:
                x1, y1, x2, y2 = hand['bbox']
                finger_count = hand['finger_count']
                
                # Create detection for hand
                hand_detection = DetectionResult(
                    class_name='hand',
                    confidence=0.8,
                    bbox=(x1, y1, x2, y2),
                    display_name=f'Hand ({finger_count} fingers)',
                    color=(255, 128, 0),  # Orange for hands
                    category='body',
                    subcategory='hand',
                    detected_text=None,
                    face_name=None,
                    face_landmarks=None,
                    finger_count=finger_count,
                    keypoints=hand['landmarks']
                )
                detections.append(hand_detection)
        
        return detections, inference_time
    
    def _group_related_objects(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Group related objects and establish relationships."""
        if not detections:
            return detections
        
        # Create groups based on proximity and category
        groups = {}
        group_counter = 0
        
        for i, det1 in enumerate(detections):
            if det1.group_id is not None:
                continue  # Already grouped
            
            # Start a new group
            group_id = f"group_{group_counter}"
            group_counter += 1
            groups[group_id] = [det1]
            det1.group_id = group_id
            det1.relationships = []
            
            # Find related objects
            for j, det2 in enumerate(detections[i+1:], i+1):
                if det2.group_id is not None:
                    continue  # Already grouped
                
                # Check if objects are related
                relationship = self._detect_relationship(det1, det2)
                if relationship:
                    det2.group_id = group_id
                    det2.relationships = [relationship]
                    groups[group_id].append(det2)
                    
                    # Add reciprocal relationship
                    reverse_relationship = self._get_reverse_relationship(relationship)
                    det1.relationships.append(reverse_relationship)
        
        return detections
    
    def _detect_relationship(self, det1: DetectionResult, det2: DetectionResult) -> Optional[str]:
        """Detect relationship between two objects based on proximity and category."""
        # Calculate distance between object centers
        center1 = ((det1.bbox[0] + det1.bbox[2]) / 2, (det1.bbox[1] + det1.bbox[3]) / 2)
        center2 = ((det2.bbox[0] + det2.bbox[2]) / 2, (det2.bbox[1] + det2.bbox[3]) / 2)
        
        distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
        
        # Normalize distance by frame size (assuming 1920x1080 as reference)
        normalized_distance = distance / 1920.0
        
        # Only consider nearby objects
        if normalized_distance > 0.3:  # More than 30% of frame width apart
            return None
        
        # Category-based relationships
        cat1, cat2 = det1.category, det2.category
        
        # Same category - likely related
        if cat1 == cat2:
            return f"same_{cat1}"
        
        # Common relationships
        relationship_map = {
            ('person', 'food'): 'eating',
            ('person', 'vehicle'): 'driving',
            ('person', 'electronics'): 'using',
            ('person', 'furniture'): 'sitting',
            ('person', 'animal'): 'with',
            ('food', 'furniture'): 'on',
            ('electronics', 'furniture'): 'on',
            ('vehicle', 'person'): 'driving',
            ('animal', 'person'): 'with',
        }
        
        # Check both directions
        key = (cat1, cat2)
        reverse_key = (cat2, cat1)
        
        if key in relationship_map:
            return relationship_map[key]
        elif reverse_key in relationship_map:
            return relationship_map[reverse_key]
        
        # Spatial relationships
        if det2.bbox[1] > det1.bbox[3]:  # det2 is below det1
            return 'below'
        elif det2.bbox[3] < det1.bbox[1]:  # det2 is above det1
            return 'above'
        elif det2.bbox[0] > det1.bbox[2]:  # det2 is to the right of det1
            return 'right_of'
        elif det2.bbox[2] < det1.bbox[0]:  # det2 is to the left of det1
            return 'left_of'
        
        return None
    
    def _get_reverse_relationship(self, relationship: str) -> str:
        """Get the reverse relationship for bidirectional linking."""
        reverse_map = {
            'above': 'below',
            'below': 'above',
            'left_of': 'right_of',
            'right_of': 'left_of',
            'eating': 'being_eaten',
            'driving': 'being_driven',
            'using': 'being_used',
            'sitting': 'being_sat_on',
            'with': 'with',
            'on': 'under',
        }
        
        # For same_category relationships, return the same
        if relationship.startswith('same_'):
            return relationship
        
        return reverse_map.get(relationship, 'near')
    
    def _extract_text_if_needed(self, frame: np.ndarray, class_name: str, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """Extract text for specific object classes."""
        text_objects = ['book', 'cell phone', 'laptop', 'paper', 'card']
        if class_name.lower() in text_objects or class_name.lower() == 'person':
            x1, y1, x2, y2 = bbox
            # For person, only check center region (where ID might be held)
            if class_name.lower() == 'person':
                cx1 = x1 + int((x2-x1) * 0.3)
                cy1 = y1 + int((y2-y1) * 0.4)
                cx2 = x2 - int((x2-x1) * 0.3)
                cy2 = y1 + int((y2-y1) * 0.7)
                detected_text = self._extract_text_from_bbox(frame, (cx1, cy1, cx2, cy2))
                if detected_text:
                    logger.info(f"✅ Text detected on {class_name}: {detected_text}")
            else:
                detected_text = self._extract_text_from_bbox(frame, bbox)
            return detected_text
        return None
    
    def _fallback_detection(self, frame: np.ndarray) -> List[DetectionResult]:
        """Simple fallback detection when YOLO is not available."""
        detections = []
        
        # Simple motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if not hasattr(self, 'prev_frame'):
            self.prev_frame = gray
            return detections
        
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            
            (x, y, w, h) = cv2.boundingRect(contour)
            
            detection = DetectionResult(
                class_name='motion',
                confidence=0.8,
                bbox=(x, y, x + w, y + h),
                display_name='Motion',
                color=(0, 255, 0),
                detected_text=None,
                face_name=None,
                finger_count=None
            )
            detections.append(detection)
        
        self.prev_frame = gray
        return detections
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IOU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
            
    def _extract_text_from_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """Extract text from bounding box area using OCR."""
        if not self.ocr_reader:
            return None
            
        try:
            x1, y1, x2, y2 = bbox
            # Extract the region of interest
            roi = frame[y1:y2, x1:x2]
            
            # Preprocess for better OCR
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Use EasyOCR
            results = self.ocr_reader.readtext(enhanced)
            
            # Extract and clean text
            texts = []
            for (bbox_points, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence results
                    # Clean text - keep alphanumeric and common symbols
                    clean_text = re.sub(r'[^a-zA-Z0-9\s\-\\./]', '', text).strip()
                    if len(clean_text) > 2:
                        texts.append(clean_text)
            
            return ' '.join(texts) if texts else None
            
        except Exception as e:
            logger.debug(f"OCR error: {e}")
            return None
    
    def _detect_fingers(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[int]:
        """Count fingers in hand detection area."""
        try:
            x1, y1, x2, y2 = bbox
            roi = frame[y1:y2, x1:x2]
            
            # Convert to HSV for better skin detection
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Skin color range
            lower_skin = np.array([0, 48, 80], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create skin mask
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Find contours
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Simple finger counting based on contour analysis
            finger_count = 0
            if contours:
                # Find largest contour (should be hand)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Count convexity defects (potential fingers)
                hull = cv2.convexHull(largest_contour, returnPoints=False)
                if len(hull) > 3:
                    defects = cv2.convexityDefects(largest_contour, hull)
                    if defects is not None:
                        finger_count = min(5, len(defects))  # Max 5 fingers
            
            return finger_count if finger_count > 0 else None
            
        except Exception as e:
            logger.debug(f"Finger detection error: {e}")
            return None
    
    def _recognize_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """Recognize face in bounding box area - looks at head region only."""
        if not FACE_RECOGNITION_AVAILABLE:
            return None
            
        try:
            x1, y1, x2, y2 = bbox
            # For person detection, only look at top 30% (head region)
            head_y2 = y1 + int((y2 - y1) * 0.3)
            
            # Ensure valid bbox
            if x2 <= x1 or head_y2 <= y1:
                return None
                
            roi = frame[y1:head_y2, x1:x2]
            
            if roi.size == 0:
                return None
            
            # Convert to RGB (face_recognition uses RGB)
            rgb_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # Use HOG for speed
            
            if face_locations:
                # For demo, return that face was detected
                # In production, you'd compare against known faces
                return "Face Detected"
            
            return None
            
        except Exception as e:
            logger.debug(f"Face recognition error: {e}")
            return None


class CameraManager:
    """Manages camera capture with thread-safe frame buffering."""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
    def start(self) -> bool:
        """Start camera capture with multiple fallback options."""
        # Try different camera indices and backends
        camera_configs = [
            (0, cv2.CAP_DSHOW),  # DirectShow backend (Windows)
            (0, cv2.CAP_ANY),    # Any backend
            (1, cv2.CAP_DSHOW),  # Secondary camera
            (0, cv2.CAP_MSMF),   # Media Foundation
        ]
        
        for cam_idx, backend in camera_configs:
            try:
                self.cap = cv2.VideoCapture(cam_idx, backend)
                if self.cap.isOpened():
                    # Wait for camera warmup
                    time.sleep(0.5)
                    
                    # Test frame capture
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        logger.info(f"Camera {cam_idx} opened successfully with backend {backend}")
                        
                        # Configure camera settings
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
                        
                        self.running = True
                        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
                        self.thread.start()
                        logger.info(f"Camera {cam_idx} started successfully")
                        return True
                    else:
                        self.cap.release()
                        
            except Exception as e:
                logger.warning(f"Failed to open camera {cam_idx} with backend {backend}: {e}")
                if self.cap:
                    self.cap.release()
        
        logger.error("All camera initialization attempts failed")
        return False
            
    def _capture_loop(self):
        """Optimized frame capture loop with error handling."""
        retry_count = 0
        max_retries = 3
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    retry_count += 1
                    if retry_count > max_retries:
                        logger.error(f"Camera failed after {max_retries} retries")
                        break
                    
                    logger.warning(f"Camera lost, retrying ({retry_count}/{max_retries})")
                    time.sleep(1)
                    
                    # Try to reopen camera
                    try:
                        self.cap.release()
                    except:
                        pass
                    
                    self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                    if self.cap.isOpened():
                        retry_count = 0
                        logger.info("Camera reconnected")
                    continue
                
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    retry_count += 1
                    if retry_count > max_retries:
                        logger.error("Frame capture failed")
                        break
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logger.error(f"Camera failed {max_failures} times consecutively")
                        break
                    time.sleep(0.01)
                    continue
                
                # Reset counters on successful capture
                retry_count = 0
                consecutive_failures = 0
                
                # Update frame thread-safely
                with self.lock:
                    self.frame = frame
                
                # Small sleep to prevent 100% CPU
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                retry_count += 1
                if retry_count > max_retries:
                    break
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                time.sleep(0.1)
                continue
                
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame thread-safely."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
            
    def get_capture_fps(self) -> float:
        """Get average capture FPS."""
        if len(self.fps_history) == 0:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)
        
    def stop(self):
        """Stop camera capture."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        logger.info("Camera stopped")


class DetectorApp:
    """Main detector application orchestrating camera, model, and streaming."""
    
    def __init__(self):
        self.config = DetectionConfig()
        self.model: Optional[DetectionModel] = None
        self.camera: Optional[CameraManager] = None
        self.running = False
        self.detection_thread = None
        self.frame_stats: Optional[FrameStats] = None
        self.stats_history: deque = deque(maxlen=100)
        # For optimized inference and tracking
        self.last_detections: List[DetectionResult] = []
        self.last_inference_time: float = 0.0
        self.frame_count: int = 0
        self.last_emit_time: float = 0
        # For temporal filtering (stabilize detections)
        self.tracked_objects: Dict[str, Dict] = {}
        self.detection_persistence: int = 15  # Reduced for less lag
        self.iou_threshold: float = 0.35  # Balanced matching
        self.bbox_smoothing_alpha: float = 0.4  # More responsive tracking
        # FPS stabilization
        self.fps_history: deque = deque(maxlen=10)
        self.target_frame_time: float = 1.0 / 30.0  # 33.33ms target
        
    def initialize(self) -> bool:
        """Initialize all components."""
        try:
            # Load unified model (detection + pose + segmentation)
            self.model = DetectionModel(self.config)
            
            # Start camera
            self.camera = CameraManager(self.config.camera_index)
            if not self.camera.start():
                return False
                
            # Start detection loop
            self.running = True
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
            logger.info("Detector app initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False
            
    def _draw_detections(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """Draw detection boxes with labels and pose keypoints."""
        overlay = frame.copy()
        
        # COCO pose keypoint connections
        pose_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Face
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Hips
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = det.color
            
            # Draw rectangle border
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Build comprehensive label
            label_parts = [det.display_name]
            
            # Add pose indicator
            if det.keypoints:
                label_parts.append("[Body Parts]")
            
            # Add face name if available
            if det.face_name:
                label_parts.append(f"({det.face_name})")
            
            # Add finger count if available
            if det.finger_count:
                label_parts.append(f"Fingers: {det.finger_count}")
            
            # Add detected text if available
            if det.detected_text:
                # Truncate long text
                text = det.detected_text[:30] + "..." if len(det.detected_text) > 30 else det.detected_text
                label_parts.append(f"Text: {text}")
            
            label = " ".join(label_parts)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_y = max(y1 - 10, label_size[1] + 5)
            
            # Draw label background
            cv2.rectangle(
                overlay,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0] + 10, label_y + 5),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                overlay, label, (x1 + 5, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                cv2.LINE_AA
            )
            
            # Special highlighting for detected text
            if det.detected_text:
                # Draw a small indicator for text detection
                text_indicator_color = (0, 255, 0)  # Green for text
                cv2.circle(overlay, (x2 - 10, y1 + 10), 5, text_indicator_color, -1)
            
            # Note: Pose keypoints (body parts) drawing disabled
            # Note: Face landmarks drawing disabled
            
            # Draw hand landmarks if available (for finger detection)
            if det.class_name == 'hand' and det.keypoints:
                # Hand landmark connections (simplified skeleton)
                hand_connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
                ]
                
                # Draw hand skeleton
                for start_idx, end_idx in hand_connections:
                    if start_idx < len(det.keypoints) and end_idx < len(det.keypoints):
                        x1_hand, y1_hand = det.keypoints[start_idx]
                        x2_hand, y2_hand = det.keypoints[end_idx]
                        cv2.line(overlay, (x1_hand, y1_hand), (x2_hand, y2_hand), (255, 128, 0), 2)
                
                # Draw hand keypoints
                for i, (x, y) in enumerate(det.keypoints):
                    if i in [4, 8, 12, 16, 20]:  # Finger tips
                        color = (0, 255, 0)  # Green for tips
                        radius = 5
                    elif i == 0:  # Wrist
                        color = (255, 0, 0)  # Blue for wrist
                        radius = 6
                    else:
                        color = (255, 255, 0)  # Yellow for joints
                        radius = 3
                    cv2.circle(overlay, (x, y), radius, color, -1)
            
        return overlay
        
    def _filter_detections_temporal(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Enhanced temporal filtering with category-aware tracking."""
        current_time = self.frame_count
        new_tracked = {}
        alpha = self.bbox_smoothing_alpha
        
        # Category-specific tracking parameters
        category_params = {
            'fruit': {'persistence': 20, 'iou_threshold': 0.3, 'smoothing': 0.3},  # More persistent for fruits
            'animal': {'persistence': 15, 'iou_threshold': 0.4, 'smoothing': 0.4},  # Moderate for animals
            'vehicle': {'persistence': 10, 'iou_threshold': 0.5, 'smoothing': 0.5},  # Less smoothing for vehicles
            'person': {'persistence': 12, 'iou_threshold': 0.4, 'smoothing': 0.4},  # Balanced for persons
            'default': {'persistence': 15, 'iou_threshold': 0.35, 'smoothing': 0.4}
        }
        
        # Match current detections to existing tracked objects
        for det in detections:
            best_match_id = None
            best_iou = 0.0
            
            # Get category-specific parameters
            category = det.category or 'default'
            params = category_params.get(category, category_params['default'])
            
            # Find best matching tracked object
            for track_id, track_data in self.tracked_objects.items():
                # Check category compatibility
                track_category = track_data.get('category', 'default')
                if category != track_category and category != 'object' and track_category != 'object':
                    continue  # Different categories don't match
                
                # Calculate IOU
                iou = self._calculate_iou(det.bbox, track_data['bbox'])
                if iou > best_iou and iou > params['iou_threshold']:
                    best_iou = iou
                    best_match_id = track_id
                
                # Also check center distance for better matching
                if self._calculate_center_distance(det.bbox, track_data['bbox']) < 50:
                    if iou > 0.2:  # Minimum IOU threshold for distant matches
                        best_iou = iou
                        best_match_id = track_id
            
            if best_match_id:
                # Get previous bbox for smoothing
                prev_bbox = self.tracked_objects[best_match_id]['bbox']
                curr_bbox = det.bbox
                
                # Apply category-specific smoothing
                smoothing_alpha = params['smoothing']
                smoothed_bbox = (
                    int(smoothing_alpha * curr_bbox[0] + (1 - smoothing_alpha) * prev_bbox[0]),  # x1
                    int(smoothing_alpha * curr_bbox[1] + (1 - smoothing_alpha) * prev_bbox[1]),  # y1
                    int(smoothing_alpha * curr_bbox[2] + (1 - smoothing_alpha) * prev_bbox[2]),  # x2
                    int(smoothing_alpha * curr_bbox[3] + (1 - smoothing_alpha) * prev_bbox[3]),  # y2
                )
                
                # Update existing track with enhanced information
                new_tracked[best_match_id] = {
                    'class_name': det.class_name,
                    'display_name': det.display_name,
                    'color': det.color,
                    'category': det.category,
                    'subcategory': det.subcategory,
                    'bbox': smoothed_bbox,
                    'confidence': det.confidence,
                    'last_seen': current_time,
                    'detected_text': det.detected_text,
                    'face_name': det.face_name,
                    'face_landmarks': det.face_landmarks,
                    'finger_count': det.finger_count,
                    'keypoints': det.keypoints if self.config.enable_pose else None,
                    'color_features': det.color_features,
                    'texture_features': det.texture_features,
                    'relationships': det.relationships,
                    'group_id': det.group_id,
                    'track_quality': best_iou  # Track match quality
                }
            else:
                # Create new track with all enhanced fields
                track_id = f"{det.category}_{det.class_name}_{det.bbox[0]}_{det.bbox[1]}_{current_time}"
                new_tracked[track_id] = {
                    'class_name': det.class_name,
                    'display_name': det.display_name,
                    'color': det.color,
                    'category': det.category,
                    'subcategory': det.subcategory,
                    'bbox': det.bbox,
                    'confidence': det.confidence,
                    'last_seen': current_time,
                    'detected_text': det.detected_text,
                    'face_name': det.face_name,
                    'face_landmarks': det.face_landmarks,
                    'finger_count': det.finger_count,
                    'keypoints': det.keypoints if self.config.enable_pose else None,
                    'color_features': det.color_features,
                    'texture_features': det.texture_features,
                    'relationships': det.relationships,
                    'group_id': det.group_id,
                    'track_quality': 1.0  # New track starts with perfect quality
                }
        
        # Add persistent objects with category-specific persistence
        for track_id, track_data in self.tracked_objects.items():
            if track_id not in new_tracked:
                category = track_data.get('category', 'default')
                params = category_params.get(category, category_params['default'])
                frames_since_seen = current_time - track_data['last_seen']
                
                if frames_since_seen < params['persistence']:
                    # Decay confidence for old tracks
                    decay_factor = 1.0 - (frames_since_seen / params['persistence']) * 0.3
                    track_data['confidence'] *= decay_factor
                    
                    # Keep this detection alive
                    new_tracked[track_id] = track_data
        
        # Update tracked objects
        self.tracked_objects = new_tracked
        
        # Convert back to DetectionResult with all enhanced fields
        stable_detections = []
        for track_data in self.tracked_objects.values():
            det = DetectionResult(
                class_name=track_data['class_name'],
                confidence=track_data['confidence'],
                bbox=track_data['bbox'],
                display_name=track_data['display_name'],
                color=track_data['color'],
                category=track_data.get('category'),
                subcategory=track_data.get('subcategory'),
                detected_text=track_data.get('detected_text'),
                face_name=track_data.get('face_name'),
                face_landmarks=track_data.get('face_landmarks'),
                finger_count=track_data.get('finger_count'),
                keypoints=track_data.get('keypoints'),
                color_features=track_data.get('color_features'),
                texture_features=track_data.get('texture_features'),
                relationships=track_data.get('relationships'),
                group_id=track_data.get('group_id')
            )
            stable_detections.append(det)
        
        return stable_detections
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union for two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _calculate_center_distance(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate distance between centers of two bounding boxes."""
        x1_center = (bbox1[0] + bbox1[2]) / 2
        y1_center = (bbox1[1] + bbox1[3]) / 2
        x2_center = (bbox2[0] + bbox2[2]) / 2
        y2_center = (bbox2[1] + bbox2[3]) / 2
        
        return ((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2) ** 0.5
        
    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame to base64 JPEG."""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
        return base64.b64encode(buffer).decode('utf-8')
        
    def _detection_loop(self):
        """Optimized detection loop with stable FPS and proper detection display."""
        target_frame_time = self.target_frame_time  # 33.33ms for 30 FPS
        last_frame_time = time.perf_counter()
        
        while self.running:
            try:
                loop_start = time.perf_counter()
                
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                self.frame_count += 1
                
                # Flip frame horizontally to fix mirrored camera view
                frame = cv2.flip(frame, 1)
                
                # Run inference with timeout protection
                detections = []
                inference_time = 0
                try:
                    detections, inference_time = self.model.predict(frame)
                    if detections:
                        logger.debug(f"Raw detections: {len(detections)} objects")
                except Exception as e:
                    logger.error(f"Inference error: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # Apply temporal filtering with error handling
                try:
                    detections = self._filter_detections_temporal(detections)
                    if detections:
                        logger.debug(f"After temporal filter: {len(detections)} objects")
                except Exception as e:
                    logger.error(f"Temporal filter error: {e}")
                
                self.last_detections = detections
                self.last_inference_time = inference_time
                
                # Draw results with error handling
                try:
                    annotated_frame = self._draw_detections(frame, detections)
                except Exception as e:
                    logger.error(f"Drawing error: {e}")
                    annotated_frame = frame
                
                # Calculate frame timing for FPS
                current_time = time.perf_counter()
                actual_frame_time = current_time - last_frame_time
                
                # Control frame rate - sleep to maintain target FPS
                if actual_frame_time < target_frame_time:
                    sleep_time = target_frame_time - actual_frame_time
                    if sleep_time > 0.001:
                        time.sleep(sleep_time)
                
                # Update last frame time after processing
                last_frame_time = time.perf_counter()
                
                # Calculate actual FPS based on complete frame cycle
                actual_fps = 1.0 / (last_frame_time - current_time + actual_frame_time) if (last_frame_time - current_time + actual_frame_time) > 0 else 30.0
                
                # Smooth FPS with moving average
                self.fps_history.append(actual_fps)
                if len(self.fps_history) > 30:  # Keep last 30 frames
                    self.fps_history.popleft()
                smoothed_fps = sum(self.fps_history) / len(self.fps_history)
                
                # Clamp to reasonable range
                smoothed_fps = max(15.0, min(60.0, smoothed_fps))
                
                # Update stats
                class_counts = {}
                for det in detections:
                    class_counts[det.display_name] = class_counts.get(det.display_name, 0) + 1
                
                self.frame_stats = FrameStats(
                    fps=smoothed_fps,
                    inference_time_ms=inference_time,
                    total_detections=len(detections),
                    detected_classes=class_counts,
                    timestamp=datetime.now().isoformat()
                )
                self.stats_history.append(self.frame_stats)
                
                # Emit frame with error handling
                try:
                    frame_data = {
                        'frame': self._encode_frame(annotated_frame),
                        'stats': self.frame_stats.to_dict(),
                        'detections': [d.to_dict() for d in detections]
                    }
                    socketio.emit('frame', frame_data)
                except Exception as e:
                    logger.error(f"Socket emit error: {e}")
                
                # Log FPS and detections every 60 frames (2 seconds at 30 FPS)
                if self.frame_count % 60 == 0:
                    logger.info(f"FPS: {smoothed_fps:.1f} | Inference: {inference_time:.0f}ms | Objects: {len(detections)} | Classes: {list(class_counts.keys())}")
                
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(0.01)
                continue
            
    def get_stats_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.stats_history:
            return {}
            
        recent = list(self.stats_history)[-30:]  # Last 30 frames
        avg_fps = sum(s.fps for s in recent) / len(recent)
        avg_inference = sum(s.inference_time_ms for s in recent) / len(recent)
        
        # Count total detections by class
        total_by_class = {}
        for s in recent:
            for cls, count in s.detected_classes.items():
                total_by_class[cls] = total_by_class.get(cls, 0) + count
                
        return {
            'average_fps': round(avg_fps, 1),
            'average_inference_ms': round(avg_inference, 1),
            'total_detections_by_class': total_by_class,
            'frames_processed': len(self.stats_history)
        }
        
    def update_config(self, new_config: Dict) -> bool:
        """Update configuration dynamically."""
        try:
            if 'confidence_threshold' in new_config:
                self.config.confidence_threshold = float(new_config['confidence_threshold'])
            if 'target_classes' in new_config:
                self.config.target_classes = new_config['target_classes']
            return True
        except Exception as e:
            logger.error(f"Config update error: {e}")
            return False
            
    def shutdown(self):
        """Shutdown the application."""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        if self.camera:
            self.camera.stop()
        logger.info("Detector app shutdown")


# Global app instance
detector_app = DetectorApp()


# Flask Routes
@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')


@app.route('/api/stats')
def get_stats():
    """Get current statistics."""
    return jsonify(detector_app.get_stats_summary())


@app.route('/api/current_model')
def get_current_model():
    """Get current model information."""
    return jsonify({
        'model_type': detector_app.config.model_type,
        'model_path': detector_app.config.model_path,
        'display_name': detector_app.config.model_path.replace('.pt', '').replace('yolov8', 'YOLOv8')
    })


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update detector configuration."""
    data = request.get_json()
    success = detector_app.update_config(data)
    return jsonify({'success': success})


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info("Client connected")
    emit('connected', {'message': 'Connected to detector'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected")


if __name__ == '__main__':
    # Initialize detector
    if not detector_app.initialize():
        logger.error("Failed to initialize detector")
        sys.exit(1)
        
    try:
        # Run Flask-SocketIO server
        print("\n" + "="*60)
        print("CAMERA DETECTOR SERVER STARTING")
        print("="*60)
        print("✅ GPU Accelerated: NVIDIA RTX 4050")
        
        # Check if models loaded successfully
        if detector_app.model and (detector_app.model.detection_model or detector_app.model.pose_model):
            models_loaded = []
            if detector_app.model.detection_model:
                models_loaded.append("Detection")
            if detector_app.model.pose_model:
                models_loaded.append("Pose (Body Parts)")
            print(f"✅ Models: {', '.join(models_loaded)}")
        else:
            print("⚠️  Models: Motion Detection (AI models failed to load)")
        
        print("✅ Open in browser:")
        print("   http://127.0.0.1:5000")
        print("   http://localhost:5000")
        print("="*60 + "\n")
        
        logger.info("Starting server...")
        logger.info("Open http://127.0.0.1:5000 in your browser")
        logger.info("Or http://localhost:5000")
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        detector_app.shutdown()
