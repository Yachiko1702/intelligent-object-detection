#!/usr/bin/env python3
"""
Pose estimation integration for human body detection.
Uses YOLOv6-pose for detecting human keypoints.
"""

import cv2
import numpy as np
from ultralytics import YOLO

class PoseDetector:
    """Human pose detection using YOLOv6-pose."""
    
    def __init__(self, model_path="yolov6s-pose.pt"):
        self.model = YOLO(model_path)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
    def detect_pose(self, frame):
        """Detect human pose keypoints."""
        results = self.model(frame, verbose=False)
        poses = []
        
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                confidences = result.keypoints.conf.cpu().numpy()
                
                for i, (kpts, conf) in enumerate(zip(keypoints, confidences)):
                    pose_data = {
                        'keypoints': kpts.tolist(),
                        'confidences': conf.tolist(),
                        'bbox': result.boxes.xyxy[i].cpu().numpy().tolist() if result.boxes is not None else None
                    }
                    poses.append(pose_data)
        
        return poses
    
    def draw_pose(self, frame, poses):
        """Draw pose keypoints and skeleton on frame."""
        # Define skeleton connections
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 15)  # Legs
        ]
        
        for pose in poses:
            keypoints = np.array(pose['keypoints'])
            confidences = np.array(pose['confidences'])
            
            # Draw keypoints
            for i, (kpt, conf) in enumerate(zip(keypoints, confidences)):
                if conf > 0.5 and kpt[0] > 0 and kpt[1] > 0:
                    cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 4, (0, 255, 0), -1)
            
            # Draw skeleton
            for start_idx, end_idx in skeleton:
                if (confidences[start_idx] > 0.5 and confidences[end_idx] > 0.5 and
                    keypoints[start_idx][0] > 0 and keypoints[start_idx][1] > 0 and
                    keypoints[end_idx][0] > 0 and keypoints[end_idx][1] > 0):
                    
                    start_pt = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                    end_pt = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                    cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)
        
        return frame
