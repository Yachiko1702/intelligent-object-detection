#!/usr/bin/env python3
"""
YOLOv8 Model Downloader & Manager
Downloads and manages all YOLOv8 model variants
"""

from ultralytics import YOLO
import os

class ModelManager:
    """Manages YOLOv8 model downloads and switching."""
    
    # All available YOLOv8 models
    MODELS = {
        # Detection Models (COCO dataset - 80 classes)
        'detection': {
            'yolov8n': {
                'file': 'yolov8n.pt',
                'name': 'YOLOv8 Nano',
                'speed': '400+ FPS',
                'accuracy': '40.9 mAP',
                'description': 'Fastest detection for edge devices',
                'use_case': 'Real-time applications, mobile devices'
            },
            'yolov8s': {
                'file': 'yolov8s.pt',
                'name': 'YOLOv8 Small',
                'speed': '400+ FPS',
                'accuracy': '48.6 mAP',
                'description': 'Best balance speed/accuracy',
                'use_case': 'General purpose, recommended'
            },
            'yolov8m': {
                'file': 'yolov8m.pt',
                'name': 'YOLOv8 Medium',
                'speed': '200+ FPS',
                'accuracy': '53.1 mAP',
                'description': 'High accuracy, still fast',
                'use_case': 'Production systems'
            },
            'yolov8l': {
                'file': 'yolov8l.pt',
                'name': 'YOLOv8 Large',
                'speed': '160+ FPS',
                'accuracy': '55.0 mAP',
                'description': 'Very accurate detection',
                'use_case': 'Quality control, inspection'
            },
            'yolov8x': {
                'file': 'yolov8x.pt',
                'name': 'YOLOv8 X-Large',
                'speed': '85+ FPS',
                'accuracy': '57.5 mAP',
                'description': 'Maximum accuracy',
                'use_case': 'Critical applications'
            }
        },
        
        # Segmentation Models (COCO-Seg dataset)
        'segmentation': {
            'yolov8n-seg': {
                'file': 'yolov8n-seg.pt',
                'name': 'YOLOv8 Nano Seg',
                'speed': '300+ FPS',
                'accuracy': '47.3 mAP',
                'description': 'Pixel-perfect masks',
                'use_case': 'AR applications, image editing'
            },
            'yolov8s-seg': {
                'file': 'yolov8s-seg.pt',
                'name': 'YOLOv8 Small Seg',
                'speed': '250+ FPS',
                'accuracy': '52.5 mAP',
                'description': 'Balanced segmentation',
                'use_case': 'Photo editing, background removal'
            },
            'yolov8m-seg': {
                'file': 'yolov8m-seg.pt',
                'name': 'YOLOv8 Medium Seg',
                'speed': '150+ FPS',
                'accuracy': '54.4 mAP',
                'description': 'Accurate masks',
                'use_case': 'Medical imaging, analysis'
            },
            'yolov8x-seg': {
                'file': 'yolov8x-seg.pt',
                'name': 'YOLOv8 X-Large Seg',
                'speed': '80+ FPS',
                'accuracy': '56.5 mAP',
                'description': 'Max accuracy segmentation',
                'use_case': 'Professional editing'
            }
        },
        
        # Pose Estimation Models (COCO-Pose dataset)
        'pose': {
            'yolov8n-pose': {
                'file': 'yolov8n-pose.pt',
                'name': 'YOLOv8 Nano Pose',
                'speed': '370+ FPS',
                'accuracy': '63.0 mAP',
                'description': '17 body keypoints',
                'use_case': 'Fitness apps, motion tracking'
            },
            'yolov8s-pose': {
                'file': 'yolov8s-pose.pt',
                'name': 'YOLOv8 Small Pose',
                'speed': '300+ FPS',
                'accuracy': '68.8 mAP',
                'description': 'Accurate pose estimation',
                'use_case': 'Sports analysis, PT'
            },
            'yolov8m-pose': {
                'file': 'yolov8m-pose.pt',
                'name': 'YOLOv8 Medium Pose',
                'speed': '200+ FPS',
                'accuracy': '70.4 mAP',
                'description': 'High precision poses',
                'use_case': 'Dance, gymnastics'
            },
            'yolov8x-pose': {
                'file': 'yolov8x-pose.pt',
                'name': 'YOLOv8 X-Large Pose',
                'speed': '85+ FPS',
                'accuracy': '71.6 mAP',
                'description': 'Max accuracy pose',
                'use_case': 'Professional motion capture'
            }
        },
        
        # Classification Models (ImageNet dataset - 1000 classes)
        'classification': {
            'yolov8n-cls': {
                'file': 'yolov8n-cls.pt',
                'name': 'YOLOv8 Nano Class',
                'speed': '500+ FPS',
                'accuracy': '71.4% top-1',
                'description': 'Fast classification',
                'use_case': 'Sorting, categorization'
            },
            'yolov8x-cls': {
                'file': 'yolov8x-cls.pt',
                'name': 'YOLOv8 X-Large Class',
                'speed': '100+ FPS',
                'accuracy': '79.9% top-1',
                'description': 'Accurate classification',
                'use_case': 'Medical diagnosis, analysis'
            }
        },
        
        # Oriented Bounding Box (OBB) Models
        'obb': {
            'yolov8n-obb': {
                'file': 'yolov8n-obb.pt',
                'name': 'YOLOv8 Nano OBB',
                'speed': '200+ FPS',
                'accuracy': '52.4 mAP',
                'description': 'Rotated boxes detection',
                'use_case': 'Satellite imagery, aerial photos'
            }
        }
    }
    
    def __init__(self):
        self.downloaded_models = []
        
    def download_model(self, model_key, category='detection'):
        """Download a specific YOLOv8 model."""
        if category not in self.MODELS:
            print(f"❌ Unknown category: {category}")
            return None
            
        if model_key not in self.MODELS[category]:
            print(f"❌ Unknown model: {model_key}")
            return None
        
        model_info = self.MODELS[category][model_key]
        model_file = model_info['file']
        
        try:
            print(f"📥 Downloading {model_info['name']}...")
            model = YOLO(model_file)
            print(f"✅ {model_info['name']} downloaded successfully!")
            print(f"   • Speed: {model_info['speed']}")
            print(f"   • Accuracy: {model_info['accuracy']}")
            print(f"   • Use: {model_info['use_case']}")
            self.downloaded_models.append(model_key)
            return model
        except Exception as e:
            print(f"❌ Failed to download {model_key}: {e}")
            return None
    
    def download_all_detection(self):
        """Download all detection models."""
        print("📦 Downloading all detection models...\n")
        for model_key in self.MODELS['detection']:
            self.download_model(model_key, 'detection')
            print()
    
    def download_all_segmentation(self):
        """Download all segmentation models."""
        print("🎨 Downloading all segmentation models...\n")
        for model_key in self.MODELS['segmentation']:
            self.download_model(model_key, 'segmentation')
            print()
    
    def download_all_pose(self):
        """Download all pose estimation models."""
        print("🧍 Downloading all pose models...\n")
        for model_key in self.MODELS['pose']:
            self.download_model(model_key, 'pose')
            print()
    
    def download_essential_models(self):
        """Download only essential models (recommended)."""
        print("⭐ Downloading essential models...\n")
        essentials = [
            ('yolov8n', 'detection'),  # Fast
            ('yolov8s', 'detection'),  # Balanced
            ('yolov8x', 'detection'),  # Accurate
            ('yolov8n-pose', 'pose'),  # Pose
            ('yolov8n-seg', 'segmentation'),  # Segmentation
        ]
        
        for model_key, category in essentials:
            self.download_model(model_key, category)
            print()
    
    def list_available_models(self):
        """List all available models."""
        print("📋 Available YOLOv8 Models:\n")
        
        for category, models in self.MODELS.items():
            print(f"\n{category.upper()}:")
            print("=" * 60)
            for key, info in models.items():
                print(f"\n  {info['name']} ({key})")
                print(f"    File: {info['file']}")
                print(f"    Speed: {info['speed']} | Accuracy: {info['accuracy']}")
                print(f"    Use: {info['description']}")


def main():
    """Main function to download models."""
    manager = ModelManager()
    
    print("🚀 YOLOv8 Model Downloader\n")
    print("Choose download option:")
    print("1. Download essential models (recommended)")
    print("2. Download all detection models")
    print("3. Download all segmentation models")
    print("4. Download all pose models")
    print("5. List all available models")
    print("6. Download specific model")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-6): ").strip()
    
    if choice == '1':
        manager.download_essential_models()
    elif choice == '2':
        manager.download_all_detection()
    elif choice == '3':
        manager.download_all_segmentation()
    elif choice == '4':
        manager.download_all_pose()
    elif choice == '5':
        manager.list_available_models()
    elif choice == '6':
        category = input("Enter category (detection/segmentation/pose/classification/obb): ").strip()
        model = input("Enter model key (e.g., yolov8n): ").strip()
        manager.download_model(model, category)
    else:
        print("Exiting...")


if __name__ == '__main__':
    main()
