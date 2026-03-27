# Object Detector Pro

A professional-grade, web-based camera object detector with real-time AI detection using YOLO models. Features a modern, senior developer-quality UI with live statistics, detection history, and configurable confidence thresholds.

## Features

- **Real-time Detection**: WebSocket streaming for low-latency video feed
- **Modern UI/UX**: Dark theme with glassmorphism, smooth animations, responsive design
- **Target Classes**: Detects Bag, Ballpen, and Tumbler (with COCO class mapping)
- **Live Statistics**: FPS counter, inference time, detection count
- **Interactive Controls**: Confidence slider, fullscreen mode, snapshot capture
- **Detection Overlay**: Real-time bounding boxes with color-coded classes

## Quick Start

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the Application**:
```bash
python app.py
```

3. **Open in Browser**:
Navigate to `http://localhost:5000`

## Target Objects

| Display Name | Mapped COCO Classes | Color |
|-------------|---------------------|-------|
| **Bag** | backpack, handbag, suitcase | Blue |
| **Tumbler** | bottle, cup, wine_glass | Green |
| **Ballpen** | pen (requires custom model) | Orange |

## Architecture

```
┌─────────────┐     WebSocket      ┌─────────────┐
│   Browser   │◄──────────────────►│ Flask Server│
│  (UI/UX)    │    JPEG Frames     │  (app.py)   │
└─────────────┘                    └──────┬──────┘
                                          │
                                    ┌─────┴─────┐
                                    │  YOLOv8   │
                                    │  Model    │
                                    └───────────┘
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `F` | Toggle Fullscreen |
| `S` | Take Snapshot |
| `Esc` | Close Modal / Exit Fullscreen |

## Configuration

Adjust detection sensitivity using the confidence slider in the UI (0-100%).

## Notes

- The standard YOLOv8 model is trained on COCO dataset
- "Ballpen" detection uses the "pen" class which may not be in standard COCO
- For best results with ballpen detection, a custom-trained model is recommended
- GPU acceleration is automatically used if available

## Browser Support

- Chrome/Edge 90+
- Firefox 90+
- Safari 14+
