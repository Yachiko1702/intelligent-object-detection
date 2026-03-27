# 🤖 Intelligent Object Detection System

A state-of-the-art real-time object detection system with enhanced accuracy for fruits, animals, vehicles, electronics, furniture, and food items. Features intelligent categorization, adaptive thresholds, and relationship detection.

## ✨ Key Features

### 🎯 **Enhanced Detection Accuracy**
- **YOLOv8 Large model** for maximum precision
- **1280px inference size** for better small object detection
- **Adaptive confidence thresholds** per object category
- **100+ object types** across 6 major categories

### 🧠 **Intelligent Analysis**
- **Smart categorization**: Fruits, Animals, Vehicles, Electronics, Furniture, Food
- **Object relationship detection** (eating, driving, using, etc.)
- **Color analysis** using HSV and LAB color spaces
- **Proximity-based grouping** of related objects

### ⚡ **Performance Optimized**
- **30 FPS real-time processing**
- **Category-aware temporal filtering**
- **Efficient memory management**
- **GPU acceleration support**

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/intelligent-object-detection.git
cd intelligent-object-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app.py
```

5. **Open in browser**
```
http://localhost:5000
```

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
docker-compose up -d
```

### Manual Docker Build

```bash
docker build -t object-detection .
docker run -p 5000:5000 --gpus all object-detection
```

## ☁️ Cloud Deployment Options

### 1. Heroku
```bash
# Install Heroku CLI
heroku create your-app-name
heroku stack:set container
git push heroku main
```

### 2. Railway
```bash
# Install Railway CLI
railway login
railway init
railway up
```

### 3. Render
- Connect your GitHub repository to Render
- Select "Web Service"
- Use Docker deployment option
- Set port to 5000

### 4. Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/object-detection
gcloud run deploy --image gcr.io/PROJECT-ID/object-detection --platform managed
```

## 📋 Supported Objects

### 🍎 **Fruits** (22 types)
Apple, Banana, Orange, Grape, Strawberry, Lemon, Lime, Peach, Pear, Plum, Cherry, Blueberry, Raspberry, Watermelon, Cantaloupe, Mango, Pineapple, Kiwi, Avocado, Coconut, Papaya, Pomegranate

### 🦁 **Animals** (30 types)
Dog, Cat, Horse, Cow, Sheep, Goat, Pig, Chicken, Duck, Goose, Turkey, Bird, Rabbit, Mouse, Rat, Squirrel, Deer, Elephant, Giraffe, Zebra, Lion, Tiger, Bear, Wolf, Fox, Monkey, Ape, Kangaroo, Koala, Panda, Hippo, Rhino

### 🚗 **Vehicles** (20 types)
Car, Truck, Bus, Motorcycle, Bicycle, Scooter, Airplane, Helicopter, Boat, Ship, Train, Subway, Tractor, Ambulance, Police Car, Fire Truck, Taxi, Van, SUV, Pickup

### 💻 **Electronics** (20 types)
Cell Phone, Laptop, Computer, Tablet, TV, Monitor, Keyboard, Mouse, Remote, Camera, Printer, Scanner, Speaker, Headphones, Microwave, Refrigerator, Washing Machine, Dryer, Oven, Toaster

### 🪑 **Furniture** (14 types)
Chair, Table, Desk, Bed, Sofa, Couch, Bookshelf, Cabinet, Dresser, Nightstand, Stool, Bench, Armchair, Recliner, Ottoman

### 🍕 **Food** (20 types)
Pizza, Hamburger, Hot Dog, Sandwich, Salad, Soup, Pasta, Bread, Cake, Cookie, Donut, Ice Cream, Coffee, Tea, Juice, Water, Soda, Milk, Cheese, Meat, Fish, Vegetable, Fruit

## ⚙️ Configuration

### Environment Variables
```env
# Model Configuration
MODEL_TYPE=detection
MODEL_PATH=yolov8l.pt
CONFIDENCE_THRESHOLD=0.25
INFERENCE_SIZE=1280

# Performance
MAX_FPS=30
JPEG_QUALITY=95
MAX_DETECTIONS=100

# Features
ENABLE_ADAPTIVE_THRESHOLD=true
ENABLE_OBJECT_GROUPING=true
ENABLE_COLOR_ANALYSIS=true
ENABLE_POSE=false
ENABLE_OCR=false
```

### Adaptive Thresholds
- **Fruits**: 0.20 (optimized for small/occluded items)
- **Animals**: 0.30 (balanced confidence)
- **Vehicles**: 0.40 (higher for distinct shapes)
- **Electronics**: 0.35 (medium-high confidence)
- **Furniture**: 0.45 (higher for large objects)
- **Food**: 0.25 (lower for varied items)

## 🔧 API Endpoints

### Main Endpoints
- `GET /` - Web interface
- `GET /api/stats` - Detection statistics
- `GET /api/current_model` - Model information
- `POST /api/config` - Update configuration

### WebSocket Events
- `frame` - Real-time detection frames
- `connect` - Client connection
- `disconnect` - Client disconnection

## 📊 Detection Features

### Enhanced Detection Results
```json
{
  "class_name": "apple",
  "display_name": "Apple [Fruit]",
  "confidence": 0.85,
  "category": "fruit",
  "subcategory": "produce",
  "bbox": [100, 150, 200, 250],
  "color_features": {
    "mean_hsv": [25.5, 180.2, 200.1],
    "dominant_colors": [[255, 0, 0], [200, 50, 50]],
    "brightness": 200.1,
    "saturation": 180.2
  },
  "relationships": ["same_fruit"],
  "group_id": "group_0"
}
```

### Object Relationships
- **Eating**: Person + Food
- **Driving**: Person + Vehicle  
- **Using**: Person + Electronics
- **Sitting**: Person + Furniture
- **With**: Person + Animal
- **Spatial**: Above, Below, Left, Right

## 🛠️ Development

### Project Structure
```
intelligent-object-detection/
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── .github/workflows/    # GitHub Actions
├── templates/            # HTML templates
├── static/              # Static assets
├── models/              # YOLO model files
└── README.md            # This file
```

### Adding New Object Categories
1. Update `DetectionConfig.__post_init__()` method
2. Add to class categorization mapping
3. Set adaptive threshold
4. Update documentation

## 🔒 Security Considerations

- Camera access requires HTTPS in production
- Model files are large (ensure proper storage)
- GPU resources may be expensive on cloud platforms
- Consider rate limiting for API endpoints

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the amazing detection model
- [OpenCV](https://opencv.org/) for computer vision operations
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [MediaPipe](https://mediapipe.dev/) for hand detection

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

---

**⭐ If this project helps you, please give it a star!**
