/**
 * Object Detector Pro - Frontend Application
 * Real-time camera detection with WebSocket streaming
 */

class DetectorApp {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.currentFrame = null;
        this.detections = [];
        this.stats = null;
        this.lastDetectionSignature = '';
        
        // Class aliases - map COCO classes to custom display names
        this.classAliases = {
            'person': 'Person',
            'cell phone': 'Phone',
            'book': 'Book',
            'bottle': 'Bottle',
            'cup': 'Cup',
            'wine glass': 'Glass',
            'backpack': 'Bag',
            'handbag': 'Bag',
            'suitcase': 'Luggage',
            'chair': 'Chair',
            'couch': 'Couch',
            'bed': 'Bed',
            'dining table': 'Table',
            'clock': 'Clock',
            'vase': 'Vase',
            'scissors': 'Scissors',
            'car': 'Car',
            'truck': 'Truck',
            'bus': 'Bus',
            'motorcycle': 'Motorcycle',
            'bicycle': 'Bicycle',
            'dog': 'Dog',
            'cat': 'Cat',
            'bird': 'Bird',
            'horse': 'Horse',
            'cow': 'Cow',
            'sheep': 'Sheep',
            'laptop': 'Laptop',
            'mouse': 'Mouse',
            'keyboard': 'Keyboard',
            'tv': 'TV',
            'remote': 'Remote',
            'microwave': 'Microwave',
            'oven': 'Oven',
            'toaster': 'Toaster',
            'sink': 'Sink',
            'refrigerator': 'Fridge',
            'potted plant': 'Plant',
            'teddy bear': 'Teddy Bear',
            'hair drier': 'Hair Dryer',
            'toothbrush': 'Toothbrush',
            'tie': 'Tie',
            'umbrella': 'Umbrella',
            'handbag': 'Handbag',
            'frisbee': 'Frisbee',
            'skis': 'Skis',
            'snowboard': 'Snowboard',
            'sports ball': 'Ball',
            'kite': 'Kite',
            'baseball bat': 'Bat',
            'baseball glove': 'Glove',
            'skateboard': 'Skateboard',
            'surfboard': 'Surfboard',
            'tennis racket': 'Tennis Racket',
            'banana': 'Banana',
            'apple': 'Apple',
            'sandwich': 'Sandwich',
            'orange': 'Orange',
            'broccoli': 'Broccoli',
            'carrot': 'Carrot',
            'hot dog': 'Hot Dog',
            'pizza': 'Pizza',
            'donut': 'Donut',
            'cake': 'Cake'
        };
        
        // DOM Elements
        this.elements = {
            videoFeed: document.getElementById('videoFeed'),
            loadingOverlay: document.getElementById('loadingOverlay'),
            connectionStatus: document.getElementById('connectionStatus'),
            fpsValue: document.getElementById('fpsValue'),
            inferenceValue: document.getElementById('inferenceValue'),
            objectsValue: document.getElementById('objectsValue'),
            modelValue: document.getElementById('modelValue'),
            detectionList: document.getElementById('detectionList'),
            fullscreenBtn: document.getElementById('fullscreenBtn'),
            snapshotBtn: document.getElementById('snapshotBtn'),
            snapshotModal: document.getElementById('snapshotModal'),
            snapshotImage: document.getElementById('snapshotImage'),
            modalClose: document.getElementById('modalClose'),
            downloadSnapshot: document.getElementById('downloadSnapshot'),
            detectionOverlay: document.getElementById('detectionOverlay')
        };
        
        // Cache for model info to reduce API calls
        this.cachedModelPath = null;
        this.lastModelCheck = 0;
        
        this.init();
    }
    
    init() {
        this.setupSocket();
        this.setupEventListeners();
    }
    
    setupSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const socketUrl = `${protocol}//${window.location.host}`;
        
        this.socket = io(socketUrl, {
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionAttempts: 5,
            reconnectionDelay: 1000
        });
        
        this.socket.on('connect', () => {
            console.log('Connected to detector server');
            this.isConnected = true;
            this.updateConnectionStatus('connected', 'Connected');
            this.elements.loadingOverlay.classList.add('hidden');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.isConnected = false;
            this.updateConnectionStatus('error', 'Disconnected');
            this.elements.loadingOverlay.classList.remove('hidden');
        });
        
        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            this.updateConnectionStatus('error', 'Connection Error');
        });
        
        this.socket.on('frame', (data) => {
            this.handleFrame(data);
        });
    }
    
    setupEventListeners() {
        // Fullscreen button
        this.elements.fullscreenBtn.addEventListener('click', () => {
            this.toggleFullscreen();
        });
        
        // Snapshot button
        this.elements.snapshotBtn.addEventListener('click', () => {
            this.takeSnapshot();
        });
        
        // Modal close buttons
        this.elements.modalClose.addEventListener('click', () => {
            this.closeModal();
        });
        
        this.elements.downloadSnapshot.addEventListener('click', () => {
            this.downloadSnapshot();
        });
        
        // Close modal on outside click
        this.elements.snapshotModal.addEventListener('click', (e) => {
            if (e.target === this.elements.snapshotModal) {
                this.closeModal();
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
            }
            if (e.key === 'f' || e.key === 'F') {
                this.toggleFullscreen();
            }
            if (e.key === 's' || e.key === 'S') {
                this.takeSnapshot();
            }
        });
    }
    
    updateConnectionStatus(status, text) {
        const indicator = this.elements.connectionStatus;
        indicator.className = 'status-indicator';
        
        if (status === 'connected') {
            indicator.classList.add('connected');
        } else if (status === 'error') {
            indicator.classList.add('error');
        }
        
        indicator.querySelector('.status-text').textContent = text;
    }
    
    handleFrame(data) {
        // Hide loading overlay on first frame
        if (this.elements.loadingOverlay) {
            this.elements.loadingOverlay.style.display = 'none';
        }
        
        // Update video feed with smooth transition
        if (this.elements.videoFeed.src !== `data:image/jpeg;base64,${data.frame}`) {
            this.elements.videoFeed.style.opacity = '1';
            this.elements.videoFeed.src = `data:image/jpeg;base64,${data.frame}`;
        }
        this.currentFrame = data.frame;
        
        // Update stats (async but we don't await to avoid blocking)
        this.updateStats(data.stats);
        
        // Update detections list
        this.updateDetections(data.detections);
        
        // Render detection overlay
        this.renderOverlay(data.detections);
    }
    
    async updateStats(stats) {
        this.stats = stats;
        
        // Update stat values with animation
        this.animateValue(this.elements.fpsValue, stats.fps, 1);
        this.animateValue(this.elements.inferenceValue, stats.inference_time_ms, 1, ' ms');
        this.animateValue(this.elements.objectsValue, stats.total_detections, 0);
        
        // Update model display based on current model type
        const modelPath = await this.getCurrentModelPath();
        const modelName = this.getModelDisplayName(modelPath);
        this.elements.modelValue.textContent = modelName;
    }
    
    async getCurrentModelPath() {
        // Cache model info to avoid excessive API calls
        if (this.cachedModelPath && Date.now() - this.lastModelCheck < 5000) {
            return this.cachedModelPath;
        }
        
        try {
            const response = await fetch('/api/current_model');
            const data = await response.json();
            this.cachedModelPath = data.model_path.replace('.pt', '');
            this.lastModelCheck = Date.now();
            return this.cachedModelPath;
        } catch (error) {
            console.error('Failed to get current model:', error);
            return 'yolov8n'; // Default to Nano
        }
    }
    
    getModelDisplayName(modelPath) {
        const modelNames = {
            'yolov8n': 'YOLOv8 Nano',
            'yolov8s': 'YOLOv8 Small',
            'yolov8m': 'YOLOv8 Medium',
            'yolov8l': 'YOLOv8 Large',
            'yolov8x': 'YOLOv8 X-Large',
            'yolov8n-pose': 'YOLOv8 Pose',
            'yolov8n-seg': 'YOLOv8 Seg'
        };
        return modelNames[modelPath] || 'YOLOv8';
    }
    
    animateValue(element, newValue, decimals, suffix = '') {
        const currentValue = parseFloat(element.textContent) || 0;
        const targetValue = newValue;
        
        if (isNaN(targetValue)) {
            element.textContent = '--';
            return;
        }
        
        // Simple direct update for performance
        element.textContent = targetValue.toFixed(decimals) + suffix;
        
        // Add color coding for FPS
        if (element === this.elements.fpsValue) {
            element.classList.remove('positive', 'warning');
            if (targetValue >= 25) {
                element.classList.add('positive');
            } else if (targetValue < 15) {
                element.classList.add('warning');
            }
        }
            
    if (status === 'connected') {
        indicator.classList.add('connected');
    } else if (status === 'error') {
        indicator.classList.add('error');
    }
            
    indicator.querySelector('.status-text').textContent = text;
}
        
handleFrame(data) {
    // Hide loading overlay on first frame
    if (this.elements.loadingOverlay) {
        this.elements.loadingOverlay.style.display = 'none';
    }
            
    // Update video feed with smooth transition
    if (this.elements.videoFeed.src !== `data:image/jpeg;base64,${data.frame}`) {
        this.elements.videoFeed.style.opacity = '1';
        this.elements.videoFeed.src = `data:image/jpeg;base64,${data.frame}`;
    }
    this.currentFrame = data.frame;
            
    // Update stats (async but we don't await to avoid blocking)
    this.updateStats(data.stats);
            
    // Update detections list
    this.updateDetections(data.detections);
            
    // Render detection overlay
    this.renderOverlay(data.detections);
}
        
async updateStats(stats) {
    this.stats = stats;
            
    // Update stat values with animation
    this.animateValue(this.elements.fpsValue, stats.fps, 1);
    this.animateValue(this.elements.inferenceValue, stats.inference_time_ms, 1, ' ms');
    this.animateValue(this.elements.objectsValue, stats.total_detections, 0);
            
    // Update model display based on current model type
    const modelPath = await this.getCurrentModelPath();
    const modelName = this.getModelDisplayName(modelPath);
    this.elements.modelValue.textContent = modelName;
}
        
async getCurrentModelPath() {
    // Cache model info to avoid excessive API calls
    if (this.cachedModelPath && Date.now() - this.lastModelCheck < 5000) {
        return this.cachedModelPath;
    }
            
    try {
        const response = await fetch('/api/current_model');
        const data = await response.json();
        this.cachedModelPath = data.model_path.replace('.pt', '');
        this.lastModelCheck = Date.now();
        return this.cachedModelPath;
    } catch (error) {
        console.error('Failed to get current model:', error);
        return 'yolov8n'; // Default to Nano
    }
}
        
getModelDisplayName(modelPath) {
    const modelNames = {
        'yolov8n': 'YOLOv8 Nano',
        'yolov8s': 'YOLOv8 Small',
        'yolov8m': 'YOLOv8 Medium',
        'yolov8l': 'YOLOv8 Large',
        'yolov8x': 'YOLOv8 X-Large',
        'yolov8n-pose': 'YOLOv8 Pose',
        'yolov8n-seg': 'YOLOv8 Seg'
    };
    return modelNames[modelPath] || 'YOLOv8';
}
        
animateValue(element, newValue, decimals, suffix = '') {
    const currentValue = parseFloat(element.textContent) || 0;
    const targetValue = newValue;
            
    if (isNaN(targetValue)) {
        element.textContent = '--';
        return;
    }
            
    // Simple direct update for performance
    element.textContent = targetValue.toFixed(decimals) + suffix;
            
    // Add color coding for FPS
    if (element === this.elements.fpsValue) {
        element.classList.remove('positive', 'warning');
        if (targetValue >= 25) {
            element.classList.add('positive');
        } else if (targetValue < 15) {
            element.classList.add('warning');
        }
    }
}
        
updateDetections(detections) {
    this.detections = detections;
            
    // Create signature based on class names and approximate positions
    const signature = detections.map(d => 
        `${d.class_name}-${Math.round(d.bbox[0]/10)}-${Math.round(d.bbox[1]/10)}`
    ).sort().join('|');
            
    // Skip update if nothing changed
    if (signature === this.lastDetectionSignature) {
        return;
    }
    this.lastDetectionSignature = signature;
            
    if (detections.length === 0) {
        this.elements.detectionList.innerHTML = `
            <div class="empty-state">
                <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="8" y1="12" x2="16" y2="12"/>
                </svg>
                <p>No objects detected</p>
                <span>Point camera at objects to detect: people, phones, IDs, books, etc.</span>
            </div>
        `;
        return;
    }
            
    // Group detections by class
    const grouped = {};
    detections.forEach(det => {
        const displayName = this.classAliases[det.class_name] || this.capitalize(det.class_name);
        if (!grouped[displayName]) {
            grouped[displayName] = [];
        }
        grouped[displayName].push({...det, displayName});
    });
            
    // Render detection items with enhanced information
    const html = Object.entries(grouped).map(([className, items]) => {
        const best = items.reduce((a, b) => a.confidence > b.confidence ? a : b);
        const color = `rgb(${best.color.join(',')})`;
                
        return items.map((det) => {
            let detailsHtml = '';
                    
            // Add face name if available
            if (det.face_name) {
                detailsHtml += `<div class="detection-detail">👤 ${det.face_name}</div>`;
            }
                    
            // Add finger count if available
            if (det.finger_count) {
                detailsHtml += `<div class="detection-detail">🤚 ${det.finger_count} fingers</div>`;
            }
                    
            // Add detected text if available
            if (det.detected_text) {
                detailsHtml += `<div class="detection-detail">📝 ${det.detected_text}</div>`;
            }
                    
            return `
                <div class="detection-item">
                    <span class="detection-color" style="background: ${color};"></span>
                    <div class="detection-info">
                        <span class="detection-name">${det.displayName}</span>
                        <span class="detection-meta">${(det.confidence * 100).toFixed(0)}% confident</span>
                        ${detailsHtml}
                    </div>
                </div>
            `;
        }).join('');
    }).join('');
            
    this.elements.detectionList.innerHTML = html;
}
        
renderOverlay(detections) {
    // Clear previous overlay
    this.elements.detectionOverlay.innerHTML = '';
    
    // No overlay boxes drawn - showing clean video feed
}
    
capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}
    
    updateConfig(config) {
        if (!this.isConnected) return;
        
        fetch('/api/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        }).catch(err => {
            console.error('Failed to update config:', err);
        });
    }
    
    toggleFullscreen() {
        const container = document.querySelector('.video-container');
        
        if (!document.fullscreenElement) {
            container.requestFullscreen().catch(err => {
                console.error('Fullscreen error:', err);
            });
        } else {
            document.exitFullscreen();
        }
    }
    
    takeSnapshot() {
        if (!this.currentFrame) return;
        
        this.elements.snapshotImage.src = `data:image/jpeg;base64,${this.currentFrame}`;
        this.elements.snapshotModal.classList.add('active');
    }
    
    closeModal() {
        this.elements.snapshotModal.classList.remove('active');
    }
    
    downloadSnapshot() {
        const link = document.createElement('a');
        link.download = `detection-snapshot-${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`;
        link.href = this.elements.snapshotImage.src;
        link.click();
    }
}

// Add CSS for detection details and smooth transitions
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.98); }
        to { opacity: 1; transform: scale(1); }
    }
    
    .overlay-box {
        transition: all 0.15s ease-out;
    }
    
    .detection-item {
        transition: background-color 0.2s ease;
    }
    
    .detection-detail {
        font-size: 0.85r    em;
        color: #94a3b8;
        margin-top: 4px;
        padding: 2px 0;
    }
    
    #videoFeed {
        transition: opacity 0.1s ease;
        will-change: opacity;
    }
`;
document.head.appendChild(style);

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.detectorApp = new DetectorApp();
});
