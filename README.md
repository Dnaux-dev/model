# OBEX Security System - ML-Powered Surveillance

A comprehensive security system with real-time object detection, person tracking, loitering detection, zone-based intrusion alerts, motion heatmaps, and face recognition in low-light conditions.

## 🚀 Features

- **Theft Detection System** - Object tagging and owner association
- **Loitering & Suspicious Behavior Detection** - Track people staying in zones too long
- **Zone-Based Intrusion Alerts** - Define restricted areas and get alerts
- **Real-Time Video Streaming** - Live video feed with detection overlays
- **Motion & Heatmap Tracking** - Visualize movement patterns in enclosed spaces
- **Face Recognition in Low-Light** - Enhanced detection with CLAHE preprocessing
- **Interactive Web UI** - Draw zones, toggle features, download heatmaps

## 📋 Prerequisites

- Python 3.8+
- Node.js 14+
- Webcam or video file
- YOLOv8 model weights

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Dnaux-dev/Primus-Lite-Model.git
cd Primus-Lite-Model
```

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

### 3. Download YOLOv8 Weights
Download `yolov8s.pt` from [Ultralytics releases](https://github.com/ultralytics/ultralytics/releases) and place it in:
```
models/yolov8/yolov8s.pt
```

### 4. Frontend Setup
```bash
cd ../frontend
npm install
```

## 🚀 Running the Application

### 1. Start the Backend
```bash
cd backend
uvicorn app:app --reload
```
Backend will be available at: http://localhost:8000

### 2. Start the Frontend
```bash
cd frontend
npm start
```
Frontend will be available at: http://localhost:3000

## 📡 API Endpoints

### Core Detection
- `GET /video_feed` - Live video stream with detection overlays
- `GET /latest_detections` - Latest object detection results
- `POST /detect_objects` - Upload image for object detection
- `POST /tag_object` - Tag object with owner
- `GET /get_objects` - List all tagged objects

### Zone Management
- `GET /zones` - Get current zone coordinates
- `POST /set_zone` - Set new zone coordinates
- `GET /loitering_alerts` - Get current loitering events
- `GET /intrusion_alerts` - Get current intrusion events

### Motion & Heatmaps
- `GET /heatmap` - Download current heatmap as image

### Face Recognition
- `GET /faces` - Get latest detected face locations

### API Documentation
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 🎯 Usage Guide

### 1. Basic Setup
1. Start both backend and frontend servers
2. Open http://localhost:3000 in your browser
3. You'll see the live video feed with detection overlays

### 2. Zone-Based Detection
1. Click "Draw Zone" button in the frontend
2. Click and drag on the video to define a restricted area
3. The system will alert when people enter or loiter in the zone

### 3. Motion Heatmap
1. Toggle "Show Heatmap" switch in the frontend
2. Move around in front of the camera
3. Areas with more movement will appear "hotter" (red/yellow)
4. Use "Download Heatmap" to save the current heatmap

### 4. Face Recognition
- Faces are automatically detected and highlighted with blue rectangles
- Works even in low-light conditions thanks to CLAHE enhancement

### 5. Object Detection & Tagging
- Objects are detected and labeled in real-time
- Use the API to tag objects with owners for theft detection

## 🔧 Configuration

### Video Source
Edit `backend/app.py`:
```python
VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
```

### Heatmap Settings
Adjust in `backend/app.py`:
```python
HEATMAP_DECAY = 0.95  # Lower = faster fade
HEATMAP_ALPHA = 0.5   # Higher = stronger overlay
```

### Loitering Threshold
```python
LOITER_THRESHOLD = 30  # seconds before loitering alert
```

## 📁 Project Structure

```
model/
├── backend/
│   ├── app.py              # Main FastAPI application
│   ├── detection/
│   │   ├── detector.py     # YOLOv8 object detection
│   │   └── tracker.py      # DeepSORT person tracking
│   ├── db/
│   │   ├── models.py       # Database models
│   │   └── db_utils.py     # Database utilities
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.js          # Main React app
│   │   └── components/
│   │       └── VideoFeed.js # Video display and controls
│   └── package.json        # Node.js dependencies
└── models/
    └── yolov8/
        └── yolov8s.pt      # YOLOv8 model weights
```

## 🐛 Troubleshooting

### Common Issues

1. **"Module not found" errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

2. **Video feed not showing**
   - Check if webcam is working or video file exists
   - Verify VIDEO_SOURCE setting

3. **Face detection not working**
   - Install face_recognition: `pip install face_recognition`
   - May require additional system dependencies on Linux

4. **Frontend can't connect to backend**
   - Ensure backend is running on port 8000
   - Check CORS settings if needed

### Performance Tips

- Use a GPU for faster YOLOv8 inference
- Reduce video resolution for better performance
- Adjust HEATMAP_DECAY for faster/slower heatmap updates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [DeepSORT](https://github.com/nwojke/deep_sort)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/) 