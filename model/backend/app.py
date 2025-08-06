from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from detection.detector import YoloV8Detector
from db.models import Owner, TaggedObject
from db.db_utils import SessionLocal, init_db
import shutil
import os
import json
import threading
import time
import cv2
from detection.tracker import DeepSortTracker
import time as pytime
import numpy as np
import face_recognition
import math
from collections import defaultdict, deque

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = YoloV8Detector()
init_db()

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Video source configuration
# VIDEO_SOURCE = 0  # Webcam
# VIDEO_SOURCE = "rtsp://admin:YourPassword@192.168.1.2:554/h264/ch1/main/av_stream"  # IP Camera
VIDEO_SOURCE = "scene 2.mp4"  # Video file for testing (replace with your video file name)
# VIDEO_SOURCE = "scene 1.mp4"  # Alternative video file
latest_detections = []
latest_frame = None
latest_faces = []  # Store latest detected faces

tracker = DeepSortTracker()

# Replace hardcoded ZONE with a mutable global
zone_coords = None  # No zone by default
LOITER_THRESHOLD = 30  # seconds

person_zone_times = {}  # track_id: entry_time
loitering_alerts = []   # [{track_id, entry_time, duration}]
intrusion_alerts = []   # [{track_id, entry_time}]

# Heatmap tracking
heatmap = None
prev_gray = None
HEATMAP_DECAY = 0.95  # Decay factor for fading old motion
HEATMAP_ALPHA = 0.5   # Overlay strength

# Theft detection variables
theft_alerts = []  # [{object_id, owner_id, timestamp, type}]
suspicious_behavior_alerts = []  # [{person_id, behavior_type, timestamp, details}]
tracked_objects = {}  # object_id: {bbox, owner_id, last_seen, movement_history}
person_object_associations = {}  # person_id: [object_ids]
exit_zones = [(500, 400, 640, 480)]  # Define exit zones
high_value_zones = [(100, 100, 300, 300)]  # Define high-value item zones
suspicious_movement_threshold = 50  # pixels per frame for suspicious speed
group_coordination_threshold = 3  # number of people for group behavior

# Performance settings - Ultra-optimized for slower systems
PROCESS_EVERY_N_FRAMES = 5  # Process every 5th frame (was 2)
DETECTION_INTERVAL = 0.2  # Run detection every 200ms (was 50ms)
VIDEO_FPS = 10  # Reduce FPS for better performance (was 20)
FRAME_SKIP_COUNT = 0  # Counter for frame skipping

# Performance monitoring
frame_count = 0
last_detection_time = 0
processing_times = []

@app.post("/set_zone")
async def set_zone(request: Request):
    data = await request.json()
    global zone_coords
    zone_coords = [int(data['x1']), int(data['y1']), int(data['x2']), int(data['y2'])]
    return {"status": "ok", "zone": zone_coords}

@app.get("/zones")
def get_zones():
    return {"zones": [zone_coords] if zone_coords else []}

@app.get("/heatmap")
def get_heatmap():
    global heatmap
    if heatmap is None:
        return Response(content=b'', media_type='image/jpeg')
    # Normalize and apply colormap
    norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_map = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    _, jpeg = cv2.imencode('.jpg', color_map)
    return Response(content=jpeg.tobytes(), media_type='image/jpeg')

@app.get("/faces")
def get_faces():
    return {"faces": latest_faces}

@app.get("/theft_alerts")
def get_theft_alerts():
    return {"theft_alerts": theft_alerts}

@app.get("/suspicious_behavior")
def get_suspicious_behavior():
    return {"suspicious_behavior": suspicious_behavior_alerts}

# Background thread for video capture and detection
def video_capture_thread():
    global latest_detections, latest_frame, latest_faces, heatmap, prev_gray, frame_count, last_detection_time, processing_times
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    # Check if video file opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source: {VIDEO_SOURCE}")
        print("Please check if the video file exists in the backend directory")
        return
    
    # For video files, don't set FPS as it can cause issues
    if not VIDEO_SOURCE.isdigit():  # If it's a file, not webcam
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    else:
        cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print(f"Successfully opened video source: {VIDEO_SOURCE}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break
        
        frame_count += 1
        current_time = pytime.time()
        
        # Always update the video feed frame
        latest_frame = frame.copy()
        
        # Skip heavy processing for performance
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            time.sleep(1/VIDEO_FPS)
            continue
        
        # Low-light enhancement using CLAHE
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        
        # Heatmap processing (every frame)
        if heatmap is None:
            heatmap = np.zeros_like(gray, dtype=np.float32)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, motion_mask = cv2.threshold(diff, 25, 1, cv2.THRESH_BINARY)
            heatmap = HEATMAP_DECAY * heatmap + motion_mask.astype(np.float32)
        prev_gray = gray.copy()
        
        # Run detection more frequently to reduce flickering
        if current_time - last_detection_time >= DETECTION_INTERVAL:
            start_time = pytime.time()
            
            # Face detection (on enhanced frame)
            rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            latest_faces = face_locations
            
            # Run detection on the enhanced frame
            detections = detector.detect_np(enhanced_frame)
            latest_detections = detections  # Update latest detections
            
            # Person tracking
            tracked_persons = tracker.update(detections, enhanced_frame)
            now = pytime.time()
            
            # Zone logic
            for person in tracked_persons:
                x1, y1, x2, y2 = map(int, person['bbox'])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                in_zone = False
                if zone_coords is not None:
                    in_zone = (zone_coords[0] <= cx <= zone_coords[2]) and (zone_coords[1] <= cy <= zone_coords[3])
                tid = person['track_id']
                if in_zone:
                    if tid not in person_zone_times:
                        person_zone_times[tid] = now
                        intrusion_alerts.append({'track_id': tid, 'entry_time': now})
                    else:
                        duration = now - person_zone_times[tid]
                        if duration > LOITER_THRESHOLD and not any(a['track_id'] == tid for a in loitering_alerts):
                            loitering_alerts.append({'track_id': tid, 'entry_time': person_zone_times[tid], 'duration': duration})
                else:
                    if tid in person_zone_times:
                        del person_zone_times[tid]
            
            # Theft and suspicious behavior detection
            detect_theft(detections, tracked_persons, now)
            detect_suspicious_behavior(tracked_persons, now)
            
            # Draw overlays on the enhanced frame
            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                label = f"{det['class']} {det['confidence']:.2f}"
                color = (0, 255, 0)  # Green box
                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(enhanced_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw zone
            if zone_coords is not None:
                cv2.rectangle(enhanced_frame, (zone_coords[0], zone_coords[1]), (zone_coords[2], zone_coords[3]), (0, 0, 255), 2)
            
            # Draw faces
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(enhanced_frame, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(enhanced_frame, "Face", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw exit zones and high-value zones
            for zone in exit_zones:
                cv2.rectangle(enhanced_frame, (zone[0], zone[1]), (zone[2], zone[3]), (0, 255, 255), 2)
                cv2.putText(enhanced_frame, "EXIT ZONE", (zone[0], zone[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            for zone in high_value_zones:
                cv2.rectangle(enhanced_frame, (zone[0], zone[1]), (zone[2], zone[3]), (255, 255, 0), 2)
                cv2.putText(enhanced_frame, "HIGH VALUE", (zone[0], zone[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Performance monitoring
            processing_time = pytime.time() - start_time
            processing_times.append(processing_time)
            if len(processing_times) > 10:
                processing_times.pop(0)
            
            last_detection_time = current_time
            # Update latest_frame with the enhanced frame that has overlays
            latest_frame = enhanced_frame.copy()
        else:
            # If we skipped processing, still update the frame but without overlays
            latest_frame = frame.copy()
        
        time.sleep(1/VIDEO_FPS)
    cap.release()

# Start video capture in background
t = threading.Thread(target=video_capture_thread, daemon=True)
t.start()

def mjpeg_generator():
    global latest_frame
    while True:
        if latest_frame is not None:
            ret, jpeg = cv2.imencode('.jpg', latest_frame)
            if ret:
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(mjpeg_generator(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/latest_detections")
def latest_detections_api():
    return {"detections": latest_detections}

@app.post("/detect_objects")
def detect_objects(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    detections = detector.detect(file_path)
    return {"detections": detections, "image_path": file_path}

@app.post("/tag_object")
def tag_object(
    owner_name: str = Form(...),
    object_class: str = Form(...),
    bbox: str = Form(...),  # JSON string
    image_path: str = Form(...)
):
    db = SessionLocal()
    owner = db.query(Owner).filter_by(name=owner_name).first()
    if not owner:
        owner = Owner(name=owner_name)
        db.add(owner)
        db.commit()
        db.refresh(owner)
    tagged_object = TaggedObject(
        object_class=object_class,
        bbox=bbox,
        image_path=image_path,
        owner_id=owner.id
    )
    db.add(tagged_object)
    db.commit()
    db.refresh(tagged_object)
    db.close()
    return {"status": "success", "tagged_object_id": tagged_object.id}

@app.get("/get_objects")
def get_objects():
    db = SessionLocal()
    objects = db.query(TaggedObject).all()
    result = []
    for obj in objects:
        owner = db.query(Owner).filter_by(id=obj.owner_id).first()
        result.append({
            "id": obj.id,
            "object_class": obj.object_class,
            "bbox": obj.bbox,
            "image_path": obj.image_path,
            "owner": owner.name if owner else None
        })
    db.close()
    return {"tagged_objects": result}

@app.get("/loitering_alerts")
def get_loitering_alerts():
    return {"loitering_alerts": loitering_alerts}

@app.get("/intrusion_alerts")
def get_intrusion_alerts():
    return {"intrusion_alerts": intrusion_alerts} 

def calculate_movement_speed(history, frames=10):
    """Calculate movement speed based on recent history"""
    if len(history) < 2:
        return 0
    recent = history[-frames:] if len(history) > frames else history
    total_distance = 0
    for i in range(1, len(recent)):
        x1, y1 = recent[i-1]['center']
        x2, y2 = recent[i]['center']
        distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        total_distance += distance
    return total_distance / len(recent)

def detect_suspicious_behavior(tracked_persons, frame_time):
    """Detect suspicious behavior patterns"""
    global suspicious_behavior_alerts
    
    # Track movement history for each person
    for person in tracked_persons:
        person_id = person['track_id']
        bbox = person['bbox']
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        
        if person_id not in person_movement_history:
            person_movement_history[person_id] = deque(maxlen=30)
        
        person_movement_history[person_id].append({
            'center': center,
            'timestamp': frame_time,
            'bbox': bbox
        })
        
        # Detect suspicious movement speed
        speed = calculate_movement_speed(person_movement_history[person_id])
        if speed > suspicious_movement_threshold:
            alert = {
                'person_id': person_id,
                'behavior_type': 'suspicious_speed',
                'timestamp': frame_time,
                'details': f'Person {person_id} moving at {speed:.1f} pixels/frame'
            }
            if not any(a['person_id'] == person_id and a['behavior_type'] == 'suspicious_speed' for a in suspicious_behavior_alerts):
                suspicious_behavior_alerts.append(alert)
    
    # Detect group coordination (multiple people moving together)
    if len(tracked_persons) >= group_coordination_threshold:
        # Check if multiple people are moving in similar patterns
        moving_people = [p for p in tracked_persons if calculate_movement_speed(person_movement_history.get(p['track_id'], [])) > 10]
        if len(moving_people) >= group_coordination_threshold:
            alert = {
                'person_id': 'group',
                'behavior_type': 'coordinated_movement',
                'timestamp': frame_time,
                'details': f'{len(moving_people)} people moving in coordinated pattern'
            }
            if not any(a['behavior_type'] == 'coordinated_movement' for a in suspicious_behavior_alerts[-10:]):
                suspicious_behavior_alerts.append(alert)

def detect_theft(detections, tracked_persons, frame_time):
    """Detect theft based on object-owner relationships and movement"""
    global theft_alerts, tracked_objects, person_object_associations
    
    # Track all detected objects
    for det in detections:
        if det['class'] in ['bottle', 'cup', 'book', 'cell phone', 'laptop', 'handbag', 'backpack']:
            object_id = f"{det['class']}_{det['confidence']:.2f}"
            bbox = det['bbox']
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            
            if object_id not in tracked_objects:
                tracked_objects[object_id] = {
                    'bbox': bbox,
                    'owner_id': None,
                    'last_seen': frame_time,
                    'movement_history': deque(maxlen=30)
                }
            
            tracked_objects[object_id]['bbox'] = bbox
            tracked_objects[object_id]['last_seen'] = frame_time
            tracked_objects[object_id]['movement_history'].append({
                'center': center,
                'timestamp': frame_time
            })
    
    # Check for theft scenarios
    for object_id, obj_data in tracked_objects.items():
        if frame_time - obj_data['last_seen'] < 5:  # Object seen recently
            # Check if object is moving toward exit zones
            current_center = ((obj_data['bbox'][0] + obj_data['bbox'][2]) // 2, 
                            (obj_data['bbox'][1] + obj_data['bbox'][3]) // 2)
            
            for exit_zone in exit_zones:
                if (exit_zone[0] <= current_center[0] <= exit_zone[2] and 
                    exit_zone[1] <= current_center[1] <= exit_zone[3]):
                    
                    # Check if owner is nearby
                    owner_nearby = False
                    if obj_data['owner_id']:
                        for person in tracked_persons:
                            if person['track_id'] == obj_data['owner_id']:
                                person_center = ((person['bbox'][0] + person['bbox'][2]) // 2,
                                               (person['bbox'][1] + person['bbox'][3]) // 2)
                                distance = math.sqrt((current_center[0] - person_center[0])**2 + 
                                                   (current_center[1] - person_center[1])**2)
                                if distance < 100:  # Owner within 100 pixels
                                    owner_nearby = True
                                    break
                    
                    if not owner_nearby:
                        alert = {
                            'object_id': object_id,
                            'owner_id': obj_data['owner_id'],
                            'timestamp': frame_time,
                            'type': 'exit_zone_theft',
                            'details': f'Object {object_id} in exit zone without owner'
                        }
                        if not any(a['object_id'] == object_id and a['type'] == 'exit_zone_theft' for a in theft_alerts[-10:]):
                            theft_alerts.append(alert)

# Initialize movement history tracking
person_movement_history = {} 

@app.get("/performance")
def get_performance():
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    return {
        "fps": VIDEO_FPS,
        "frame_count": frame_count,
        "avg_processing_time": avg_processing_time,
        "processing_every_n_frames": PROCESS_EVERY_N_FRAMES
    } 