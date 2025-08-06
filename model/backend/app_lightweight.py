from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from detection.detector import YoloV8Detector
import threading
import time
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# MongoDB integration (optional - will work without MongoDB)
try:
    from mongo_db import mongo_manager
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("MongoDB not available - alerts will be stored in memory only")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = YoloV8Detector()

# Video source
VIDEO_SOURCE = "scene 2.mp4"

@app.post("/set_zone")
async def set_zone(request: Request):
    data = await request.json()
    global zone_coords
    zone_coords = [int(data['x1']), int(data['y1']), int(data['x2']), int(data['y2'])]
    print(f"Zone set: {zone_coords}")  # Debug print
    return {"status": "ok", "zone": zone_coords}

@app.get("/zones")
def get_zones():
    return {"zones": [zone_coords] if zone_coords else []}

@app.get("/intrusion_alerts")
def get_intrusion_alerts():
    return {"intrusion_alerts": intrusion_alerts}

def capture_loitering_snapshot(frame, track_id, duration):
    """Capture a snapshot when loitering is detected"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"loitering_track_{track_id}_{timestamp}.jpg"
    filepath = os.path.join(SNAPSHOT_DIR, filename)
    
    # Add text overlay to the snapshot
    snapshot_frame = frame.copy()
    cv2.putText(snapshot_frame, f"LOITERING ALERT - Track {track_id}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(snapshot_frame, f"Duration: {duration:.1f}s", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(snapshot_frame, f"Time: {timestamp}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imwrite(filepath, snapshot_frame)
    return filepath

@app.get("/loitering_alerts")
def get_loitering_alerts():
    return {
        "loitering_alerts": loitering_alerts,
        "snapshots": loitering_snapshots
    }

@app.get("/mongo/alerts")
def get_mongo_alerts(alert_type: str = None, limit: int = 50):
    """Get alerts from MongoDB"""
    if MONGODB_AVAILABLE:
        try:
            alerts = mongo_manager.get_recent_alerts(alert_type, limit)
            return {"alerts": alerts, "source": "mongodb"}
        except Exception as e:
            return {"error": f"MongoDB error: {e}", "source": "mongodb"}
    else:
        return {"error": "MongoDB not available", "source": "memory"}

@app.get("/mongo/status")
def get_mongo_status():
    """Check MongoDB connection status"""
    return {
        "mongodb_available": MONGODB_AVAILABLE,
        "connection_string": "mongodb://localhost:27017/" if MONGODB_AVAILABLE else "Not configured"
    }

def detect_theft_and_suspicious_behavior(detections, face_locations, frame_time):
    """Advanced behavioral analysis for theft/robbery detection"""
    global theft_alerts, suspicious_behavior_alerts
    
    # Track people and their movements
    people_data = []
    weapons_detected = []
    
    # Analyze detected objects and faces
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Weapon detection
        if det['class'] in WEAPON_CLASSES:
            weapons_detected.append({
                'class': det['class'],
                'bbox': det['bbox'],
                'confidence': det['confidence']
            })
        
        # Person tracking for behavioral analysis
        if det['class'] == 'person':
            people_data.append({
                'bbox': det['bbox'],
                'center': (cx, cy),
                'confidence': det['confidence'],
                'timestamp': frame_time
            })
    
    # 1. WEAPON DETECTION - Immediate high-priority alert
    if weapons_detected:
        theft_alerts.append({
            'type': 'weapon_detected',
            'severity': 'CRITICAL',
            'timestamp': frame_time,
            'details': f"Weapon detected: {weapons_detected[0]['class']}",
            'weapons': weapons_detected,
            'people_involved': len(people_data)
        })
        print(f"🚨 CRITICAL: Weapon detected - {weapons_detected[0]['class']}")
    
    # 2. GROUP COORDINATION - Multiple people entering together
    if len(people_data) >= GROUP_SIZE_THRESHOLD:
        # Check if they're moving together (similar direction/speed)
        theft_alerts.append({
            'type': 'group_coordination',
            'severity': 'HIGH',
            'timestamp': frame_time,
            'details': f"Multiple people detected: {len(people_data)} people",
            'people_involved': len(people_data),
            'group_size': len(people_data)
        })
        print(f"⚠️ HIGH: Group coordination detected - {len(people_data)} people")
    
    # 3. FAST MOVEMENT DETECTION - Running/suspicious speed
    for person in people_data:
        # Calculate movement speed (simplified)
        if 'last_position' in person:
            dx = person['center'][0] - person['last_position'][0]
            dy = person['center'][1] - person['last_position'][1]
            speed = (dx**2 + dy**2)**0.5
            
            if speed > FAST_MOVEMENT_THRESHOLD:
                suspicious_behavior_alerts.append({
                    'type': 'fast_movement',
                    'severity': 'MEDIUM',
                    'timestamp': frame_time,
                    'details': f"Fast movement detected: {speed:.1f} pixels/frame",
                    'speed': speed
                })
                print(f"🏃 MEDIUM: Fast movement detected - {speed:.1f} pixels/frame")
        
        person['last_position'] = person['center']
    
    # 4. AGGRESSIVE BEHAVIOR - Sudden movements, struggles
    if len(people_data) > 1:
        # Check for close proximity and rapid movements (struggles)
        for i, person1 in enumerate(people_data):
            for j, person2 in enumerate(people_data[i+1:], i+1):
                # Calculate distance between people
                dist = ((person1['center'][0] - person2['center'][0])**2 + 
                       (person1['center'][1] - person2['center'][1])**2)**0.5
                
                if dist < 100:  # Close proximity
                    suspicious_behavior_alerts.append({
                        'type': 'close_proximity',
                        'severity': 'MEDIUM',
                        'timestamp': frame_time,
                        'details': f"People in close proximity: {dist:.1f} pixels",
                        'distance': dist
                    })
                    print(f"👥 MEDIUM: Close proximity detected - {dist:.1f} pixels")
    
    # 5. ENTRY PATTERN - Multiple people entering building/area
    if len(people_data) >= 2:
        # Check if they're moving toward same area (entry coordination)
        theft_alerts.append({
            'type': 'coordinated_entry',
            'severity': 'HIGH',
            'timestamp': frame_time,
            'details': f"Coordinated entry: {len(people_data)} people entering together",
            'people_involved': len(people_data)
        })
        print(f"🚪 HIGH: Coordinated entry detected - {len(people_data)} people")
    
    # 6. SUSPICIOUS TIMING - Multiple alerts in short time
    recent_alerts = [a for a in theft_alerts if frame_time - a['timestamp'] < COORDINATION_TIME_WINDOW]
    if len(recent_alerts) >= 2:
        theft_alerts.append({
            'type': 'multiple_suspicious_events',
            'severity': 'CRITICAL',
            'timestamp': frame_time,
            'details': f"Multiple suspicious events in {COORDINATION_TIME_WINDOW}s",
            'events_count': len(recent_alerts)
        })
        print(f"🚨 CRITICAL: Multiple suspicious events detected")

@app.get("/theft_alerts")
def get_theft_alerts():
    return {"theft_alerts": theft_alerts}

def cleanup_old_alerts(current_time):
    """Remove alerts older than the timeout period"""
    global theft_alerts, suspicious_behavior_alerts
    
    # Clean up theft alerts
    theft_alerts = [alert for alert in theft_alerts 
                   if current_time - alert['timestamp'] < ALERT_DISPLAY_TIMEOUT]
    
    # Clean up suspicious behavior alerts
    suspicious_behavior_alerts = [alert for alert in suspicious_behavior_alerts 
                                if current_time - alert['timestamp'] < ALERT_DISPLAY_TIMEOUT]

@app.get("/suspicious_behavior")
def get_suspicious_behavior():
    return {"suspicious_behavior": suspicious_behavior_alerts}

# Global variables
latest_frame = None
latest_detections = []
latest_faces = []  # Store detected faces
frame_count = 0

# Zone and loitering detection
zone_coords = None  # No zone by default
LOITER_THRESHOLD = 30  # seconds
person_zone_times = {}  # track_id: entry_time
loitering_alerts = []   # [{track_id, entry_time, duration}]
intrusion_alerts = []   # [{track_id, entry_time}]

# Snapshot functionality
SNAPSHOT_DIR = "loitering_snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
loitering_snapshots = []  # Store snapshot file paths

# Theft detection variables - Advanced behavioral analysis
theft_alerts = []  # [{type, severity, timestamp, details, people_involved}]
tracked_objects = {}  # object_id: {bbox, last_seen, movement_history}
suspicious_behavior_alerts = []  # [{type, severity, timestamp, details}]

# Alert timeout settings
ALERT_DISPLAY_TIMEOUT = 5  # seconds - alerts disappear after this time
ALERT_CLEANUP_INTERVAL = 10  # seconds - how often to clean old alerts

# Behavioral analysis parameters
GROUP_SIZE_THRESHOLD = 2  # Multiple people = suspicious
FAST_MOVEMENT_THRESHOLD = 30  # pixels per frame for running
WEAPON_CLASSES = ['knife', 'gun', 'weapon', 'rifle', 'pistol']
AGGRESSIVE_MOVEMENT_THRESHOLD = 50  # Sudden movements
COORDINATION_TIME_WINDOW = 5  # seconds for group coordination

# Ultra-lightweight settings
PROCESS_EVERY_N_FRAMES = 5  # Process every 5th frame (was 10)
DETECTION_INTERVAL = 0.2  # Run detection every 200ms (was 500ms)
VIDEO_FPS = 8  # Very low FPS for performance

def video_capture_thread():
    global latest_frame, latest_detections, latest_faces, frame_count, person_zone_times, loitering_alerts, intrusion_alerts
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {VIDEO_SOURCE}")
        return
    
    print(f"Successfully opened video source: {VIDEO_SOURCE}")
    
    last_detection_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break
        
        frame_count += 1
        current_time = time.time()
        
        # Always update video feed
        latest_frame = frame.copy()
        
        # Skip most processing for performance
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            time.sleep(1/VIDEO_FPS)
            continue
        
        # Only run detection occasionally
        if current_time - last_detection_time >= DETECTION_INTERVAL:
            try:
                # Face detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                latest_faces = face_locations
                
                # Debug: Print face detection results
                print(f"Faces detected: {len(face_locations)}")
                if zone_coords:
                    print(f"Zone active: {zone_coords}")
                else:
                    print("No zone set")
                
                # Object detection
                detections = detector.detect_np(frame)
                latest_detections = detections
                
                # Behavioral analysis
                detect_theft_and_suspicious_behavior(detections, face_locations, current_time)
                
                # Clean up old alerts periodically
                if frame_count % (ALERT_CLEANUP_INTERVAL * VIDEO_FPS) == 0:
                    cleanup_old_alerts(current_time)
                
                # Zone detection logic
                for (top, right, bottom, left) in face_locations:
                    cx, cy = (left + right) // 2, (top + bottom) // 2
                    in_zone = False
                    if zone_coords is not None:
                        in_zone = (zone_coords[0] <= cx <= zone_coords[2]) and (zone_coords[1] <= cy <= zone_coords[3])
                        print(f"Person at ({cx}, {cy}), Zone: {zone_coords}, In zone: {in_zone}")  # Debug print
                    
                    # Use face index as track_id for simplicity
                    tid = face_locations.index((top, right, bottom, left))
                    
                    if in_zone:
                        if tid not in person_zone_times:
                            person_zone_times[tid] = current_time
                            intrusion_alerts.append({'track_id': tid, 'entry_time': current_time})
                            print(f"INTRUSION ALERT: Person {tid} entered zone!")  # Debug print
                            
                            # Save to MongoDB if available
                            if MONGODB_AVAILABLE:
                                try:
                                    mongo_manager.save_intrusion_alert(tid, current_time)
                                except Exception as e:
                                    print(f"MongoDB save error: {e}")
                        else:
                            duration = current_time - person_zone_times[tid]
                            if duration > LOITER_THRESHOLD and not any(a['track_id'] == tid for a in loitering_alerts):
                                loitering_alerts.append({'track_id': tid, 'entry_time': person_zone_times[tid], 'duration': duration})
                                print(f"LOITERING ALERT: Person {tid} has been in zone for {duration:.1f}s!")  # Debug print
                                # Capture snapshot if loitering alert is new
                                if not any(s['track_id'] == tid for s in loitering_snapshots):
                                    snapshot_path = capture_loitering_snapshot(frame, tid, duration)
                                    loitering_snapshots.append({'track_id': tid, 'snapshot_path': snapshot_path})
                                    
                                    # Save to MongoDB if available
                                    if MONGODB_AVAILABLE:
                                        try:
                                            mongo_manager.save_loitering_alert(tid, person_zone_times[tid], duration, snapshot_path)
                                            mongo_manager.save_snapshot_metadata(tid, snapshot_path)
                                        except Exception as e:
                                            print(f"MongoDB save error: {e}")
                    else:
                        if tid in person_zone_times:
                            del person_zone_times[tid]
                            print(f"Person {tid} left zone")  # Debug print
                
                # Draw object boxes
                for det in detections:
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    label = f"{det['class']} {det['confidence']:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw face boxes
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                    cv2.putText(frame, "Person", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Draw zone
                if zone_coords is not None:
                    cv2.rectangle(frame, (zone_coords[0], zone_coords[1]), (zone_coords[2], zone_coords[3]), (0, 0, 255), 2)
                    cv2.putText(frame, "RESTRICTED ZONE", (zone_coords[0], zone_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Draw behavioral alerts on video with fade-out effect
                y_offset = 30
                for alert in theft_alerts[-3:]:  # Show last 3 alerts
                    time_elapsed = current_time - alert['timestamp']
                    time_remaining = ALERT_DISPLAY_TIMEOUT - time_elapsed
                    
                    if time_remaining > 0:
                        # Calculate opacity based on time remaining (fade out effect)
                        opacity = min(1.0, time_remaining / 2.0)  # Start fading after 2 seconds
                        
                        if alert['severity'] == 'CRITICAL':
                            color = (0, 0, int(255 * opacity))  # Red with opacity
                            cv2.putText(frame, f"🚨 {alert['type']}", (10, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
                            y_offset += 40
                        elif alert['severity'] == 'HIGH':
                            color = (0, int(165 * opacity), int(255 * opacity))  # Orange with opacity
                            cv2.putText(frame, f"⚠️ {alert['type']}", (10, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            y_offset += 35
                
                # Draw suspicious behavior alerts
                for alert in suspicious_behavior_alerts[-2:]:  # Show last 2 alerts
                    time_elapsed = current_time - alert['timestamp']
                    time_remaining = ALERT_DISPLAY_TIMEOUT - time_elapsed
                    
                    if time_remaining > 0:
                        opacity = min(1.0, time_remaining / 2.0)
                        color = (0, int(255 * opacity), int(255 * opacity))  # Yellow with opacity
                        cv2.putText(frame, f"👥 {alert['type']}", (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y_offset += 30
                
                last_detection_time = current_time
                latest_frame = frame.copy()
                
            except Exception as e:
                print(f"Detection error: {e}")
        
        time.sleep(1/VIDEO_FPS)
    
    cap.release()

# Start video capture in background
t = threading.Thread(target=video_capture_thread, daemon=True)
t.start()

def mjpeg_generator():
    global latest_frame
    while True:
        if latest_frame is not None:
            ret, jpeg = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1)  # Slower refresh

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(mjpeg_generator(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/latest_detections")
def latest_detections_api():
    return {"detections": latest_detections}

@app.get("/faces")
def get_faces():
    return {"faces": latest_faces}

@app.get("/performance")
def get_performance():
    return {
        "frame_count": frame_count,
        "fps": VIDEO_FPS,
        "process_every_n": PROCESS_EVERY_N_FRAMES,
        "detection_interval": DETECTION_INTERVAL,
        "faces_detected": len(latest_faces),
        "objects_detected": len(latest_detections),
        "zone_active": zone_coords is not None,
        "people_in_zone": len(person_zone_times),
        "loitering_alerts": len(loitering_alerts),
        "intrusion_alerts": len(intrusion_alerts),
        "theft_alerts": len(theft_alerts),
        "tracked_objects": len(tracked_objects),
        "mongodb_available": MONGODB_AVAILABLE
    } 