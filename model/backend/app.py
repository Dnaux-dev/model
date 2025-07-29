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

VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
latest_detections = []
latest_frame = None
latest_faces = []  # Store latest detected faces

tracker = DeepSortTracker()

# Replace hardcoded ZONE with a mutable global
zone_coords = None  # No zone by default

person_zone_times = {}  # track_id: entry_time
loitering_alerts = []   # [{track_id, entry_time, duration}]
intrusion_alerts = []  

# Heatmap tracking
heatmap = None
prev_gray = None
HEATMAP_DECAY = 0.95  # Decay factor for fading old motion
HEATMAP_ALPHA = 0.5   # Overlay strength

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

# Background thread for video capture and detection
def video_capture_thread():
    global latest_detections, latest_frame, latest_faces
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Low-light enhancement using CLAHE
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        if heatmap is None:
            heatmap = np.zeros_like(gray, dtype=np.float32)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, motion_mask = cv2.threshold(diff, 25, 1, cv2.THRESH_BINARY)
            heatmap = HEATMAP_DECAY * heatmap + motion_mask.astype(np.float32)
        prev_gray = gray.copy()
        # Face detection (on enhanced frame)
        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        latest_faces = face_locations
        # Run detection on the enhanced frame (use detect_np for efficiency)
        detections = detector.detect_np(enhanced_frame)
        # Person tracking, zone logic, etc. (unchanged)
        tracked_persons = tracker.update(detections, enhanced_frame)
        now = pytime.time()
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
        # Draw overlays
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
        # Overlay heatmap
        if heatmap is not None:
            norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            color_map = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            enhanced_frame = cv2.addWeighted(enhanced_frame, 1 - HEATMAP_ALPHA, color_map, HEATMAP_ALPHA, 0)
        latest_frame = enhanced_frame.copy()
        time.sleep(0.05)  # ~20 FPS
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