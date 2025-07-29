from ultralytics import YOLO
import cv2
from typing import List, Dict

class YoloV8Detector:
    def __init__(self, model_path: str = "../../models/yolov8/yolov8s.pt"):
        self.model = YOLO(model_path)

    def detect(self, image_path: str) -> List[Dict]:
        img = cv2.imread(image_path)
        results = self.model(img)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": self.model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": [float(x) for x in box.xyxy[0].tolist()]
                })
        return detections

    def detect_np(self, img_np) -> List[Dict]:
        results = self.model(img_np)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": self.model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": [float(x) for x in box.xyxy[0].tolist()]
                })
        return detections 