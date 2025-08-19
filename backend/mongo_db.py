from pymongo import MongoClient
from datetime import datetime
import json
import certifi

class MongoDBManager:
    def __init__(self, connection_string="mongodb+srv://ajiloredaniel58:dal4X36nsrJQFopL@cluster7.qszpoc1.mongodb.net/"):
        self.client = MongoClient(connection_string, tlsCAFile=certifi.where())
        self.db = self.client.obex_security
        self.alerts_collection = self.db.alerts
        self.snapshots_collection = self.db.snapshots
        self.known_faces_collection = self.db.known_faces

    def save_known_face(self, name, encoding, image_path=None):
        """Save a known face encoding to MongoDB"""
        face_data = {
            "name": name,
            "encoding": encoding.tolist(),
            "image_path": image_path,
            "timestamp": datetime.now()
        }
        return self.known_faces_collection.insert_one(face_data)

    def get_known_faces(self):
        """Get all known face encodings from MongoDB"""
        return list(self.known_faces_collection.find({}))

    def save_recognized_face(self, name, encoding, image_path=None):
        """Save a recognized face event to MongoDB"""
        rec_data = {
            "name": name,
            "encoding": encoding.tolist(),
            "image_path": image_path,
            "timestamp": datetime.now()
        }
        return self.db.recognized_faces.insert_one(rec_data)

    def get_recent_recognized_faces(self, limit=50):
        """Get recent recognized faces from MongoDB"""
        return list(self.db.recognized_faces.find({}).sort("timestamp", -1).limit(limit))
    
    def save_loitering_alert(self, track_id, entry_time, duration, snapshot_path=None):
        """Save loitering alert to MongoDB"""
        alert_data = {
            "type": "loitering",
            "track_id": track_id,
            "entry_time": entry_time,
            "duration": duration,
            "timestamp": datetime.now(),
            "snapshot_path": snapshot_path
        }
        return self.alerts_collection.insert_one(alert_data)
    
    def save_intrusion_alert(self, track_id, entry_time):
        """Save intrusion alert to MongoDB"""
        alert_data = {
            "type": "intrusion",
            "track_id": track_id,
            "entry_time": entry_time,
            "timestamp": datetime.now()
        }
        return self.alerts_collection.insert_one(alert_data)
    
    def get_recent_alerts(self, alert_type=None, limit=50):
        """Get recent alerts from MongoDB"""
        query = {"type": alert_type} if alert_type else {}
        return list(self.alerts_collection.find(query).sort("timestamp", -1).limit(limit))
    
    def save_snapshot_metadata(self, track_id, snapshot_path, alert_type="loitering"):
        """Save snapshot metadata to MongoDB"""
        snapshot_data = {
            "track_id": track_id,
            "snapshot_path": snapshot_path,
            "alert_type": alert_type,
            "timestamp": datetime.now()
        }
        return self.snapshots_collection.insert_one(snapshot_data)

# Initialize MongoDB manager
mongo_manager = MongoDBManager()