from pymongo import MongoClient
from datetime import datetime
import json

class MongoDBManager:
    def __init__(self, connection_string="mongodb://localhost:27017/"):
        self.client = MongoClient(connection_string)
        self.db = self.client.obex_security
        self.alerts_collection = self.db.alerts
        self.snapshots_collection = self.db.snapshots
        
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