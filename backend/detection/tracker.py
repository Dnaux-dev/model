from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class DeepSortTracker:
    def __init__(self):
        # You can tune parameters like max_age, n_init, max_cosine_distance if needed
        self.tracker = DeepSort(max_age=30)

    def update(self, detections, frame):
        """
        Update the tracker with new detections.

        detections: list of dicts like:
        {
            'bbox': [x1, y1, x2, y2],
            'class': 'person',
            'confidence': 0.92
        }
        frame: numpy array of the current frame
        """

        # Only track 'person' class
        person_dets = [d for d in detections if d['class'] == 'person']
        valid_boxes = []

        for d in person_dets:
            box = d['bbox']

            # Ensure bbox is valid (4 numbers)
            if (
                isinstance(box, (list, tuple)) and
                len(box) == 4 and
                all(isinstance(x, (int, float)) for x in box)
            ):
                confidence = d.get('confidence', 1.0)  # default to 1.0 if missing
                class_name = d.get('class', 'person')  # default to 'person'

                # DeepSort expects: ([x1, y1, x2, y2], confidence, class_name)
                detection_tuple = (list(box), confidence, class_name)
                valid_boxes.append(detection_tuple)

        # 🚨 LOGGING: See what we are about to send
        print(f"[DeepSortTracker] Sending {len(valid_boxes)} detections to DeepSort:")
        for vb in valid_boxes:
            print("   →", vb)

        # ✅ Defensive check to avoid bad input
        if not (isinstance(valid_boxes, list) and valid_boxes and all(
            isinstance(b, tuple) and len(b) == 3 and isinstance(b[0], list) and len(b[0]) == 4
            for b in valid_boxes
        )):
            print("[DeepSortTracker] Skipping update_tracks due to invalid input:", valid_boxes)
            return []

        # 🔄 Update DeepSort with detections
        tracks = self.tracker.update_tracks(valid_boxes, frame=frame)

        # 📦 Collect results
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            results.append({
                'track_id': track_id,
                'bbox': ltrb,
                'class': 'person',
            })

        return results
