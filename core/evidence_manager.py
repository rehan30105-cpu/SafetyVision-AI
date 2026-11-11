import os
import cv2
import time
from datetime import datetime
from database.db import get_db

EVIDENCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'evidence')
os.makedirs(EVIDENCE_DIR, exist_ok=True)

class EvidenceManager:
    def __init__(self, max_evidence_per_session=50):
        self.max_evidence_per_session = max_evidence_per_session
        self.captured_count = 0

    def capture_frame(self, frame, incident_data, detections=None):
        """
        Annotates and saves an evidence frame on disk and records it in the database.
        frame: numpy BGR image
        incident_data: dict with 'incident_id', 'incident_uid', 'person_id', 'violation_type', 'risk_score', 'camera_id', 'zone_id'
        """
        if self.captured_count >= self.max_evidence_per_session or frame is None:
            return None

        try:
            annotated = frame.copy()
            h, w = annotated.shape[:2]

            # Draw header banner
            cv2.rectangle(annotated, (0, 0), (w, 45), (15, 23, 42), -1) # Dark navy
            header_text = f"SAFETYVISION AI EVIDENCE | INCIDENT: {incident_data.get('incident_uid', 'N/A')}"
            cv2.putText(annotated, header_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Draw footer metadata banner
            cv2.rectangle(annotated, (0, h - 40), (w, h), (15, 23, 42), -1)
            ts_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            meta_text = f"TIME: {ts_str} | PERSON: #{incident_data.get('person_id', 'N/A')} | VIOLATION: {incident_data.get('violation_type', 'N/A')} | RISK: {incident_data.get('risk_score', 0)}/100"
            cv2.putText(annotated, meta_text, (15, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1)

            # Generate filename
            uid = incident_data.get('incident_uid', f"INC_{int(time.time())}")
            filename = f"evidence_{uid}_{int(time.time())}.jpg"
            rel_path = f"static/evidence/{filename}"
            abs_path = os.path.join(EVIDENCE_DIR, filename)

            # Save JPEG
            cv2.imwrite(abs_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            file_size = os.path.getsize(abs_path) if os.path.exists(abs_path) else 0

            # Store in Database
            conn = get_db()
            conn.execute('''
                INSERT INTO evidence (incident_id, person_tracking_id, violation_type, risk_score, camera_id, zone_id, image_path, file_size, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                incident_data.get('incident_db_id'),
                incident_data.get('person_id'),
                incident_data.get('violation_type'),
                incident_data.get('risk_score'),
                incident_data.get('camera_id'),
                incident_data.get('zone_id'),
                rel_path,
                file_size,
                incident_data.get('user_id')
            ))
            conn.commit()
            conn.close()

            self.captured_count += 1
            return rel_path
        except Exception as e:
            print(f"Error capturing evidence: {e}")
            return None
