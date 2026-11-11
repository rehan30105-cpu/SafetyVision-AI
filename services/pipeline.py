import time
import threading
import uuid
import json
from datetime import datetime
from database.db import get_db, get_setting
from core.model_manager import ModelManager
from core.tracker import PersonTracker
from core.ppe_association import PPEAssociationEngine
from core.zone_engine import SafetyZoneEngine
from core.context_intelligence import ContextIntelligenceEngine
from core.risk_engine import DynamicRiskEngine
from core.evidence_manager import EvidenceManager
from core.ppe_metrics import summarize_ppe

class SafetyVisionPipeline:
    def __init__(self):
        self.model_mgr = ModelManager.get_instance()
        self.tracker = PersonTracker(max_lost_frames=30, min_iou=0.20)
        self._trackers_by_user = {}
        self._tracker_locks = {}
        self._lock = threading.RLock()
        self.ppe_engine = PPEAssociationEngine(hardhat_weight=33.33, mask_weight=33.33, vest_weight=33.34)
        self.zone_engine = SafetyZoneEngine()
        self.context_engine = ContextIntelligenceEngine(proximity_threshold_px=140.0, required_cones=4)
        self.risk_engine = DynamicRiskEngine()
        self._evidence_managers = {}

    def process_frame(self, frame, camera_name="Laptop Webcam", auto_persist=True, user_id=None):
        """
        Executes end-to-end AI Safety pipeline for a controlled detection cycle.
        Returns: full person-wise analysis dictionary and state-based alerts.
        """
        # Keep person tracking and active incident IDs isolated by authenticated owner.
        with self._lock:
            tracker = self._trackers_by_user.setdefault(user_id, PersonTracker(max_lost_frames=30, min_iou=0.20))
            owner_lock = self._tracker_locks.setdefault(user_id, threading.RLock())
        with owner_lock:
            return self._process_frame_for_owner(frame, camera_name, auto_persist, user_id, tracker)

    def _process_frame_for_owner(self, frame, camera_name, auto_persist, user_id, tracker):
        """Run one frame while the per-owner tracker lock is held."""
        h, w = frame.shape[:2]
        current_time = time.time()

        # Step 1: Centralized YOLO Inference (10 Classes)
        conf_thresh = float(get_setting('confidence_threshold', 0.35))
        iou_thresh = float(get_setting('iou_threshold', 0.45))
        detections, _ = self.model_mgr.predict(frame, conf=conf_thresh, iou=iou_thresh)

        # Step 2: Semantic Domain Separation
        person_detections = []
        ppe_detections = []
        hazard_detections = []

        for d in detections:
            cname = d['class_name']
            if cname == 'Person':
                person_detections.append(d)
            elif cname in ['Hardhat', 'NO-Hardhat', 'Mask', 'NO-Mask', 'Safety Vest', 'NO-Safety Vest']:
                ppe_detections.append(d)
            elif cname in ['machinery', 'vehicle', 'Safety Cone']:
                hazard_detections.append(d)

        # Step 3: Multi-Person Tracking (Temporary session IDs: Person #1, Person #2)
        tracked_persons = tracker.update(person_detections, current_time)

        # Step 4: Individual Person-PPE Association
        tracked_persons = self.ppe_engine.associate(tracked_persons, ppe_detections)

        # Step 5: Environmental Context Analysis (Machinery / Vehicle Proximity & Safety Cones)
        context_data = self.context_engine.analyze(tracked_persons, hazard_detections, frame_shape=(h, w))

        # Step 6: Simple, Explainable Risk Classification & Compliance Status
        for person in tracked_persons:
            self.risk_engine.evaluate_person_risk(person)

        # Step 7: State-Based Incident Management & Smart Alerting (NO FRAME SPAM)
        new_alerts = []
        persisted_incidents = []

        if auto_persist:
            conn = get_db()
            cursor = conn.cursor()

            for person in tracked_persons:
                pid = person.track_id
                current_st = person.status

                # State 1: Person has a violation
                if current_st in ['PPE VIOLATION', 'CRITICAL PPE VIOLATION']:
                    if person.active_incident_db_id is None:
                        # New violation episode triggered! Create ONE incident row in DB
                        inc_uid = f"INC-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
                        v_desc = ", ".join(person.active_violations) if person.active_violations else "Missing PPE Gear"

                        cursor.execute('''
                            INSERT INTO incidents (
                                incident_uid, person_tracking_id, violation_type, confidence,
                                duration_seconds, zone_name, camera_name,
                                risk_score, severity, status, operator, user_id
                            ) VALUES (?, ?, ?, 0.88, 1.0, 'Laptop Camera Area', ?, ?, ?, 'DETECTED', 'System AI', ?)
                        ''', (inc_uid, pid, v_desc, camera_name, person.risk_score, person.risk_level, user_id))
                        
                        db_id = cursor.lastrowid
                        conn.commit()

                        person.active_incident_db_id = db_id
                        person.active_incident_uid = inc_uid
                        person.violation_start_time = current_time

                        # Capture ONE high-quality evidence image
                        ev_data = {
                            'incident_db_id': db_id,
                            'incident_uid': inc_uid,
                            'person_id': pid,
                            'violation_type': v_desc,
                            'risk_score': person.risk_score,
                            'camera_id': 1,
                            'zone_id': None,
                            'user_id': user_id
                        }
                        evidence_manager = self._evidence_managers.setdefault(
                            user_id, EvidenceManager(max_evidence_per_session=50))
                        evidence_path = evidence_manager.capture_frame(frame, ev_data)
                        if evidence_path:
                            cursor.execute('UPDATE incidents SET evidence_image = ? WHERE id = ? AND user_id = ?', (evidence_path, db_id, user_id))
                            conn.commit()

                        # Emit ONE state-based alert
                        alert_payload = {
                            'type': 'VIOLATION',
                            'person_id': pid,
                            'title': f"Person #{pid}: {person.status}",
                            'message': person.risk_reason,
                            'severity': person.risk_level,
                            'incident_uid': inc_uid,
                            'evidence_image': evidence_path
                        }
                        new_alerts.append(alert_payload)
                        persisted_incidents.append(alert_payload)

                        # Record in notifications table
                        cursor.execute('''
                            INSERT INTO notifications (title, message, severity, category, incident_id, user_id)
                            VALUES (?, ?, ?, 'VIOLATION', ?, ?)
                        ''', (f"Person #{pid} {person.status}", person.risk_reason, person.risk_level, db_id, user_id))
                        conn.commit()

                    else:
                        # Same person remains visible with ongoing violation -> UPDATE duration only, NO new alert/incident
                        dur = max(1.0, round(current_time - (person.violation_start_time or current_time), 1))
                        cursor.execute('''
                            UPDATE incidents SET duration_seconds = ?, risk_score = ?
                            WHERE id = ? AND user_id = ?
                        ''', (dur, person.risk_score, person.active_incident_db_id, user_id))
                        conn.commit()

                # State 2: Person is now compliant
                elif current_st == 'COMPLIANT':
                    if person.active_incident_db_id is not None:
                        # State changed from VIOLATION -> COMPLIANT! Worker put on their safety gear!
                        cursor.execute('''
                            UPDATE incidents
                            SET status = 'RESOLVED', resolved_at = CURRENT_TIMESTAMP,
                                resolution_notes = 'PPE gear verified. Compliance restored.'
                            WHERE id = ? AND user_id = ?
                        ''', (person.active_incident_db_id, user_id))
                        conn.commit()

                        person.active_incident_db_id = None
                        person.active_incident_uid = None
                        person.violation_start_time = None

                        # Emit ONE compliance restored alert
                        restored_alert = {
                            'type': 'COMPLIANCE_RESTORED',
                            'person_id': pid,
                            'title': f"Person #{pid}: Compliance Restored ✓",
                            'message': "All mandated PPE items are now present.",
                            'severity': 'LOW'
                        }
                        new_alerts.append(restored_alert)

                        cursor.execute('''
                            INSERT INTO notifications (title, message, severity, category, user_id)
                            VALUES (?, ?, 'LOW', 'VIOLATION', ?)
                        ''', (f"Person #{pid}: Compliance Restored", "All required PPE gear verified on worker.", user_id))
                        conn.commit()

                person.previous_status = current_st

            conn.close()

        # Step 8: Build Individual Person Results List (MANDATORY REQUIREMENT)
        person_results = [p.to_person_result() for p in tracked_persons]

        # Step 9: Aggregate Summary Statistics
        # Canonical rule: PPE violations = individually missing required items
        # (Hardhat/Mask/Safety Vest); non-compliant people tracked separately.
        total_p = len(tracked_persons)
        person_results = [p.to_person_result() for p in tracked_persons]
        ppe_summary = summarize_ppe(person_results)
        comp_p = ppe_summary['compliant']
        non_comp_p = ppe_summary['non_compliant_people']
        item_viol_p = ppe_summary['ppe_violations']
        crit_p = sum(1 for p in tracked_persons if p.risk_level == 'CRITICAL')
        
        avg_comp = round(sum(p.compliance_score for p in tracked_persons) / max(total_p, 1), 1) if total_p > 0 else 'N/A'
        avg_risk = round(sum(p.risk_score for p in tracked_persons) / max(total_p, 1), 1) if total_p > 0 else 0.0

        num_comp_rate = float(avg_comp) if avg_comp != 'N/A' else 100.0
        site_score = DynamicRiskEngine.calculate_site_safety_score(
            avg_risk=avg_risk,
            compliance_rate=num_comp_rate,
            critical_count=crit_p,
            total_incidents=len(persisted_incidents)
        )

        # Output logging per Phase 24
        if total_p > 0:
            print(f"[DETECTION LIVE] People: {total_p} | Compliant: {comp_p} | Non-compliant people: {non_comp_p} | PPE violations (items): {item_viol_p} | Overall Compliance: {avg_comp}%")
            for pr in person_results:
                print(f"  [LIVE PERSON] {pr['label']}: Comp={pr['compliance_score']}% | Hardhat={pr['hardhat_label']} | Mask={pr['mask_label']} | Vest={pr['vest_label']}")

        return {
            'fps': self.model_mgr.last_fps,
            'latency_ms': self.model_mgr.last_latency_ms,
            'device': self.model_mgr.device.upper(),
            'device_name': self.model_mgr.device_name,
            'frame_size': [w, h],
            'raw_detections': detections,
            'person_results': person_results,       # Person #1, Person #2 separate results!
            'tracked_persons': person_results,      # Backward compatibility
            'people': person_results,
            'context_intelligence': context_data,
            'new_alerts': new_alerts,               # State-based alerts only!
            'persisted_incidents': persisted_incidents,
            'summary': {
                'people_detected': total_p,
                'compliant': comp_p,
                'non_compliant_people': non_comp_p,
                'violations': item_viol_p,
                'critical': crit_p,
                'overall_compliance': avg_comp
            },
            'live_metrics': {
                'total_persons': total_p,
                'compliant_persons': comp_p,
                'non_compliant_persons': non_comp_p,
                'violating_persons': non_comp_p,
                'ppe_violations': item_viol_p,
                'compliance_rate': avg_comp,
                'critical_count': crit_p,
                'average_risk': avg_risk,
                'site_safety_score': site_score
            }
        }

    def reset_tracker(self, user_id=None):
        with self._lock:
            if user_id is not None:
                tracker = self._trackers_by_user.pop(user_id, None)
                if tracker:
                    tracker.reset()
                self._tracker_locks.pop(user_id, None)
                self._evidence_managers.pop(user_id, None)
                return
            for tracker in self._trackers_by_user.values():
                tracker.reset()
            self._trackers_by_user.clear()
            self._tracker_locks.clear()
            self._evidence_managers.clear()
            self.tracker.reset()

    def get_live_session_person_results(self, user_id, session_uid=None, tracker_ids=None):
        """Snapshot unique currently retained tracks for the authenticated owner."""
        with self._lock:
            tracker = self._trackers_by_user.get(user_id)
            if not tracker:
                return []
            if tracker_ids:
                tracker_ids = {int(track_id) for track_id in tracker_ids}
            live_people = []
            for track in tracker.tracks.values():
                if getattr(track, 'hits', 0) < 2 or (tracker_ids is not None and track.track_id not in tracker_ids):
                    continue
                result = track.to_person_result()
                active_incident_id = getattr(track, 'active_incident_db_id', None)
                if active_incident_id:
                    conn = get_db()
                    row = conn.execute('SELECT user_id,session_uid FROM incidents WHERE id=?', (active_incident_id,)).fetchone()
                    conn.close()
                    if not row or row['user_id'] != user_id:
                        track.active_incident_db_id = None
                        track.active_incident_uid = None
                        active_incident_id = None
                if active_incident_id:
                    result['incident_db_id'] = active_incident_id
                live_people.append(result)
            return live_people
