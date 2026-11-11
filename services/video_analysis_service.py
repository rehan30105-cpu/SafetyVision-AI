"""
SAFETYVISION AI - Dedicated Live AI Video Analysis Service
Provides real-time frame-by-frame video inference, temporal multi-person tracking,
spatial PPE association, state-change timeline event generation (no alert spam),
representative evidence frame capturing, and session lifecycle management.
"""

import os
import cv2
import json
import time
import base64
import numpy as np
from datetime import datetime

from database.db import get_db, get_setting
from core.model_manager import ModelManager
from core.tracker import PersonTracker, TrackedPerson
from core.ppe_association import PPEAssociationEngine
from core.ppe_metrics import summarize_ppe
from core.risk_engine import DynamicRiskEngine
from core.context_intelligence import ContextIntelligenceEngine
from core.annotator import draw_person_annotations
from services.session_service import SessionManager

class VideoSessionState:
    def __init__(self, session_uid, filename, duration_seconds=0.0, user_id=None):
        self.session_uid = session_uid
        self.user_id = user_id
        self.filename = filename
        self.duration_seconds = duration_seconds
        self.start_time = time.time()

        # Dedicated tracker and association engine per video session
        # max_distance is generous to support high-res video without track fragmentation
        self.tracker = PersonTracker(max_lost_frames=35, min_iou=0.10, max_distance=650.0)
        self.ppe_engine = PPEAssociationEngine(hardhat_weight=33.33, mask_weight=33.33, vest_weight=33.34)
        self.risk_engine = DynamicRiskEngine()
        self.context_engine = ContextIntelligenceEngine(proximity_threshold_px=140.0, required_cones=4)

        # Timeline events (meaningful state changes only, no alert spam)
        self.timeline_events = []
        # person_id -> { 'status': str, 'active_violations': set, 'last_event_time': float }
        self.person_state_cache = {}

        # Cumulative person records (for backward compatibility and external testing)
        self.cumulative_persons = {}

        # Representative evidence tracking
        self.best_evidence_frame = None
        self.best_evidence_score = -1.0
        self.best_evidence_time = 0.0
        self.best_evidence_results = []
        self.best_evidence_detections = []

        self.frames_analyzed = 0

    def format_time_str(self, seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

class VideoAnalysisService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = VideoAnalysisService()
        return cls._instance

    def __init__(self):
        self.model_mgr = ModelManager.get_instance()
        self.session_mgr = SessionManager.get_instance()
        self.active_sessions = {} # session_uid -> VideoSessionState

    def start_session(self, filename="uploaded_video.mp4", duration_seconds=0.0, user_id=None):
        """
        Creates a new video analysis session and initializes isolated tracking state.
        Guarantees clean state isolation between runs.
        """
        session_uid, _ = self.session_mgr.create_session('VIDEO', filename, user_id=user_id)
        state = VideoSessionState(session_uid, filename, duration_seconds, user_id=user_id)
        self.active_sessions[session_uid] = state
        print(f"[VIDEO_SERVICE] Initialized Live Video Session: {session_uid} for '{filename}'")
        return session_uid

    def get_or_create_session(self, session_uid, filename="uploaded_video.mp4", duration_seconds=0.0, user_id=None):
        if session_uid in self.active_sessions:
            state = self.active_sessions[session_uid]
            if state.user_id != user_id:
                raise PermissionError('Video session belongs to another account.')
            return state
        persisted = self.session_mgr.get_session(session_uid, user_id=user_id)
        if not persisted or persisted.get('status') != 'IN_PROGRESS':
            raise PermissionError('Video session is unavailable for this account.')
        state = VideoSessionState(session_uid, filename, duration_seconds, user_id=user_id)
        self.active_sessions[session_uid] = state
        return state

    def analyze_frame(self, session_uid, frame, video_time=0.0, user_id=None):
        """
        Executes real-time YOLO inference, multi-person tracking, PPE association,
        state change timeline event emission, and representative evidence evaluation.
        """
        state = self.get_or_create_session(session_uid, user_id=user_id)
        h, w = frame.shape[:2]
        state.frames_analyzed += 1

        conf_thresh = float(get_setting('confidence_threshold', 0.35))
        iou_thresh = float(get_setting('iou_threshold', 0.45))

        # 1. YOLO Inference
        detections, _ = self.model_mgr.predict(frame, conf=conf_thresh, iou=iou_thresh)

        # 2. Semantic Separation
        person_dets = [d for d in detections if d['class_name'] == 'Person']
        ppe_dets = [d for d in detections if d['class_name'] in [
            'Hardhat', 'NO-Hardhat', 'Mask', 'NO-Mask', 'Safety Vest', 'NO-Safety Vest'
        ]]
        hazard_dets = [d for d in detections if d['class_name'] in ['machinery', 'vehicle', 'Safety Cone']]

        # 3. Multi-Person Tracking with Video Playback Timestamp
        tracked_persons = state.tracker.update(person_dets, current_time=video_time)

        # 4. Spatial Person-PPE Association
        tracked_persons = state.ppe_engine.associate(tracked_persons, ppe_dets)

        # 5. Context & Environmental Hazards
        state.context_engine.analyze(tracked_persons, hazard_dets, frame_shape=(h, w))

        # 6. Risk Evaluation
        for p in tracked_persons:
            state.risk_engine.evaluate_person_risk(p)

        # 7. Convert to UI person results
        person_results = [p.to_person_result() for p in tracked_persons]

        # 8. State-Based Timeline Event Management (NO ALERT SPAM)
        new_events = []
        time_str = state.format_time_str(video_time)

        for pr in person_results:
            pid = pr['person_id']
            curr_status = pr['status']
            curr_violations = set(pr.get('active_violations', []))

            prev_state = state.person_state_cache.get(pid)

            if prev_state is None:
                state.person_state_cache[pid] = {
                    'status': curr_status,
                    'violations': curr_violations,
                    'last_event_time': video_time
                }
                if curr_status != 'COMPLIANT':
                    v_text = ", ".join(curr_violations) if curr_violations else "Missing PPE Gear"
                    evt = {
                        'timestamp_str': time_str,
                        'video_time': round(video_time, 1),
                        'person_id': pid,
                        'label': pr['label'],
                        'event_type': 'VIOLATION_TRIGGERED',
                        'title': f"{pr['label']}: Violation State ({v_text})",
                        'description': f"{v_text} detected with compliance at {pr['compliance_score']}%.",
                        'severity': pr.get('risk', 'HIGH'),
                        'badge_color': '#DC2626'
                    }
                    new_events.append(evt)
                    state.timeline_events.append(evt)
            else:
                prev_status = prev_state['status']
                prev_violations = prev_state['violations']

                if prev_status != curr_status and (video_time - prev_state['last_event_time']) > 1.5:
                    if curr_status == 'COMPLIANT':
                        evt = {
                            'timestamp_str': time_str,
                            'video_time': round(video_time, 1),
                            'person_id': pid,
                            'label': pr['label'],
                            'event_type': 'COMPLIANCE_RESTORED',
                            'title': f"{pr['label']}: Compliance Restored ✓",
                            'description': "All required PPE gear verified on worker.",
                            'severity': 'LOW',
                            'badge_color': '#10B981'
                        }
                        new_events.append(evt)
                        state.timeline_events.append(evt)
                    else:
                        v_text = ", ".join(curr_violations) if curr_violations else "Missing PPE Gear"
                        evt = {
                            'timestamp_str': time_str,
                            'video_time': round(video_time, 1),
                            'person_id': pid,
                            'label': pr['label'],
                            'event_type': 'VIOLATION_TRIGGERED',
                            'title': f"{pr['label']}: Violation State ({v_text})",
                            'description': f"{v_text} detected with compliance at {pr['compliance_score']}%.",
                            'severity': pr.get('risk', 'HIGH'),
                            'badge_color': '#DC2626'
                        }
                        new_events.append(evt)
                        state.timeline_events.append(evt)

                    prev_state['status'] = curr_status
                    prev_state['violations'] = curr_violations
                    prev_state['last_event_time'] = video_time

        # 9. Evaluate Representative Evidence Candidate (high-severity frame)
        frame_evidence_score = 0.0
        if person_results:
            violations_in_frame = sum(1 for p in person_results if p['status'] != 'COMPLIANT')
            missing_items_count = sum(len(p.get('active_violations', [])) for p in person_results)
            critical_count = sum(1 for p in person_results if p.get('risk') == 'CRITICAL')

            if violations_in_frame > 0:
                frame_evidence_score = 100.0 + (critical_count * 50.0) + (missing_items_count * 20.0) + violations_in_frame
            else:
                frame_evidence_score = 10.0 + len(person_results)

            if frame_evidence_score > state.best_evidence_score:
                state.best_evidence_score = frame_evidence_score
                state.best_evidence_frame = frame.copy()
                state.best_evidence_time = video_time
                state.best_evidence_results = person_results
                state.best_evidence_detections = detections

        # 10. Live Summary Calculation based on currently visible unique workers
        tot_visible = len(person_results)
        comp_visible = sum(1 for p in person_results if p['status'] == 'COMPLIANT')
        viol_visible = tot_visible - comp_visible
        crit_visible = sum(1 for p in person_results if p.get('risk') == 'CRITICAL')
        
        # Live compliance %: compliant workers / total visible workers
        current_comp_rate = round((comp_visible / tot_visible) * 100.0, 1) if tot_visible > 0 else 'N/A'

        return {
            'success': True,
            'session_uid': session_uid,
            'video_time': round(video_time, 2),
            'video_time_str': time_str,
            'fps': self.model_mgr.last_fps,
            'latency_ms': self.model_mgr.last_latency_ms,
            'person_results': person_results,
            'raw_detections': detections,
            'new_events': new_events,
            'all_events': state.timeline_events,
            'live_summary': {
                'people_detected': tot_visible,
                'compliant': comp_visible,
                'violations': viol_visible,
                'critical': crit_visible,
                'overall_compliance': current_comp_rate
            }
        }

    def complete_session(self, session_uid, duration_seconds=0.0, filename="uploaded_video.mp4", client_evidence_b64=None, user_id=None):
        """
        Finalizes video analysis session:
        - Retrieves all confirmed unique person tracks observed across the video
        - Finalizes temporal PPE states per person (aggregates observations, not just last frame)
        - Calculates mathematically traceable final summary
        - Saves representative annotated evidence image
        - Persists atomic session to database
        """
        state = self.active_sessions.get(session_uid)
        upload_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)

        if state is None:
            state = self.get_or_create_session(session_uid, filename, duration_seconds, user_id=user_id)
        elif state.user_id != user_id:
            raise PermissionError('Video session belongs to another account.')

        final_duration = max(duration_seconds, state.duration_seconds)
        if final_duration <= 0 and state.best_evidence_time > 0:
            final_duration = state.best_evidence_time + 1.0

        # Gather finalized person-wise results from all confirmed tracks across the video session
        confirmed_tracks = state.tracker.get_all_confirmed_tracks()
        
        final_persons = []
        # Sort tracks by track_id for deterministic presentation
        for tid in sorted(confirmed_tracks.keys()):
            track = confirmed_tracks[tid]
            track.finalize_session_state()
            state.risk_engine.evaluate_person_risk(track)
            res = track.to_person_result()
            final_persons.append(res)
            state.cumulative_persons[tid] = res

        if not final_persons and state.cumulative_persons:
            final_persons = list(state.cumulative_persons.values())

        total_observed = len(final_persons)

        if total_observed == 0:
            compliant_count = 0
            violation_count = 0
            critical_count = 0
            hh_violations = 0
            mk_violations = 0
            vt_violations = 0
            overall_compliance = 'N/A'
            safety_status = 'NO PERSONNEL OBSERVED'
            safety_badge_color = '#64748B'
        else:
            ppe_totals = summarize_ppe(final_persons)
            compliant_count = ppe_totals['compliant']
            non_compliant_people = ppe_totals['non_compliant_people']
            violation_count = ppe_totals['ppe_violations']
            critical_count = sum(1 for p in final_persons if p.get('risk') == 'CRITICAL')

            # Class-specific unique person violation counts (at most 1 per person)
            hh_violations = sum(1 for p in final_persons if p.get('hardhat') == 'MISSING')
            mk_violations = sum(1 for p in final_persons if p.get('mask') == 'MISSING')
            vt_violations = sum(1 for p in final_persons if p.get('vest') == 'MISSING')

            # Overall compliance %: compliant workers / total observed workers
            overall_compliance = round((compliant_count / total_observed) * 100.0, 1)

            if violation_count == 0:
                safety_status = 'SAFE'
                safety_badge_color = '#10B981' # Green
            elif critical_count > 0 or hh_violations > 0 or overall_compliance < 60.0:
                safety_status = 'UNSAFE'
                safety_badge_color = '#EF4444' # Red
            else:
                safety_status = 'CAUTION'
                safety_badge_color = '#F59E0B' # Amber

        # Save Representative Evidence Image
        evidence_filename = f"evidence_{session_uid}.jpg"
        evidence_rel_path = f"static/uploads/{evidence_filename}"
        evidence_abs_path = os.path.join(upload_folder, evidence_filename)

        if state.best_evidence_frame is not None:
            annotated_frame = draw_person_annotations(
                state.best_evidence_frame,
                state.best_evidence_results,
                state.best_evidence_detections
            )
            cv2.imwrite(evidence_abs_path, annotated_frame)
        elif client_evidence_b64:
            try:
                b64_clean = client_evidence_b64.split(',', 1)[1] if ',' in client_evidence_b64 else client_evidence_b64
                img_bytes = base64.b64decode(b64_clean)
                with open(evidence_abs_path, 'wb') as f:
                    f.write(img_bytes)
            except Exception as e:
                print(f"[VIDEO_SERVICE] Client evidence write failed: {e}")
                self._create_fallback_evidence_image(evidence_abs_path, session_uid, filename)
        else:
            self._create_fallback_evidence_image(evidence_abs_path, session_uid, filename)

        # Contextual Recommendations based strictly on observed unique person violations
        recommendations = self._generate_video_recommendations(hh_violations, mk_violations, vt_violations, critical_count)

        # Persist to Database via SessionManager
        notes_str = f"Live video analysis ({state.frames_analyzed} frames analyzed, duration {final_duration:.1f}s)"
        session_res = self.session_mgr.complete_session(
            session_uid=session_uid,
            person_results=final_persons,
            evidence_image=evidence_rel_path,
            duration_seconds=final_duration,
            source_name=filename,
            notes=notes_str,
            user_id=user_id
        )

        mins = int(final_duration // 60)
        secs = int(final_duration % 60)
        duration_formatted = f"{mins:02d}:{secs:02d}"

        result_payload = {
            'success': True,
            'session_uid': session_uid,
            'video_filename': filename,
            'duration_seconds': round(final_duration, 1),
            'duration_formatted': duration_formatted,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'people_observed': total_observed,
            'compliant_count': compliant_count,
            'non_compliant_people': non_compliant_people if total_observed else 0,
            'violation_count': violation_count,
            'ppe_violations': violation_count,
            'critical_count': critical_count,
            'overall_compliance': overall_compliance,
            'safety_status': safety_status,
            'safety_badge_color': safety_badge_color,
            'safety_score': session_res.get('safety_score', overall_compliance if isinstance(overall_compliance, (int, float)) else 100.0),
            'ppe_breakdown': {
                'hardhat_violations': hh_violations,
                'mask_violations': mk_violations,
                'vest_violations': vt_violations
            },
            'person_results': final_persons,
            'timeline_events': state.timeline_events,
            'evidence_image_url': f"/{evidence_rel_path}",
            'recommendations': recommendations,
            'report_download_url': f"/reports/session/{session_uid}/download",
            'session_detail_url': f"/session/{session_uid}"
        }

        print(f"[VIDEO_SERVICE] Finalized Session {session_uid}: {total_observed} People, {compliant_count} Compliant, {violation_count} Violations ({hh_violations} HH, {mk_violations} Mask, {vt_violations} Vest), Compliance: {overall_compliance}%, Status: {safety_status}")
        return result_payload

    def _create_fallback_evidence_image(self, file_path, session_uid, filename):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        blank[:] = (20, 24, 33)
        cv2.putText(blank, f"SafetyVision AI - Video Analysis", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(blank, f"Session: {session_uid}", (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (56, 189, 248), 2)
        cv2.putText(blank, f"File: {filename}", (40, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)
        cv2.imwrite(file_path, blank)

    def _generate_video_recommendations(self, hh_viol, mk_viol, vt_viol, crit_count):
        recs = []
        if hh_viol > 0:
            recs.append({
                'priority': 'HIGH',
                'title': 'Mandatory Head Protection Enforcement',
                'action': f"Detected {hh_viol} worker(s) without Hardhats. Restrict zone access until certified Type I/II helmets are equipped (OSHA 1926.100).",
                'color': '#DC2626'
            })
        if vt_viol > 0:
            recs.append({
                'priority': 'HIGH' if crit_count > 0 else 'MEDIUM',
                'title': 'High-Visibility Safety Vest Distribution',
                'action': f"Detected {vt_viol} worker(s) without reflective safety vests. Issue ANSI/ISEA 107 Class 2 reflective vests immediately for vehicle and machinery visibility.",
                'color': '#F97316'
            })
        if mk_viol > 0:
            recs.append({
                'priority': 'MEDIUM',
                'title': 'Respiratory Face Covering Compliance',
                'action': f"Detected {mk_viol} worker(s) without protective face masks. Ensure appropriate respiratory equipment is worn in dust, chemical, or spray areas.",
                'color': '#F59E0B'
            })
        if hh_viol == 0 and vt_viol == 0 and mk_viol == 0:
            recs.append({
                'priority': 'LOW',
                'title': 'Standard Operating Protocol Sustained',
                'action': "100% PPE compliance achieved across all observed personnel throughout video playback. Maintain active automated monitoring.",
                'color': '#10B981'
            })
        return recs
