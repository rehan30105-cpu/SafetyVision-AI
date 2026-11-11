import time
import math

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea

class TrackedPerson:
    def __init__(self, track_id, bbox, timestamp=None):
        self.track_id = track_id
        self.bbox = bbox # [x1, y1, x2, y2]
        self.centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        self.first_seen = timestamp or time.time()
        self.last_seen = self.first_seen
        self.lost_frames = 0
        self.hits = 1
        self.is_confirmed = False
        self.history = [self.centroid]
        
        # 3 Mandatory PPE Categories with Temporal Aggregation
        self.categories = ['Hardhat', 'Mask', 'Safety Vest']
        
        # Current instant state
        self.ppe_status = {
            'Hardhat': 'UNCERTAIN',     # 'PRESENT', 'MISSING', 'UNCERTAIN'
            'Mask': 'UNCERTAIN',
            'Safety Vest': 'UNCERTAIN'
        }
        self.associated_ppe_items = []
        self.ppe_details = {}
        
        # Sliding observation window for robust temporal consistency (prevents frame flicker)
        # category -> list of ('PRESENT'/'MISSING'/'UNCERTAIN', confidence)
        self.observation_window = {cat: [] for cat in self.categories}
        
        # Cumulative session counts across all analyzed frames throughout video
        self.cumulative_counts = {
            cat: {'PRESENT': 0, 'MISSING': 0, 'UNCERTAIN': 0}
            for cat in self.categories
        }
        self.cumulative_confidences = {cat: [] for cat in self.categories}

        self.compliance_score = 100.0
        self.risk_score = 0.0
        self.risk_level = 'LOW'
        self.status = 'COMPLIANT'
        self.previous_status = 'NONE'
        self.risk_reason = 'Fully Compliant'
        self.active_violations = []

        # Lifecycle & State-Based Incident Tracking
        self.active_incident_db_id = None
        self.active_incident_uid = None
        self.status_change_event = None
        self.violation_start_time = None
        self.zone_id = None
        self.zone_name = None
        self.near_machinery = False
        self.near_vehicle = False

        self.ppe_grace = {'Hardhat': 0, 'Mask': 0, 'Safety Vest': 0}

    @property
    def duration_seconds(self):
        return max(0.0, round(self.last_seen - self.first_seen, 1))

    def update(self, bbox, timestamp=None):
        now = timestamp or time.time()
        self.bbox = bbox
        self.centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        self.last_seen = now
        self.lost_frames = 0
        self.hits += 1
        if self.hits >= 2:
            self.is_confirmed = True
        self.history.append(self.centroid)
        if len(self.history) > 30:
            self.history.pop(0)

    def record_ppe_observation(self, category, state, details=None):
        """
        Records a frame-level PPE observation into temporal window and cumulative stats.
        state must be 'PRESENT', 'MISSING', or 'UNCERTAIN'.
        """
        if category not in self.categories:
            return

        conf = details.get('confidence', 0.0) if details else 0.0
        
        # Record into sliding window (last 8 frames)
        self.observation_window[category].append((state, conf))
        if len(self.observation_window[category]) > 8:
            self.observation_window[category].pop(0)

        # Record into cumulative session totals
        self.cumulative_counts[category][state] += 1
        if conf > 0:
            self.cumulative_confidences[category].append(conf)

        # Determine temporally smoothed current state
        recent_states = [s for s, _ in self.observation_window[category]]
        pres_count = recent_states.count('PRESENT')
        miss_count = recent_states.count('MISSING')

        # A genuinely high-confidence positive detection is strong evidence that
        # PPE is present. Allow a single such detection to immediately restore a
        # PRESENT state even if the recent window was dominated by MISSING frames
        # (e.g. the worker just put on a hardhat). This prevents the smoothed
        # window from masking real compliance and avoids stale violations.
        # NOTE: we require strong confidence AND that this frame is a true positive
        # detection (not a grace-held MISSING). conf here is on a 0-100 scale.
        single_frame_present = (state == 'PRESENT' and conf >= 60.0)

        if pres_count >= 2 or single_frame_present:
            self.ppe_status[category] = 'PRESENT'
            self.ppe_grace[category] = 0
        elif miss_count >= 2:
            self.ppe_status[category] = 'MISSING'
            self.ppe_grace[category] = 0
        elif self.ppe_status[category] == 'PRESENT' and self.ppe_grace[category] < 3:
            # Temporary occlusion grace: hold PRESENT for up to 3 frames
            self.ppe_grace[category] += 1
        else:
            self.ppe_status[category] = state

        if details:
            self.ppe_details[category] = details

    def finalize_session_state(self):
        """
        Aggregates all observations across the video session to determine
        the finalized, ground-truth person result.
        Eliminates single-frame blips and transient occlusions.
        """
        final_status = {}
        for cat in self.categories:
            counts = self.cumulative_counts[cat]
            pres = counts['PRESENT']
            miss = counts['MISSING']
            uncert = counts['UNCERTAIN']

            # If person was confirmed wearing gear for a significant portion of observations:
            if pres > 0 and pres >= miss:
                final_status[cat] = 'PRESENT'
            elif miss > pres:
                final_status[cat] = 'MISSING'
            elif pres > 0:
                final_status[cat] = 'PRESENT'
            else:
                final_status[cat] = 'MISSING' # Unobserved/missing gear

        self.ppe_status = final_status
        
        # Recalculate finalized compliance
        present_count = sum(1 for cat in self.categories if self.ppe_status[cat] == 'PRESENT')
        self.compliance_score = round((present_count / 3.0) * 100.0, 1)

        self.active_violations = [f"Missing {cat}" for cat in self.categories if self.ppe_status[cat] == 'MISSING']

        if present_count == 3:
            self.status = 'COMPLIANT'
            self.risk_level = 'LOW'
            self.risk_score = 0.0
            self.risk_reason = 'Fully Compliant'
        else:
            self.status = 'PPE VIOLATION'
            if len(self.active_violations) >= 2 or 'Missing Hardhat' in self.active_violations:
                self.risk_level = 'CRITICAL' if len(self.active_violations) >= 2 else 'HIGH'
            else:
                self.risk_level = 'MEDIUM'

    def to_person_result(self):
        """
        Generates clean, human-readable person-wise result object for UI display.
        """
        missing_list = [v.replace('Missing ', '') for v in self.active_violations]
        hh_det = self.ppe_details.get('Hardhat', {})
        mk_det = self.ppe_details.get('Mask', {})
        vt_det = self.ppe_details.get('Safety Vest', {})

        is_hh_present = (self.ppe_status.get('Hardhat') == 'PRESENT')
        is_mk_present = (self.ppe_status.get('Mask') == 'PRESENT')
        is_vt_present = (self.ppe_status.get('Safety Vest') == 'PRESENT')

        all_compliant = (is_hh_present and is_mk_present and is_vt_present)
        person_status = 'COMPLIANT' if all_compliant else 'PPE VIOLATION'
        present_count = int(is_hh_present) + int(is_mk_present) + int(is_vt_present)
        comp_score = round((present_count / 3.0) * 100.0, 1)

        return {
            'person_id': self.track_id,
            'label': f"Person #{self.track_id}",
            'bbox': [round(v, 1) for v in self.bbox],
            'centroid': [round(v, 1) for v in self.centroid],
            'hardhat': 'PRESENT' if is_hh_present else 'MISSING',
            'hardhat_label': hh_det.get('label', '✓ DETECTED' if is_hh_present else '✗ MISSING'),
            'hardhat_class': hh_det.get('detected_class', 'Hardhat' if is_hh_present else 'NO-Hardhat'),
            'hardhat_conf': round(hh_det.get('confidence', 0.0) * 100.0, 1) if 0 < hh_det.get('confidence', 0.0) <= 1.0 else round(hh_det.get('confidence', 0.0), 1),

            'mask': 'PRESENT' if is_mk_present else 'MISSING',
            'mask_label': mk_det.get('label', '✓ DETECTED' if is_mk_present else '✗ MISSING'),
            'mask_class': mk_det.get('detected_class', 'Mask' if is_mk_present else 'NO-Mask'),
            'mask_conf': round(mk_det.get('confidence', 0.0) * 100.0, 1) if 0 < mk_det.get('confidence', 0.0) <= 1.0 else round(mk_det.get('confidence', 0.0), 1),

            'vest': 'PRESENT' if is_vt_present else 'MISSING',
            'vest_label': vt_det.get('label', '✓ DETECTED' if is_vt_present else '✗ MISSING'),
            'vest_class': vt_det.get('detected_class', 'Safety Vest' if is_vt_present else 'NO-Safety Vest'),
            'vest_conf': round(vt_det.get('confidence', 0.0) * 100.0, 1) if 0 < vt_det.get('confidence', 0.0) <= 1.0 else round(vt_det.get('confidence', 0.0), 1),

            'compliance': comp_score,
            'compliance_score': comp_score,
            'status': person_status,
            'risk': self.risk_level,
            'risk_score': self.risk_score,
            'risk_reason': self.risk_reason,
            'missing_items': missing_list,
            'active_violations': list(self.active_violations),
            'associated_ppe': list(self.associated_ppe_items),
            'track_id': self.track_id,
            'hits': self.hits,
            'is_confirmed': self.is_confirmed,
            'duration_seconds': self.duration_seconds,
            'near_machinery': self.near_machinery,
            'near_vehicle': self.near_vehicle
        }

class PersonTracker:
    def __init__(self, max_lost_frames=35, min_iou=0.10, max_distance=650.0):
        self.max_lost_frames = max_lost_frames
        self.min_iou = min_iou
        self.max_distance = max_distance
        self.next_track_id = 1
        self.tracks = {} # track_id -> TrackedPerson
        self.retired_tracks = {} # track_id -> TrackedPerson (for re-identification / session recall)

    def update(self, person_detections, current_time=None):
        now = current_time or time.time()

        # Mark all existing tracks as lost for this cycle
        for track in self.tracks.values():
            track.lost_frames += 1

        if not person_detections:
            # Move expired tracks to retired
            active = {}
            for tid, t in self.tracks.items():
                if t.lost_frames <= self.max_lost_frames:
                    active[tid] = t
                else:
                    self.retired_tracks[tid] = t
            self.tracks = active
            return [t for t in self.tracks.values() if t.lost_frames == 0 and t.hits >= 2]

        unmatched_detections = list(range(len(person_detections)))

        # Step 1: Match active tracks using scale-aware global scoring (IoU + Centroid proximity)
        candidate_matches = []
        for tid, track in self.tracks.items():
            track_w = max(1.0, track.bbox[2] - track.bbox[0])
            track_h = max(1.0, track.bbox[3] - track.bbox[1])
            # Scale distance allowance dynamically based on bounding box size (handles 4K, 1080p, 720p)
            max_allowed_dist = max(self.max_distance, max(track_w, track_h) * 2.0)

            for det_idx in unmatched_detections:
                det_bbox = person_detections[det_idx]['bbox']
                iou = calculate_iou(track.bbox, det_bbox)
                
                det_cx = (det_bbox[0] + det_bbox[2]) / 2.0
                det_cy = (det_bbox[1] + det_bbox[3]) / 2.0
                dist = math.hypot(det_cx - track.centroid[0], det_cy - track.centroid[1])

                if iou >= self.min_iou:
                    score = 10.0 + (iou * 5.0) # Strong geometric overlap
                    candidate_matches.append((score, tid, det_idx))
                elif dist <= max_allowed_dist:
                    score = 2.0 * (1.0 - (dist / max_allowed_dist))
                    candidate_matches.append((score, tid, det_idx))

        # Sort candidate matches in descending order of score
        candidate_matches.sort(key=lambda x: x[0], reverse=True)

        matched_tracks = set()
        matched_detections = set()

        for score, tid, det_idx in candidate_matches:
            if tid in matched_tracks or det_idx in matched_detections:
                continue
            matched_bbox = person_detections[det_idx]['bbox']
            self.tracks[tid].update(matched_bbox, now)
            matched_tracks.add(tid)
            matched_detections.add(det_idx)
            if det_idx in unmatched_detections:
                unmatched_detections.remove(det_idx)

        # Step 2: Try re-identifying unmatched detections with recently retired tracks (prevents ID fragmentation)
        if unmatched_detections and self.retired_tracks:
            retired_matches = []
            for tid, track in self.retired_tracks.items():
                track_w = max(1.0, track.bbox[2] - track.bbox[0])
                track_h = max(1.0, track.bbox[3] - track.bbox[1])
                max_reid_dist = max(self.max_distance, max(track_w, track_h) * 2.2)

                for det_idx in unmatched_detections:
                    det_bbox = person_detections[det_idx]['bbox']
                    iou = calculate_iou(track.bbox, det_bbox)
                    det_cx = (det_bbox[0] + det_bbox[2]) / 2.0
                    det_cy = (det_bbox[1] + det_bbox[3]) / 2.0
                    dist = math.hypot(det_cx - track.centroid[0], det_cy - track.centroid[1])

                    if iou >= self.min_iou or dist <= max_reid_dist:
                        score = (iou * 5.0) + (1.0 - min(1.0, dist / max_reid_dist))
                        retired_matches.append((score, tid, det_idx))

            retired_matches.sort(key=lambda x: x[0], reverse=True)
            for score, tid, det_idx in retired_matches:
                if det_idx in matched_detections:
                    continue
                # Restore track from retired
                restored_track = self.retired_tracks.pop(tid)
                restored_track.update(person_detections[det_idx]['bbox'], now)
                self.tracks[tid] = restored_track
                matched_detections.add(det_idx)
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)

        # Step 3: Initialize new tracks only for legitimately unmatched person detections
        for det_idx in unmatched_detections:
            det_bbox = person_detections[det_idx]['bbox']
            new_track = TrackedPerson(self.next_track_id, det_bbox, now)
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1

        # Step 4: Archive tracks that exceeded max_lost_frames to retired pool
        active = {}
        for tid, t in self.tracks.items():
            if t.lost_frames <= self.max_lost_frames:
                active[tid] = t
            else:
                self.retired_tracks[tid] = t
        self.tracks = active

        # Return confirmed tracks actively seen in this frame
        return [t for t in self.tracks.values() if t.lost_frames == 0]

    def get_all_confirmed_tracks(self):
        """
        Returns every track observed at least once across the video session
        (active + retired). A real person seen in only one frame is still an
        observed person, so single-frame observations are intentionally
        included to keep session people counts consistent with visible detections.
        """
        all_tracks = {**self.retired_tracks, **self.tracks}
        confirmed = {}
        for tid, t in all_tracks.items():
            # A track is included in the session summary if it was ever
            # observed by the tracker. update() only creates a track when a
            # person detection exists, so hits >= 1 is a genuine observation.
            # Do NOT drop single-frame observations: a real person detected in
            # only one frame is still an observed person and must be counted.
            if t.hits >= 1 or t.is_confirmed:
                confirmed[tid] = t
        return confirmed

    def reset(self):
        self.tracks.clear()
        self.retired_tracks.clear()
        self.next_track_id = 1
