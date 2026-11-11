import math

def box_distance(boxA, boxB):
    """
    Computes minimum image-space edge-to-edge distance between two bounding boxes.
    Returns 0.0 if boxes overlap.
    """
    # Box A: [x1, y1, x2, y2]
    # Box B: [x1, y1, x2, y2]
    left = boxB[0] > boxA[2]
    right = boxA[0] > boxB[2]
    bottom = boxB[1] > boxA[3]
    top = boxA[1] > boxB[3]

    if top and left:
        return math.hypot(boxA[0] - boxB[2], boxA[1] - boxB[3])
    elif left and bottom:
        return math.hypot(boxA[0] - boxB[2], boxA[3] - boxB[1])
    elif bottom and right:
        return math.hypot(boxA[2] - boxB[0], boxA[3] - boxB[1])
    elif right and top:
        return math.hypot(boxA[2] - boxB[0], boxA[1] - boxB[3])
    elif left:
        return boxB[0] - boxA[2]
    elif right:
        return boxA[0] - boxB[2]
    elif bottom:
        return boxB[1] - boxA[3]
    elif top:
        return boxA[1] - boxB[3]
    else:
        return 0.0 # Boxes overlap

class ContextIntelligenceEngine:
    def __init__(self, proximity_threshold_px=120.0, required_cones=4):
        self.proximity_threshold_px = proximity_threshold_px
        self.required_cones = required_cones

    def analyze(self, tracked_persons, hazard_detections, frame_shape=(720, 1280)):
        """
        hazard_detections: list of dicts with 'class_name', 'bbox', 'confidence'
        Returns dict containing context flags, proximity warnings, barrier status.
        """
        machinery = [d for d in hazard_detections if d['class_name'] == 'machinery']
        vehicles = [d for d in hazard_detections if d['class_name'] == 'vehicle']
        cones = [d for d in hazard_detections if d['class_name'] == 'Safety Cone']

        context_events = []

        # Analyze Machinery Proximity
        for person in tracked_persons:
            person.near_machinery = False
            person.near_vehicle = False

            for m in machinery:
                dist = box_distance(person.bbox, m['bbox'])
                if dist <= self.proximity_threshold_px:
                    person.near_machinery = True
                    event_msg = f"Person #{person.track_id} in close proximity ({round(dist, 1)}px est.) to machinery"
                    if person.active_violations:
                        v_name = 'Machinery Hazard Proximity'
                        if v_name not in person.active_violations:
                            person.active_violations.append(v_name)
                    context_events.append({
                        'type': 'MACHINERY_PROXIMITY',
                        'person_id': person.track_id,
                        'estimated_distance_px': round(dist, 1),
                        'message': event_msg
                    })
                    break

            for v in vehicles:
                dist = box_distance(person.bbox, v['bbox'])
                if dist <= self.proximity_threshold_px:
                    person.near_vehicle = True
                    event_msg = f"Person #{person.track_id} near vehicle movement zone ({round(dist, 1)}px est.)"
                    v_name = 'Vehicle Proximity Warning'
                    if v_name not in person.active_violations:
                        person.active_violations.append(v_name)
                    context_events.append({
                        'type': 'VEHICLE_PROXIMITY',
                        'person_id': person.track_id,
                        'estimated_distance_px': round(dist, 1),
                        'message': event_msg
                    })
                    break

        # Analyze Safety Cone Barrier Integrity
        cone_count = len(cones)
        barrier_status = {
            'required_cones': self.required_cones,
            'detected_cones': cone_count,
            'status': 'INTACT' if cone_count >= self.required_cones else 'INCOMPLETE',
            'warning': None
        }

        if cone_count < self.required_cones and (machinery or vehicles):
            warning_msg = f"Incomplete Safety Barrier: Detected {cone_count}/{self.required_cones} required cones in active equipment zone."
            barrier_status['warning'] = warning_msg
            context_events.append({
                'type': 'INCOMPLETE_SAFETY_BARRIER',
                'detected_cones': cone_count,
                'required_cones': self.required_cones,
                'message': warning_msg
            })

        return {
            'machinery_count': len(machinery),
            'vehicle_count': len(vehicles),
            'cone_count': cone_count,
            'barrier_status': barrier_status,
            'context_events': context_events,
            'disclaimer': 'Proximity values represent 2D image-space bounding box pixel estimates, not calibrated real-world Euclidean 3D metric distances.'
        }
