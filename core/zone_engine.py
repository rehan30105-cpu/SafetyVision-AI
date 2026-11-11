import json

def point_in_polygon(x, y, polygon):
    """
    Ray-casting algorithm to determine if point (x, y) is inside polygon.
    polygon: list of [x, y] vertices.
    """
    num_vertices = len(polygon)
    if num_vertices < 3:
        return False

    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, num_vertices + 1):
        p2x, p2y = polygon[i % num_vertices]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

class SafetyZoneEngine:
    def __init__(self):
        # person_id -> set of active zone_ids
        self.person_zones = {}

    def evaluate_zones(self, tracked_persons, zones_list, frame_width=1.0, frame_height=1.0):
        """
        tracked_persons: list of TrackedPerson objects
        zones_list: list of zone dicts from DB with 'id', 'name', 'zone_type', 'polygon_coords', 'required_ppe', 'risk_multiplier'
        Returns: list of zone transition events (ENTERED, EXITED)
        """
        events = []

        # Parse polygons
        active_zones = []
        for z in zones_list:
            if not z.get('is_active', 1):
                continue
            coords = z['polygon_coords']
            if isinstance(coords, str):
                try:
                    coords = json.loads(coords)
                except Exception:
                    continue
            
            # Ensure coordinates are in normalized 0..1 scale
            # If coordinates are pixel values (> 1.0), normalize with frame width/height
            norm_coords = []
            for pt in coords:
                px = pt[0] / frame_width if pt[0] > 1.0 and frame_width > 1 else pt[0]
                py = pt[1] / frame_height if pt[1] > 1.0 and frame_height > 1 else pt[1]
                norm_coords.append([px, py])

            req_ppe = z['required_ppe']
            if isinstance(req_ppe, str):
                try:
                    req_ppe = json.loads(req_ppe)
                except Exception:
                    req_ppe = []

            active_zones.append({
                'id': z['id'],
                'name': z['name'],
                'zone_type': z['zone_type'],
                'coords': norm_coords,
                'required_ppe': req_ppe,
                'risk_multiplier': float(z.get('risk_multiplier', 1.0))
            })

        for person in tracked_persons:
            pid = person.track_id
            # Normalized bottom-center of person (ground contact point / foot position)
            # which is most physically accurate for zone occupancy
            px = person.centroid[0] / max(frame_width, 1.0) if frame_width > 1.0 else person.centroid[0]
            py = person.bbox[3] / max(frame_height, 1.0) if frame_height > 1.0 else person.bbox[3]

            current_person_zones = set()
            person_zone_details = []

            for zone in active_zones:
                if point_in_polygon(px, py, zone['coords']):
                    current_person_zones.add(zone['id'])
                    person_zone_details.append(zone)

            prev_zones = self.person_zones.get(pid, set())

            # Check ENTERED transitions
            entered_zones = current_person_zones - prev_zones
            for zid in entered_zones:
                z_obj = next((z for z in active_zones if z['id'] == zid), None)
                if z_obj:
                    events.append({
                        'type': 'ENTERED',
                        'person_id': pid,
                        'zone_id': zid,
                        'zone_name': z_obj['name'],
                        'zone_type': z_obj['zone_type']
                    })

            # Check EXITED transitions
            exited_zones = prev_zones - current_person_zones
            for zid in exited_zones:
                z_obj = next((z for z in active_zones if z['id'] == zid), None)
                if z_obj:
                    events.append({
                        'type': 'EXITED',
                        'person_id': pid,
                        'zone_id': zid,
                        'zone_name': z_obj['name'],
                        'zone_type': z_obj['zone_type']
                    })

            # Update person state
            self.person_zones[pid] = current_person_zones

            # Apply zone enforcement
            if person_zone_details:
                primary_zone = person_zone_details[0]
                person.zone_id = primary_zone['id']
                person.zone_name = primary_zone['name']

                # Restricted Zone breach
                if primary_zone['zone_type'] == 'RESTRICTED':
                    person.active_violations.append('Restricted Zone Breach')

                # Check Zone PPE requirements
                for req in primary_zone['required_ppe']:
                    if str(person.ppe_status.get(req, '')).upper() == 'MISSING':
                        v_name = f"Zone Missing {req}"
                        if v_name not in person.active_violations:
                            person.active_violations.append(v_name)
            else:
                person.zone_id = None
                person.zone_name = None

        return events

    def cleanup(self, active_track_ids):
        # Remove tracks that no longer exist
        self.person_zones = {pid: zones for pid, zones in self.person_zones.items() if pid in active_track_ids}
