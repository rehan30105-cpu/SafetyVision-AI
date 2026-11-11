import time

class PersistenceEngine:
    def __init__(self, persistence_threshold_sec=1.5, escalation_threshold_sec=5.0, alert_cooldown_sec=10.0):
        self.persistence_threshold_sec = persistence_threshold_sec
        self.escalation_threshold_sec = escalation_threshold_sec
        self.alert_cooldown_sec = alert_cooldown_sec
        
        # State: (person_id, violation_type) -> { 'first_seen': t, 'last_seen': t, 'last_alerted': t, 'incident_logged': bool }
        self.violation_records = {}

    def process(self, tracked_persons, current_time=None):
        """
        Filters momentary flicker and confirms real persistent violations.
        Returns: list of confirmed actionable incidents to persist and alert.
        """
        now = current_time or time.time()
        confirmed_events = []

        active_keys = set()

        for person in tracked_persons:
            pid = person.track_id
            person.confirmed_violations = []

            for v_type in person.active_violations:
                key = (pid, v_type)
                active_keys.add(key)

                if key not in self.violation_records:
                    self.violation_records[key] = {
                        'first_seen': now,
                        'last_seen': now,
                        'last_alerted': 0.0,
                        'incident_logged': False
                    }
                else:
                    self.violation_records[key]['last_seen'] = now

                record = self.violation_records[key]
                duration = max(0.0, round(now - record['first_seen'], 1))
                person.violation_durations[v_type] = duration

                # Temporal confirmation check
                if duration >= self.persistence_threshold_sec:
                    person.confirmed_violations.append(v_type)

                    # Check alert cooldown / deduplication
                    time_since_alert = now - record['last_alerted']
                    if not record['incident_logged'] or time_since_alert >= self.alert_cooldown_sec:
                        record['last_alerted'] = now
                        record['incident_logged'] = True
                        
                        confirmed_events.append({
                            'person_id': pid,
                            'violation_type': v_type,
                            'duration_seconds': duration,
                            'risk_score': person.risk_score,
                            'risk_level': person.risk_level,
                            'zone_id': person.zone_id,
                            'zone_name': person.zone_name,
                            'is_escalated': (duration >= self.escalation_threshold_sec),
                            'timestamp': now
                        })

        # Cleanup resolved violations (violations not seen in current frame)
        to_delete = []
        for key, record in self.violation_records.items():
            if key not in active_keys:
                # If disappeared for more than 1 second, clear
                if now - record['last_seen'] > 1.0:
                    to_delete.append(key)

        for key in to_delete:
            del self.violation_records[key]

        return confirmed_events
