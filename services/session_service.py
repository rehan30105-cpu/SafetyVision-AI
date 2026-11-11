"""
SAFETYVISION AI - Detection Session Lifecycle Management
Manages atomic detection sessions (IMAGE, VIDEO, LIVE), ensures session separation,
and records consolidated person-wise results and evidence images.
"""

import os
import json
import uuid
import cv2
from datetime import datetime
from database.db import get_db
from core.ppe_metrics import count_person_ppe_violations, summarize_ppe

class SessionManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SessionManager()
        return cls._instance

    def generate_session_uid(self, session_type):
        """
        Generates clean formatted session UID:
        IMG-YYYYMMDD-XXX, VID-YYYYMMDD-XXX, LIVE-YYYYMMDD-XXX
        """
        prefix = 'IMG' if session_type == 'IMAGE' else ('VID' if session_type == 'VIDEO' else 'LIVE')
        date_str = datetime.now().strftime('%Y%m%d')
        conn = get_db()
        count = conn.execute(
            "SELECT COUNT(*) FROM detection_sessions WHERE session_type = ? AND DATE(created_at) = DATE('now')",
            (session_type,)
        ).fetchone()[0]
        conn.close()
        seq = str(count + 1).zfill(3)
        return f"{prefix}-{date_str}-{seq}"

    def create_session(self, session_type, source_name="", user_id=None):
        session_uid = self.generate_session_uid(session_type)
        conn = get_db()
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO detection_sessions (
                session_uid, session_type, source_name, start_time, status, user_id
            ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, 'IN_PROGRESS', ?)
        ''', (session_uid, session_type, source_name, user_id))
        conn.commit()
        db_id = cur.lastrowid
        conn.close()
        return session_uid, db_id

    def complete_session(self, session_uid, person_results, evidence_image, duration_seconds=0.0, source_name="", notes="", user_id=None):
        """
        Finalizes detection session with consolidated person-wise results.
        Canonical counting (single rule everywhere):
          PPE violations = individually missing required PPE items (0-3/person)
          non-compliant  = people missing >= 1 required item
        Enforces: ONE incident per violating person (zero incidents if all compliant).
        Incident grouping/severity rules unchanged.
        """
        ppe_totals = summarize_ppe(person_results or [])
        total_p = ppe_totals['people']
        comp_p = ppe_totals['compliant']
        viol_p = ppe_totals['ppe_violations']
        non_comp_p = ppe_totals['non_compliant_people']
        crit_p = sum(1 for p in (person_results or []) if p.get('risk') == 'CRITICAL')
        
        avg_comp = round(sum(p.get('compliance', 0.0) for p in person_results) / max(total_p, 1), 1) if total_p > 0 else 100.0
        avg_risk = round(sum(p.get('risk_score', 0.0) for p in person_results) / max(total_p, 1), 1) if total_p > 0 else 0.0
        safety_score = round(max(0.0, 100.0 - (0.4 * avg_risk + 0.3 * (100.0 - avg_comp) + 0.3 * min(100.0, crit_p * 15.0))), 1)

        # Standardize person results format to guarantee no undefined fields
        clean_results = []
        for idx, p in enumerate(person_results, 1):
            pid = p.get('person_id', p.get('track_id', idx))
            missing = p.get('missing_items', [])
            is_comp = (p.get('compliance', 0.0) >= 99.0)
            
            clean_results.append({
                'person_id': pid,
                'label': p.get('label', f"Person #{pid}"),
                'bbox': p.get('bbox', [0, 0, 0, 0]),
                'hardhat': p.get('hardhat', 'MISSING'),
                'mask': p.get('mask', 'MISSING'),
                'vest': p.get('vest', 'MISSING'),
                'compliance': round(float(p.get('compliance', 0.0)), 1),
                'status': 'COMPLIANT' if is_comp else 'PPE VIOLATION',
                'risk': p.get('risk', 'LOW' if is_comp else 'MEDIUM'),
                'risk_score': round(float(p.get('risk_score', 0.0)), 1),
                'risk_reason': p.get('risk_reason', 'Fully Compliant' if is_comp else f"Missing {', '.join(missing)}"),
                'missing_items': missing,
                'active_violations': p.get('active_violations', [f"Missing {m}" for m in missing]),
                'duration_seconds': round(float(p.get('duration_seconds', duration_seconds)), 1)
            })

        conn = get_db()
        cur = conn.cursor()

        owner = cur.execute('SELECT user_id FROM detection_sessions WHERE session_uid = ?', (session_uid,)).fetchone()
        if not owner:
            conn.close()
            raise ValueError('Detection session does not exist.')
        if owner['user_id'] is not None and owner['user_id'] != user_id:
            conn.close()
            raise PermissionError('Detection session belongs to another account.')
        owner_id = owner['user_id'] if owner['user_id'] is not None else user_id

        # Update detection_sessions row
        cur.execute('''
            UPDATE detection_sessions SET
                end_time = CURRENT_TIMESTAMP,
                duration_seconds = ?,
                people_count = ?,
                compliant_count = ?,
                violation_count = ?,
                critical_count = ?,
                overall_compliance = ?,
                safety_score = ?,
                evidence_image = ?,
                status = 'COMPLETED',
                person_results_json = ?,
                notes = ?,
                user_id = COALESCE(user_id, ?)
            WHERE session_uid = ?
        ''', (
            round(duration_seconds, 1),
            total_p,
            comp_p,
            viol_p,
            crit_p,
            avg_comp,
            safety_score,
            evidence_image,
            json.dumps(clean_results),
            notes,
            owner_id,
            session_uid
        ))

        # Log meaningful incidents: ONE per violating person for this session
        incident_ids_by_person = {}
        for p in clean_results:
            if p['status'] != 'COMPLIANT':
                v_desc = ", ".join(p['active_violations']) if p['active_violations'] else "Missing PPE Gear"
                inc_uid = f"INC-{session_uid}-P{p['person_id']}"
                
                # Check if this incident already exists
                existing = cur.execute('SELECT id, user_id FROM incidents WHERE incident_uid = ?', (inc_uid,)).fetchone()
                if existing and existing['user_id'] is None and owner_id is not None:
                    cur.execute('UPDATE incidents SET user_id = ?, session_uid = ? WHERE id = ? AND user_id IS NULL',
                                (owner_id, session_uid, existing['id']))
                    if cur.rowcount == 1:
                        cur.execute('UPDATE evidence SET user_id=? WHERE incident_id=? AND user_id IS NULL',
                                    (owner_id, existing['id']))
                        cur.execute('UPDATE notifications SET user_id=? WHERE incident_id=? AND user_id IS NULL',
                                    (owner_id, existing['id']))
                if not existing:
                    cur.execute('''
                        INSERT INTO incidents (
                            incident_uid, person_tracking_id, violation_type, confidence,
                            duration_seconds, zone_name, camera_name,
                            risk_score, severity, evidence_image, status, operator, session_uid, user_id
                        ) VALUES (?, ?, ?, 0.90, ?, 'Detection Workspace', ?, ?, ?, ?, 'DETECTED', 'SafetyVision AI', ?, ?)
                    ''', (
                        inc_uid,
                        p['person_id'],
                        v_desc,
                        p['duration_seconds'],
                        session_uid,
                        p['risk_score'],
                        p['risk'],
                        evidence_image,
                        session_uid,
                        owner_id
                    ))
                    incident_ids_by_person[str(p['person_id'])] = cur.lastrowid
                elif existing['user_id'] == owner_id:
                    incident_ids_by_person[str(p['person_id'])] = existing['id']

        conn.commit()
        conn.close()

        for p in clean_results:
            incident_id = incident_ids_by_person.get(str(p.get('person_id')))
            if p.get('status') == 'COMPLIANT' or incident_id is None:
                continue
            try:
                incident_uid = 'INC-{}-P{}'.format(session_uid, p.get('person_id'))
                incident_data = {
                    'incident_db_id': incident_id, 'incident_uid': incident_uid,
                    'person_id': p.get('person_id'), 'violation_type': ', '.join(p.get('active_violations') or []),
                    'risk_score': p.get('risk_score'), 'camera_id': None, 'zone_id': None, 'user_id': owner_id,
                }
                from core.evidence_manager import EvidenceManager
                session_evidence_manager = EvidenceManager(max_evidence_per_session=1)
                if evidence_image:
                    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    frame_path = evidence_image if os.path.isabs(evidence_image) else os.path.join(root_dir, evidence_image)
                    frame = cv2.imread(frame_path) if os.path.isfile(frame_path) else None
                    if frame is not None:
                        session_evidence_manager.capture_frame(frame, incident_data)
                conn = get_db()
                conn.execute('''
                    INSERT INTO notifications (title,message,severity,category,incident_id,user_id)
                    SELECT ?,?,?,?,?,?
                    WHERE NOT EXISTS (
                        SELECT 1 FROM notifications WHERE incident_id=? AND user_id IS ?
                          AND title=? AND message=?
                    )
                ''', ('Person #{} PPE VIOLATION'.format(p.get('person_id')),
                      incident_data['violation_type'] or 'PPE violation detected',
                      p.get('risk') if p.get('risk') in ('LOW','MEDIUM','HIGH','CRITICAL') else 'MEDIUM',
                      'VIOLATION', incident_id, owner_id, incident_id, owner_id,
                      'Person #{} PPE VIOLATION'.format(p.get('person_id')),
                      incident_data['violation_type'] or 'PPE violation detected'))
                conn.commit()
                conn.close()
            except Exception:
                import logging
                logging.getLogger('safetyvision.session').exception(
                    'Failed to attach supporting evidence/notification to session %s incident %s.',
                    session_uid, incident_id)

        return {
            'session_uid': session_uid,
            'people_count': total_p,
            'compliant_count': comp_p,
            'non_compliant_people': non_comp_p,
            'violation_count': viol_p,
            'ppe_violations': viol_p,
            'critical_count': crit_p,
            'overall_compliance': avg_comp,
            'safety_score': safety_score,
            'duration_seconds': round(duration_seconds, 1),
            'evidence_image': evidence_image,
            'person_results': clean_results
        }

    def _enrich_session_counts(self, sess):
        """Attach canonical display counts without touching stored rows.

        - display_violation_count: recomputed missing items (legacy rows fixed
          at read time; new rows already store item counts).
        - display_non_compliant_people: people missing >= 1 item (or unknown).
        Stored DB columns are preserved (no schema change, no rewrites).
        """
        try:
            persons = sess.get('person_results') or []
            totals = summarize_ppe(persons)
            sess['display_violation_count'] = totals['ppe_violations']
            sess['display_non_compliant_people'] = totals['non_compliant_people']
            sess['display_ppe_violations'] = totals['ppe_violations']
        except Exception:
            sess['display_violation_count'] = sess.get('violation_count', 0)
            sess['display_non_compliant_people'] = sess.get('violation_count', 0)
            sess['display_ppe_violations'] = sess.get('violation_count', 0)
        return sess

    def get_session(self, session_uid, user_id=None, include_unowned=False):
        conn = get_db()
        sql = 'SELECT * FROM detection_sessions WHERE session_uid = ?'
        params = [session_uid]
        if user_id is not None:
            sql += ' AND user_id = ?'
            params.append(user_id)
        elif not include_unowned:
            sql += ' AND user_id IS NOT NULL'
        row = conn.execute(sql, params).fetchone()
        conn.close()
        if not row:
            return None
        sess = dict(row)
        sess['person_results'] = json.loads(sess.get('person_results_json') or '[]')
        return self._enrich_session_counts(sess)

    def get_all_sessions(self, session_type=None, limit=100, search=None, date=None, user_id=None, include_unowned=False):
        """List sessions newest-first with optional server-side filters.

        - session_type: IMAGE / VIDEO / LIVE exact match (existing behaviour).
        - search: case-insensitive LIKE over session_uid, source_name, notes.
        - date: 'YYYY-MM-DD' matched against DATE(created_at).
        All filters are parameterized; schema and ordering unchanged.
        """
        conn = get_db()
        clauses = []
        params = []
        if user_id is not None:
            clauses.append('user_id = ?')
            params.append(user_id)
        elif not include_unowned:
            clauses.append('user_id IS NOT NULL')
        limit = max(1, min(int(limit or 100), 1000))
        if session_type:
            clauses.append('session_type = ?')
            params.append(session_type)
        if search:
            clauses.append('(session_uid LIKE ? OR source_name LIKE ? OR notes LIKE ?)')
            like = '%%%s%%' % search
            params.extend([like, like, like])
        if date:
            clauses.append("DATE(created_at) = DATE(?)")
            params.append(date)
        where = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
        rows = conn.execute(
            'SELECT * FROM detection_sessions %s ORDER BY id DESC LIMIT ?' % where,
            tuple(params) + (limit,)
        ).fetchall()
        conn.close()
        
        sessions = []
        for r in rows:
            s = dict(r)
            s['person_results'] = json.loads(s.get('person_results_json') or '[]')
            sessions.append(self._enrich_session_counts(s))
        return sessions

    def get_overall_stats(self, user_id=None, include_unowned=False):
        conn = get_db()
        cur = conn.cursor()

        owner_clause = 'user_id = ?' if user_id is not None else ('1=1' if include_unowned else 'user_id IS NOT NULL')
        owner_params = [user_id] if user_id is not None else []

        total_sessions = cur.execute(f'SELECT COUNT(*) FROM detection_sessions WHERE status = "COMPLETED" AND {owner_clause}', owner_params).fetchone()[0]
        img_sessions = cur.execute(f'SELECT COUNT(*) FROM detection_sessions WHERE session_type = "IMAGE" AND status = "COMPLETED" AND {owner_clause}', owner_params).fetchone()[0]
        vid_sessions = cur.execute(f'SELECT COUNT(*) FROM detection_sessions WHERE session_type = "VIDEO" AND status = "COMPLETED" AND {owner_clause}', owner_params).fetchone()[0]
        live_sessions = cur.execute(f'SELECT COUNT(*) FROM detection_sessions WHERE session_type = "LIVE" AND status = "COMPLETED" AND {owner_clause}', owner_params).fetchone()[0]

        total_people = cur.execute(f'SELECT SUM(people_count) FROM detection_sessions WHERE status = "COMPLETED" AND {owner_clause}', owner_params).fetchone()[0] or 0
        total_comp = cur.execute(f'SELECT SUM(compliant_count) FROM detection_sessions WHERE status = "COMPLETED" AND {owner_clause}', owner_params).fetchone()[0] or 0
        total_crit = cur.execute(f'SELECT SUM(critical_count) FROM detection_sessions WHERE status = "COMPLETED" AND {owner_clause}', owner_params).fetchone()[0] or 0

        # Canonical PPE-violation total: recompute missing items from stored
        # person_results_json (works for legacy person-count rows and new
        # item-count rows alike). Fall back to violation_count only when a
        # session has no parseable person results.
        total_item_viol = 0
        total_non_comp = 0
        try:
            rows = cur.execute(
                f'SELECT violation_count, person_results_json FROM detection_sessions WHERE status = "COMPLETED" AND {owner_clause}', owner_params
            ).fetchall()
            for r in rows:
                try:
                    persons = json.loads(r['person_results_json'] or '[]')
                except Exception:
                    persons = []
                if persons:
                    totals = summarize_ppe(persons)
                    total_item_viol += totals['ppe_violations']
                    total_non_comp += totals['non_compliant_people']
                else:
                    total_item_viol += int(r['violation_count'] or 0)
        except Exception:
            total_item_viol = cur.execute(f'SELECT SUM(violation_count) FROM detection_sessions WHERE status = "COMPLETED" AND {owner_clause}', owner_params).fetchone()[0] or 0

        avg_comp_raw = cur.execute(f'SELECT AVG(overall_compliance) FROM detection_sessions WHERE status = "COMPLETED" AND {owner_clause}', owner_params).fetchone()[0]
        avg_score_raw = cur.execute(f'SELECT AVG(safety_score) FROM detection_sessions WHERE status = "COMPLETED" AND {owner_clause}', owner_params).fetchone()[0]
        avg_comp = round(avg_comp_raw, 1) if avg_comp_raw is not None else None
        avg_score = round(avg_score_raw, 1) if avg_score_raw is not None else None

        # Count specific gear violations across all session JSONs
        sessions = self.get_all_sessions(limit=200, user_id=user_id, include_unowned=include_unowned)
        hh_viol = 0
        mask_viol = 0
        vest_viol = 0
        risk_dist = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}

        for s in sessions:
            for p in s.get('person_results', []):
                if p.get('hardhat') == 'MISSING': hh_viol += 1
                if p.get('mask') == 'MISSING': mask_viol += 1
                if p.get('vest') == 'MISSING': vest_viol += 1
                r = p.get('risk', 'LOW')
                risk_dist[r] = risk_dist.get(r, 0) + 1

        conn.close()

        return {
            'total_sessions': total_sessions,
            'image_sessions': img_sessions,
            'video_sessions': vid_sessions,
            'live_sessions': live_sessions,
            'total_people': total_people,
            'total_compliant': total_comp,
            'total_violations': total_item_viol,
            'total_violation_items': total_item_viol,
            'total_non_compliant_people': total_non_comp,
            'total_critical': total_crit,
            'overall_compliance': avg_comp,
            'safety_score': avg_score,
            'hardhat_violations': hh_viol,
            'mask_violations': mask_viol,
            'vest_violations': vest_viol,
            'risk_distribution': risk_dist,
            'recent_sessions': sessions[:8]
        }
