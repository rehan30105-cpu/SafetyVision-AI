"""
SAFETYVISION AI — Shared Analytics Aggregation Service
=======================================================
Single source of truth for every metric on Dashboard, Analytics & Metrics pages.

Metric definitions (canonical):
  people_observed   : SUM(detection_sessions.people_count) across completed
                      sessions in scope. Scope = completed sessions within date
                      range. These are person-observations per session, NOT
                      global cross-session unique identities.
  ppe_compliance    : SUM(compliant_count) / SUM(people_count) x 100, across
                      completed sessions with people_count > 0. None when no
                      assessed people exist.
  hardhat/mask/vest compliance : assessed-present / (present + missing) x 100
                      computed from person_results_json. None when no data.
  confirmed_violations : COUNT(incidents) — one row = one confirmed violation
                      episode per person. NOT frame counts.
  critical_incidents: COUNT(incidents WHERE severity='CRITICAL') in scope.
  resolution_rate   : resolved / (total - false_positives) x 100. None when
                      no eligible incidents.
  safety_score      : documented formula (see _calc_safety_score). Range 0-100.
                      None when no incidents or sessions exist.
  hh/mask/vest violations : count of distinct person assessments with that PPE
                      missing, summed from session person_results_json.

Date-filter rules (fully parameterized, local time, inclusive start/exclusive end):
  today     : timestamp >= today 00:00:00  AND  < tomorrow 00:00:00
  yesterday : timestamp >= yesterday 00:00:00  AND  < today 00:00:00
  7days     : timestamp >= 7 days ago 00:00:00  (rolling, no upper bound)
  30days    : timestamp >= 30 days ago 00:00:00 (rolling, no upper bound)
  all       : no filter
"""

import json
from datetime import datetime, timedelta
from database.db import get_db


# ─── Internal helpers ────────────────────────────────────────

def _range_to_dates(time_range):
    """
    Returns (start_dt, end_dt) as local datetime objects.
    start_dt: inclusive, end_dt: exclusive, None = unbounded.
    """
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    r = (time_range or 'all').lower()
    if r == 'today':
        return today, today + timedelta(days=1)
    elif r == 'yesterday':
        return today - timedelta(days=1), today
    elif r == '7days':
        return today - timedelta(days=6), None
    elif r == '30days':
        return today - timedelta(days=29), None
    else:
        return None, None


def _previous_range_dates(time_range):
    """
    Returns (start_dt, end_dt) for the immediately-preceding equivalent window,
    used only to compute honest period-over-period trend deltas. Returns
    (None, None) when there is no meaningful comparison window (e.g. 'all').
    """
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    r = (time_range or 'all').lower()
    if r == 'today':
        return today - timedelta(days=1), today
    elif r == 'yesterday':
        return today - timedelta(days=2), today - timedelta(days=1)
    elif r == '7days':
        return today - timedelta(days=13), today - timedelta(days=6)
    elif r == '30days':
        return today - timedelta(days=59), today - timedelta(days=29)
    else:
        return None, None


def _build_date_filter(start_dt, end_dt, col):
    """
    Returns (where_clause, params) for parameterized SQL.
    col must be a trusted column name (not from user input).
    """
    parts, params = [], []
    if start_dt is not None:
        parts.append(f"{col} >= ?")
        params.append(start_dt.strftime('%Y-%m-%d %H:%M:%S'))
    if end_dt is not None:
        parts.append(f"{col} < ?")
        params.append(end_dt.strftime('%Y-%m-%d %H:%M:%S'))
    return (' AND '.join(parts) if parts else '1=1'), params


def _range_to_filter(time_range, col):
    """Public helper: returns (where_fragment, params) for parameterized SQL."""
    start, end = _range_to_dates(time_range)
    return _build_date_filter(start, end, col)


def _calc_safety_score(avg_risk, compliance_rate, critical_pct):
    """
    Project safety-score formula (range 0-100, documented).

    Components:
      35% — avg_risk           (0-100; higher risk → lower score)
      35% — 100 - compliance   (lower compliance → lower score)
      30% — critical_pct       (share of incidents that are CRITICAL x 100)

    score = 100 - (0.35*avg_risk + 0.35*(100-compliance) + 0.30*critical_pct)
    Clamped to [0, 100].
    """
    penalty = (0.35 * avg_risk
               + 0.35 * (100.0 - compliance_rate)
               + 0.30 * critical_pct)
    return round(max(0.0, min(100.0, 100.0 - penalty)), 1)


def _ppe_from_sessions(sessions_list):
    """
    Calculates per-equipment and overall PPE compliance from session person_results.
    Returns dict with:
      overall, hardhat, mask, vest   — float % or None (no data)
      hh_violations, mask_violations, vest_violations — int person-counts
      people_assessed, people_compliant — int totals
    """
    hh_present = hh_missing = 0
    mask_present = mask_missing = 0
    vest_present = vest_missing = 0
    total_people = 0
    total_compliant = 0

    for s in sessions_list:
        results = s.get('person_results') or []
        if isinstance(results, str):
            try:
                results = json.loads(results)
            except Exception:
                results = []
        for p in results:
            total_people += 1
            if float(p.get('compliance', 0.0)) >= 99.0:
                total_compliant += 1
            hh = str(p.get('hardhat', '')).upper()
            mk = str(p.get('mask', '')).upper()
            vt = str(p.get('vest', '')).upper()
            if hh == 'PRESENT':
                hh_present += 1
            elif hh == 'MISSING':
                hh_missing += 1
            if mk == 'PRESENT':
                mask_present += 1
            elif mk == 'MISSING':
                mask_missing += 1
            if vt == 'PRESENT':
                vest_present += 1
            elif vt == 'MISSING':
                vest_missing += 1

    def _pct(present, missing):
        total = present + missing
        return round(present / total * 100.0, 1) if total > 0 else None

    return {
        'overall':        round(total_compliant / total_people * 100.0, 1) if total_people > 0 else None,
        'hardhat':        _pct(hh_present, hh_missing),
        'mask':           _pct(mask_present, mask_missing),
        'vest':           _pct(vest_present, vest_missing),
        'hh_violations':  hh_missing,
        'mask_violations': mask_missing,
        'vest_violations': vest_missing,
        'hh_present':     hh_present,
        'mask_present':   mask_present,
        'vest_present':   vest_present,
        'people_assessed': total_people,
        'people_compliant': total_compliant,
    }


def _load_sessions(cur, sess_clause, sess_params):
    """Loads completed sessions matching parameterized filter, parses person_results_json."""
    sql = (
        'SELECT people_count, compliant_count, violation_count, critical_count, '
        'overall_compliance, safety_score, person_results_json, people_count '
        'FROM detection_sessions '
        f'WHERE status="COMPLETED" AND people_count > 0 AND ({sess_clause})'
    )
    rows = cur.execute(sql, sess_params).fetchall()
    result = []
    for row in rows:
        s = dict(row)
        try:
            s['person_results'] = json.loads(s.get('person_results_json') or '[]')
        except Exception:
            s['person_results'] = []
        result.append(s)
    return result


# ─── Public API ──────────────────────────────────────────────

def get_dashboard_metrics(user_id=None, is_admin=False):
    """
    Returns all metrics for the Dashboard page (always all-time scope).
    The dashboard is the global health overview; date-filtered analysis
    belongs on the Analytics page.
    """
    conn = get_db()
    cur = conn.cursor()

    # Admin dashboard is deliberately global. Normal accounts receive only
    # their owned data; NULL-owner legacy rows are not exposed to users.
    incident_scope = '1=1' if is_admin else 'user_id = ?'
    session_scope = '1=1' if is_admin else 'user_id = ?'
    owner_params = [] if is_admin else [user_id]

    # Incidents — all time
    total_incidents  = cur.execute(f'SELECT COUNT(*) FROM incidents WHERE {incident_scope}', owner_params).fetchone()[0]
    critical_count   = cur.execute(f'SELECT COUNT(*) FROM incidents WHERE {incident_scope} AND severity="CRITICAL"', owner_params).fetchone()[0]
    resolved_count   = cur.execute(f'SELECT COUNT(*) FROM incidents WHERE {incident_scope} AND status="RESOLVED"', owner_params).fetchone()[0]
    false_pos_count  = cur.execute(f'SELECT COUNT(*) FROM incidents WHERE {incident_scope} AND status="FALSE_POSITIVE"', owner_params).fetchone()[0]
    unresolved_count = cur.execute(
        f'SELECT COUNT(*) FROM incidents WHERE {incident_scope} AND status NOT IN ("RESOLVED","FALSE_POSITIVE")', owner_params
    ).fetchone()[0]

    avg_risk_row = cur.execute(f'SELECT AVG(risk_score) FROM incidents WHERE {incident_scope}', owner_params).fetchone()[0]
    avg_risk = round(float(avg_risk_row), 1) if avg_risk_row is not None else 0.0

    eligible = max(total_incidents - false_pos_count, 0)
    resolution_rate = round(resolved_count / eligible * 100.0, 1) if eligible > 0 else None

    # Sessions — all time, with people
    sessions_list = _load_sessions(cur, session_scope, owner_params)
    ppe_data = _ppe_from_sessions(sessions_list)
    people_observed = sum(s['people_count'] or 0 for s in sessions_list)

    ppe_compliance = ppe_data['overall']
    critical_pct   = round(critical_count / max(total_incidents, 1) * 100.0, 1) if total_incidents > 0 else 0.0
    comp_for_score = ppe_compliance if ppe_compliance is not None else 100.0
    safety_score   = _calc_safety_score(avg_risk, comp_for_score, critical_pct) if (total_incidents > 0 or len(sessions_list) > 0) else None

    # Last-7-days trend for dashboard chart
    now = datetime.now()
    trend_labels     = [(now - timedelta(days=i)).strftime('%b %d') for i in range(6, -1, -1)]
    trend_violations = []
    trend_risk       = []
    for i in range(6, -1, -1):
        d_str = (now - timedelta(days=i)).strftime('%Y-%m-%d')
        cnt = cur.execute(f"SELECT COUNT(*) FROM incidents WHERE {incident_scope} AND DATE(timestamp)=?", owner_params + [d_str]).fetchone()[0]
        ravg = cur.execute(f"SELECT AVG(risk_score) FROM incidents WHERE {incident_scope} AND DATE(timestamp)=?", owner_params + [d_str]).fetchone()[0]
        trend_violations.append(cnt)
        trend_risk.append(round(ravg or 0.0, 1))

    # Risk / severity distribution — real values only, no fake fallback
    risk_dist = [
        cur.execute(f'SELECT COUNT(*) FROM incidents WHERE {incident_scope} AND severity="LOW"', owner_params).fetchone()[0],
        cur.execute(f'SELECT COUNT(*) FROM incidents WHERE {incident_scope} AND severity="MEDIUM"', owner_params).fetchone()[0],
        cur.execute(f'SELECT COUNT(*) FROM incidents WHERE {incident_scope} AND severity="HIGH"', owner_params).fetchone()[0],
        cur.execute(f'SELECT COUNT(*) FROM incidents WHERE {incident_scope} AND severity="CRITICAL"', owner_params).fetchone()[0],
    ]

    # Zone breakdown — real data only
    zone_rows = cur.execute(f'''
        SELECT zone_name, COUNT(*) as cnt
        FROM incidents
        WHERE {incident_scope} AND zone_name IS NOT NULL AND zone_name != ""
        GROUP BY zone_name ORDER BY cnt DESC LIMIT 5
    ''', owner_params).fetchall()
    z_names  = [r['zone_name'] for r in zone_rows]
    z_counts = [r['cnt'] for r in zone_rows]

    recent_incidents = [dict(r) for r in cur.execute(
        f'SELECT * FROM incidents WHERE {incident_scope} ORDER BY id DESC LIMIT 6', owner_params
    ).fetchall()]

    conn.close()

    return {
        'kpis': {
            'people_observed':    people_observed,
            'ppe_compliance':     ppe_compliance,       # float or None
            'total_violations':   total_incidents,
            'unresolved_count':   unresolved_count,
            'critical_incidents': critical_count,
            'resolved_incidents': resolved_count,
            'resolution_rate':    resolution_rate,      # float or None
            'average_risk':       avg_risk,
            'safety_score':       safety_score,         # float or None
        },
        'charts': {
            'trend_labels':      trend_labels,
            'trend_violations':  trend_violations,
            'trend_risk':        trend_risk,
            'risk_distribution': risk_dist,
            'hh_rate':           ppe_data['hardhat'],   # float or None
            'mask_rate':         ppe_data['mask'],
            'vest_rate':         ppe_data['vest'],
            'hh_violations':     ppe_data['hh_violations'],
            'mask_violations':   ppe_data['mask_violations'],
            'vest_violations':   ppe_data['vest_violations'],
            'zone_names':        z_names,
            'zone_counts':       z_counts,
        },
        'recent_incidents': recent_incidents,
        'has_sessions':  len(sessions_list) > 0,
        'has_incidents': total_incidents > 0,
    }


def get_analytics_metrics(time_range='7days', user_id=None, is_admin=False):
    """
    Returns all metrics for the Analytics page, filtered by time_range.
    All KPIs, chart data, and summary statistics respect the same date filter.
    """
    conn = get_db()
    cur = conn.cursor()

    inc_date_clause, inc_date_params = _range_to_filter(time_range, 'timestamp')
    sess_date_clause, sess_date_params = _range_to_filter(time_range, 'created_at')
    owner_clause = '1=1' if is_admin else 'user_id = ?'
    owner_params = [] if is_admin else [user_id]
    inc_clause = f'({inc_date_clause}) AND ({owner_clause})'
    inc_params = inc_date_params + owner_params
    sess_clause = f'({sess_date_clause}) AND ({owner_clause})'
    sess_params = sess_date_params + owner_params

    def _q1(sql, extra=None):
        """Run a single-column query with optional extra params appended."""
        p = inc_params + (extra or [])
        return cur.execute(sql, p).fetchone()[0]

    # Incidents in scope (all parameterized)
    total_incidents = _q1(f'SELECT COUNT(*) FROM incidents WHERE {inc_clause}')
    critical_count  = _q1(f'SELECT COUNT(*) FROM incidents WHERE {inc_clause} AND severity="CRITICAL"')
    resolved_count  = _q1(f'SELECT COUNT(*) FROM incidents WHERE {inc_clause} AND status="RESOLVED"')
    false_pos_count = _q1(f'SELECT COUNT(*) FROM incidents WHERE {inc_clause} AND status="FALSE_POSITIVE"')

    avg_risk_row = _q1(f'SELECT AVG(risk_score) FROM incidents WHERE {inc_clause}')
    avg_risk = round(float(avg_risk_row), 1) if avg_risk_row is not None else 0.0

    avg_dur_row = _q1(f'SELECT AVG(duration_seconds) FROM incidents WHERE {inc_clause}')
    avg_dur = round(float(avg_dur_row), 1) if avg_dur_row is not None else 0.0

    eligible = max(total_incidents - false_pos_count, 0)
    resolution_rate = round(resolved_count / eligible * 100.0, 1) if eligible > 0 else None

    # Sessions in scope
    sessions_list = _load_sessions(cur, sess_clause, sess_params)
    ppe_data = _ppe_from_sessions(sessions_list)
    ppe_compliance = ppe_data['overall']

    critical_pct   = round(critical_count / max(total_incidents, 1) * 100.0, 1) if total_incidents > 0 else 0.0
    comp_for_score = ppe_compliance if ppe_compliance is not None else 100.0
    safety_score   = _calc_safety_score(avg_risk, comp_for_score, critical_pct) if (total_incidents > 0 or len(sessions_list) > 0) else None

    # Violation types (parameterized)
    v_rows = cur.execute(
        f'SELECT violation_type, COUNT(*) as cnt FROM incidents '
        f'WHERE {inc_clause} GROUP BY violation_type ORDER BY cnt DESC LIMIT 6',
        inc_params
    ).fetchall()
    v_labels = [r['violation_type'] for r in v_rows]
    v_counts = [r['cnt'] for r in v_rows]

    # Severity spread (parameterized)
    sev_counts = [
        _q1(f'SELECT COUNT(*) FROM incidents WHERE {inc_clause} AND severity="LOW"'),
        _q1(f'SELECT COUNT(*) FROM incidents WHERE {inc_clause} AND severity="MEDIUM"'),
        _q1(f'SELECT COUNT(*) FROM incidents WHERE {inc_clause} AND severity="HIGH"'),
        _q1(f'SELECT COUNT(*) FROM incidents WHERE {inc_clause} AND severity="CRITICAL"'),
    ]

    # Daily / temporal trend
    now = datetime.now()
    if time_range in ('today', 'yesterday'):
        base_date = now if time_range == 'today' else now - timedelta(days=1)
        d_str = base_date.strftime('%Y-%m-%d')
        d_labels, d_counts = [], []
        for h in range(0, 24, 4):
            cnt = cur.execute(
                f"SELECT COUNT(*) FROM incidents WHERE {owner_clause} AND DATE(timestamp)=? AND STRFTIME('%H',timestamp)>=? AND STRFTIME('%H',timestamp)<?",
                owner_params + [d_str, f"{h:02d}", f"{h+4:02d}"]
            ).fetchone()[0]
            d_labels.append(f"{h:02d}:00")
            d_counts.append(cnt)
    elif time_range == '30days':
        d_labels, d_counts = [], []
        for i in range(29, -1, -1):
            d_str = (now - timedelta(days=i)).strftime('%Y-%m-%d')
            cnt = cur.execute(f"SELECT COUNT(*) FROM incidents WHERE {owner_clause} AND DATE(timestamp)=?", owner_params + [d_str]).fetchone()[0]
            d_labels.append((now - timedelta(days=i)).strftime('%b %d'))
            d_counts.append(cnt)
    elif time_range == 'all':
        rows = cur.execute(f'''
            SELECT STRFTIME('%Y-W%W', timestamp) as wk, COUNT(*) as cnt
            FROM incidents WHERE {owner_clause} GROUP BY wk ORDER BY wk
        ''', owner_params).fetchall()
        d_labels = [r['wk'] for r in rows] or ['No data']
        d_counts = [r['cnt'] for r in rows] or [0]
    else:  # 7days
        d_labels, d_counts = [], []
        for i in range(6, -1, -1):
            d_str = (now - timedelta(days=i)).strftime('%Y-%m-%d')
            cnt = cur.execute(f"SELECT COUNT(*) FROM incidents WHERE {owner_clause} AND DATE(timestamp)=?", owner_params + [d_str]).fetchone()[0]
            d_labels.append((now - timedelta(days=i)).strftime('%b %d'))
            d_counts.append(cnt)

    # Zone breakdown (parameterized)
    z_rows = cur.execute(
        f'SELECT zone_name, COUNT(*) as cnt FROM incidents '
        f'WHERE {inc_clause} AND zone_name IS NOT NULL AND zone_name != "" '
        f'GROUP BY zone_name ORDER BY cnt DESC LIMIT 5',
        inc_params
    ).fetchall()
    z_labels = [r['zone_name'] for r in z_rows]
    z_counts  = [r['cnt'] for r in z_rows]

    # Recent incidents (date-filtered, newest first) — for the Analytics table.
    recent_rows = cur.execute(
        f'SELECT id, incident_uid, timestamp, person_tracking_id, violation_type, '
        f'risk_score, severity, status, evidence_image, zone_name, camera_name '
        f'FROM incidents WHERE {inc_clause} ORDER BY timestamp DESC, id DESC LIMIT 8',
        inc_params
    ).fetchall()
    recent_incidents = [dict(r) for r in recent_rows]

    # Previous-equivalent-period totals for honest trend indicators (no fake data).
    prev_total = prev_critical = prev_risk_avg = prev_resolved = prev_fp = None
    prev_start, prev_end = _previous_range_dates(time_range)
    if prev_start is not None:
        p_date_clause, p_date_params = _build_date_filter(prev_start, prev_end, 'timestamp')
        p_clause = f'({p_date_clause}) AND ({owner_clause})'
        p_params = p_date_params + owner_params
        prev_total    = cur.execute(f'SELECT COUNT(*) FROM incidents WHERE {p_clause}', p_params).fetchone()[0]
        prev_critical = cur.execute(f'SELECT COUNT(*) FROM incidents WHERE {p_clause} AND severity="CRITICAL"', p_params).fetchone()[0]
        prev_resolved = cur.execute(f'SELECT COUNT(*) FROM incidents WHERE {p_clause} AND status="RESOLVED"', p_params).fetchone()[0]
        prev_fp       = cur.execute(f'SELECT COUNT(*) FROM incidents WHERE {p_clause} AND status="FALSE_POSITIVE"', p_params).fetchone()[0]
        _pravg = cur.execute(f'SELECT AVG(risk_score) FROM incidents WHERE {p_clause}', p_params).fetchone()[0]
        prev_risk_avg = round(float(_pravg), 1) if _pravg is not None else 0.0
    prev_eligible = max((prev_total or 0) - (prev_fp or 0), 0)
    prev_resolution = round((prev_resolved or 0) / prev_eligible * 100.0, 1) if prev_eligible > 0 else None

    conn.close()

    range_labels = {
        'today': 'Today', 'yesterday': 'Yesterday',
        '7days': 'Past 7 Days', '30days': 'Past 30 Days', 'all': 'All Time'
    }

    return {
        'stats': {
            'total_incidents':    total_incidents,
            'avg_risk':           avg_risk,
            'avg_duration':       avg_dur,
            'critical_incidents': critical_count,
            'resolution_rate':    resolution_rate,   # float or None
            'ppe_compliance':     ppe_compliance,    # float or None
            'site_safety_score':  safety_score,      # float or None
            'people_assessed':    ppe_data['people_assessed'],
        },
        'charts': {
            'viol_labels':     v_labels,
            'viol_counts':     v_counts,
            'severity_counts': sev_counts,
            'daily_labels':    d_labels,
            'daily_counts':    d_counts,
            'zone_labels':     z_labels,
            'zone_counts':     z_counts,
            'hh_rate':         ppe_data['hardhat'],   # float or None
            'mask_rate':       ppe_data['mask'],
            'vest_rate':       ppe_data['vest'],
            'hh_violations':   ppe_data['hh_violations'],
            'mask_violations': ppe_data['mask_violations'],
            'vest_violations': ppe_data['vest_violations'],
            'hh_present':      ppe_data['hh_present'],
            'mask_present':    ppe_data['mask_present'],
            'vest_present':    ppe_data['vest_present'],
        },
        'recent_incidents': recent_incidents,
        'trends': {
            'incidents_prev':  prev_total,       # int or None (None = no comparison window)
            'critical_prev':   prev_critical,
            'risk_prev':       prev_risk_avg,
            'resolution_prev': prev_resolution,
        },
        'time_range':    time_range,
        'range_label':   range_labels.get(time_range, 'Past 7 Days'),
        'has_incidents': total_incidents > 0,
        'has_sessions':  len(sessions_list) > 0,
    }
