from database.db import get_db

def generate_recommendations(user_id=None, is_admin=False):
    """
    Analyzes actual database incident records and produces grounded, evidence-backed recommendations.
    Never invents arbitrary suggestions.
    """
    conn = get_db()
    cursor = conn.cursor()
    scope = '1=1' if is_admin else 'user_id = ?'
    params = [] if is_admin else [user_id]

    # Total and violation breakdown
    cursor.execute(f'''
        SELECT violation_type, COUNT(*) as count, AVG(duration_seconds) as avg_duration, AVG(risk_score) as avg_risk
        FROM incidents
        WHERE {scope}
        GROUP BY violation_type
        ORDER BY count DESC
    ''', params)
    violation_stats = [dict(row) for row in cursor.fetchall()]

    # Zone breakdown
    cursor.execute(f'''
        SELECT zone_name, COUNT(*) as count, AVG(risk_score) as avg_risk
        FROM incidents
        WHERE {scope} AND zone_name IS NOT NULL
        GROUP BY zone_name
        ORDER BY avg_risk DESC
    ''', params)
    zone_stats = [dict(row) for row in cursor.fetchall()]

    # Critical incidents
    cursor.execute(f'SELECT COUNT(*) FROM incidents WHERE {scope} AND severity = "CRITICAL"', params)
    critical_count = cursor.fetchone()[0]

    # Persistent violation check
    cursor.execute(f'SELECT COUNT(*) FROM incidents WHERE {scope} AND duration_seconds >= 10.0', params)
    long_duration_count = cursor.fetchone()[0]

    conn.close()

    recommendations = []

    if not violation_stats:
        return []

    # 1. Top Violation Analysis
    top_v = violation_stats[0]
    v_type = top_v['violation_type']
    v_count = top_v['count']

    if 'Hardhat' in v_type:
        recommendations.append({
            'id': 'REC-PPE-01',
            'priority': 'HIGH',
            'title': 'Mandatory Head Protection Reinforcement',
            'category': 'PPE Compliance',
            'observation': f"Missing Hardhat is the most prevalent violation with {v_count} occurrences.",
            'action': 'Install physical signage at entrance checkpoints and enforce automated turnstile checks before entry.'
        })
    elif 'Vest' in v_type:
        recommendations.append({
            'id': 'REC-PPE-02',
            'priority': 'HIGH',
            'title': 'High-Visibility Safety Vest Compliance Intervention',
            'category': 'PPE Compliance',
            'observation': f"Missing Safety Vest constitutes the leading violation type ({v_count} incidents).",
            'action': 'Ensure high-visibility vests are stocked at all zone lockers and issue verbal safety briefings.'
        })
    elif 'Mask' in v_type:
        recommendations.append({
            'id': 'REC-PPE-03',
            'priority': 'MEDIUM',
            'title': 'Respiratory Protection Compliance',
            'category': 'PPE Compliance',
            'observation': f"Respiratory mask violations recorded {v_count} times.",
            'action': 'Replenish protective face mask dispensers in dust and chemical handling areas.'
        })

    # 2. Zone-Specific Hazard Analysis
    if zone_stats:
        top_zone = zone_stats[0]
        if top_zone['avg_risk'] >= 60.0:
            recommendations.append({
                'id': 'REC-ZONE-01',
                'priority': 'CRITICAL',
                'title': f"Elevate Supervision in {top_zone['zone_name']}",
                'category': 'Zone Safety',
                'observation': f"{top_zone['zone_name']} displays the highest average risk score ({round(top_zone['avg_risk'], 1)}/100) across {top_zone['count']} incidents.",
                'action': 'Deploy physical safety barriers and restrict access to certified personnel only.'
            })

    # 3. Persistent Violation Duration Analysis
    if long_duration_count > 0:
        recommendations.append({
            'id': 'REC-DUR-01',
            'priority': 'HIGH',
            'title': 'Automate Real-Time Floor Intervention for Persistent Breaches',
            'category': 'Incident Response',
            'observation': f"{long_duration_count} incidents persisted longer than 10 seconds before resolution.",
            'action': 'Configure automated audible sirens or operator alert push notifications to shorten response times.'
        })

    # 4. Critical Severity Response
    if critical_count > 0:
        recommendations.append({
            'id': 'REC-CRIT-01',
            'priority': 'CRITICAL',
            'title': 'Immediate Investigation of Critical Severity Events',
            'category': 'Governance',
            'observation': f"{critical_count} critical safety incidents have been detected on record.",
            'action': 'Conduct formal root-cause analysis in the Incident Management module and record corrective notes.'
        })

    return recommendations
