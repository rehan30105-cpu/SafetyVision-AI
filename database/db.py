import sqlite3
import os
import json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'safetyvision.db')

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('ADMIN', 'SAFETY_OFFICER', 'OPERATOR')),
        full_name TEXT,
        is_active INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )
    ''')

    # Cameras table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cameras (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        location TEXT NOT NULL,
        stream_source TEXT NOT NULL CHECK(stream_source IN ('webcam', 'upload', 'rtsp', 'ip')),
        rtsp_url TEXT,
        assigned_zone_id INTEGER,
        status TEXT DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE', 'INACTIVE', 'OFFLINE')),
        fps REAL DEFAULT 30.0,
        resolution TEXT DEFAULT '1280x720',
        last_frame_time TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Zones table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS zones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        zone_type TEXT NOT NULL CHECK(zone_type IN ('ENTRY', 'PPE_CHECK', 'MACHINERY', 'VEHICLE', 'CONSTRUCTION', 'RESTRICTED')),
        description TEXT,
        polygon_coords TEXT NOT NULL, -- JSON array of [x, y] coordinates (normalized 0-1)
        required_ppe TEXT NOT NULL, -- JSON array of strings e.g. ["Hardhat", "Mask", "Safety Vest"]
        risk_multiplier REAL DEFAULT 1.0,
        is_active INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Incidents table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_uid TEXT UNIQUE NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        person_tracking_id INTEGER,
        violation_type TEXT NOT NULL,
        confidence REAL,
        duration_seconds REAL DEFAULT 0.0,
        zone_id INTEGER,
        zone_name TEXT,
        camera_id INTEGER,
        camera_name TEXT,
        risk_score REAL NOT NULL,
        severity TEXT NOT NULL CHECK(severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
        evidence_image TEXT,
        status TEXT DEFAULT 'DETECTED' CHECK(status IN ('DETECTED', 'ACKNOWLEDGED', 'IN_PROGRESS', 'RESOLVED', 'FALSE_POSITIVE')),
        resolution_notes TEXT,
        operator TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        resolved_at TIMESTAMP,
        FOREIGN KEY (zone_id) REFERENCES zones(id) ON DELETE SET NULL,
        FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE SET NULL
    )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_incidents_timestamp ON incidents(timestamp)')

    # Evidence table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS evidence (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id INTEGER,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        person_tracking_id INTEGER,
        violation_type TEXT NOT NULL,
        risk_score REAL,
        camera_id INTEGER,
        zone_id INTEGER,
        image_path TEXT NOT NULL,
        file_size INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (incident_id) REFERENCES incidents(id) ON DELETE CASCADE
    )
    ''')

    # Notifications table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS notifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        message TEXT NOT NULL,
        severity TEXT NOT NULL CHECK(severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
        category TEXT NOT NULL CHECK(category IN ('VIOLATION', 'ZONE_BREACH', 'HIGH_RISK', 'INSPECTION', 'CAMERA', 'SYSTEM')),
        incident_id INTEGER,
        is_read INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (incident_id) REFERENCES incidents(id) ON DELETE SET NULL
    )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_notifications_read ON notifications(is_read)')

    # Detection Sessions (Mandatory for Image, Video and Live separation)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS detection_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_uid TEXT UNIQUE NOT NULL,
        session_type TEXT NOT NULL CHECK(session_type IN ('IMAGE', 'VIDEO', 'LIVE')),
        source_name TEXT,
        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        end_time TIMESTAMP,
        duration_seconds REAL DEFAULT 0,
        people_count INTEGER DEFAULT 0,
        compliant_count INTEGER DEFAULT 0,
        violation_count INTEGER DEFAULT 0,
        critical_count INTEGER DEFAULT 0,
        overall_compliance REAL DEFAULT 100.0,
        safety_score REAL DEFAULT 100.0,
        evidence_image TEXT,
        status TEXT DEFAULT 'COMPLETED',
        person_results_json TEXT,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_sessions_uid ON detection_sessions(session_uid)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_sessions_type ON detection_sessions(session_type)')


    # Audit Logs
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        username TEXT NOT NULL,
        action TEXT NOT NULL,
        details TEXT,
        ip_address TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_logs(created_at)')

    # Additive ownership migration. Existing rows remain NULL because their
    # original account cannot be established reliably from legacy records.
    ownership_columns = {
        'detection_sessions': {'user_id': 'INTEGER'},
        'incidents': {'user_id': 'INTEGER', 'session_uid': 'TEXT'},
        'evidence': {'user_id': 'INTEGER'},
        'notifications': {'user_id': 'INTEGER'},
    }
    for table, columns in ownership_columns.items():
        existing_columns = {row['name'] for row in cursor.execute('PRAGMA table_info("{}")'.format(table)).fetchall()}
        for column, sql_type in columns.items():
            if column not in existing_columns:
                cursor.execute('ALTER TABLE "{}" ADD COLUMN "{}" {}'.format(table, column, sql_type))
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_sessions_owner ON detection_sessions(user_id, created_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_incidents_owner ON incidents(user_id, timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_evidence_owner ON evidence(user_id, created_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_notifications_owner ON notifications(user_id, created_at)')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS generated_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id INTEGER,
            file_path TEXT NOT NULL UNIQUE,
            report_type TEXT NOT NULL,
            session_uid TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_generated_reports_owner ON generated_reports(owner_id, created_at)')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS demo_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL COLLATE NOCASE,
        company TEXT NOT NULL,
        role TEXT NOT NULL,
        message TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN ('PENDING', 'APPROVED', 'REJECTED')),
        review_notes TEXT,
        reviewed_by INTEGER,
        reviewed_at TIMESTAMP,
        activation_token_hash TEXT,
        activation_expires_at TIMESTAMP,
        activated_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (reviewed_by) REFERENCES users(id) ON DELETE SET NULL
    )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_demo_requests_status ON demo_requests(status, created_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_demo_requests_email ON demo_requests(email)')
    existing_columns = {row[1] for row in cursor.execute('PRAGMA table_info(demo_requests)').fetchall()}
    for column, definition in {
        'email_status': "TEXT NOT NULL DEFAULT 'PENDING'",
        'email_error': 'TEXT',
        'email_attempts': 'INTEGER NOT NULL DEFAULT 0',
        'email_sent_at': 'TIMESTAMP',
        'password_hash': 'TEXT',
        'request_email_status': "TEXT NOT NULL DEFAULT 'PENDING'",
        'request_email_error': 'TEXT',
        'admin_email_status': "TEXT NOT NULL DEFAULT 'PENDING'",
        'admin_email_error': 'TEXT',
        'decision_email_status': "TEXT NOT NULL DEFAULT 'PENDING'",
        'decision_email_error': 'TEXT',
        'email_approval_nonce_hash': 'TEXT',
        'email_rejection_nonce_hash': 'TEXT',
        'telegram_approval_status': "TEXT NOT NULL DEFAULT 'UNKNOWN'",
        'telegram_approval_error': 'TEXT',
        'telegram_approval_message_id': 'INTEGER',
        'telegram_approval_sent_at': 'TIMESTAMP',
    }.items():
        if column not in existing_columns:
            cursor.execute('ALTER TABLE demo_requests ADD COLUMN {} {}'.format(column, definition))

    # System Settings
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS system_settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT UNIQUE NOT NULL,
        value TEXT NOT NULL,
        category TEXT NOT NULL CHECK(category IN ('ACCOUNT', 'DETECTION', 'TRACKING', 'SAFETY', 'ALERTS', 'SYSTEM')),
        description TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Never create public/default credentials. Existing accounts remain intact;
    # administrators provision initial access through the deployment process.

    # Seed Default Cameras
    cursor.execute('SELECT COUNT(*) FROM cameras')
    if cursor.fetchone()[0] == 0:
        default_cameras = [
            ('Cam 01 - Main Floor', 'Shop Floor Bay A', 'webcam', '', 1, 'ACTIVE', 30.0, '1280x720'),
            ('Cam 02 - Assembly Line', 'Heavy Machinery Zone', 'upload', '', 2, 'ACTIVE', 30.0, '1920x1080'),
            ('Cam 03 - Forklift Crossing', 'Logistics Terminal B', 'rtsp', 'rtsp://127.0.0.1:8554/live', 3, 'INACTIVE', 25.0, '1280x720'),
            ('Cam 04 - Site Perimeter', 'Gate 2 Entrance', 'ip', 'http://192.168.1.100:8080/video', 4, 'ACTIVE', 30.0, '1280x720')
        ]
        cursor.executemany('''
            INSERT INTO cameras (name, location, stream_source, rtsp_url, assigned_zone_id, status, fps, resolution)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', default_cameras)

    # Seed Default Zones
    cursor.execute('SELECT COUNT(*) FROM zones')
    if cursor.fetchone()[0] == 0:
        default_zones = [
            ('Main Walkway', 'ENTRY', 'Factory pedestrian passage and entrance', json.dumps([[0.05, 0.1], [0.45, 0.1], [0.45, 0.9], [0.05, 0.9]]), json.dumps(['Hardhat', 'Safety Vest']), 1.0, 1),
            ('Heavy Machinery Bay', 'MACHINERY', 'CNC and stamping press high-hazard area', json.dumps([[0.5, 0.1], [0.95, 0.1], [0.95, 0.6], [0.5, 0.6]]), json.dumps(['Hardhat', 'Safety Vest', 'Mask']), 1.5, 1),
            ('Vehicle Transit Path', 'VEHICLE', 'Forklift and transport loading corridor', json.dumps([[0.5, 0.65], [0.95, 0.65], [0.95, 0.95], [0.5, 0.95]]), json.dumps(['Hardhat', 'Safety Vest']), 1.3, 1),
            ('Restricted Substation', 'RESTRICTED', 'High voltage transformer area - authorized personnel only', json.dumps([[0.1, 0.65], [0.4, 0.65], [0.4, 0.95], [0.1, 0.95]]), json.dumps(['Hardhat', 'Safety Vest', 'Mask']), 2.0, 1)
        ]
        cursor.executemany('''
            INSERT INTO zones (name, zone_type, description, polygon_coords, required_ppe, risk_multiplier, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', default_zones)

    # Seed Default Settings
    cursor.execute('SELECT COUNT(*) FROM system_settings')
    if cursor.fetchone()[0] == 0:
        default_settings = [
            ('confidence_threshold', '0.35', 'DETECTION', 'Minimum confidence threshold for YOLO detections'),
            ('iou_threshold', '0.45', 'DETECTION', 'IoU threshold for Non-Maximum Suppression'),
            ('inference_resolution', '640', 'DETECTION', 'Inference frame resolution dimension'),
            ('tracking_enabled', 'true', 'TRACKING', 'Enable real-time person tracking across frames'),
            ('tracking_max_lost', '30', 'TRACKING', 'Max frames to keep tracked person without match'),
            ('persistence_threshold_seconds', '1.5', 'SAFETY', 'Seconds violation must persist before incident confirmation'),
            ('escalation_threshold_seconds', '5.0', 'SAFETY', 'Seconds before persistent violation escalates risk level'),
            ('risk_weight_hardhat', '25', 'SAFETY', 'Risk contribution score for missing hardhat'),
            ('risk_weight_vest', '25', 'SAFETY', 'Risk contribution score for missing safety vest'),
            ('risk_weight_mask', '15', 'SAFETY', 'Risk contribution score for missing mask'),
            ('risk_weight_machinery_near', '20', 'SAFETY', 'Risk penalty when violation is near active machinery'),
            ('risk_weight_vehicle_near', '20', 'SAFETY', 'Risk penalty when violation is near active vehicle'),
            ('risk_weight_zone_breach', '25', 'SAFETY', 'Risk penalty for entering unauthorized restricted zone'),
            ('alert_cooldown_seconds', '10.0', 'ALERTS', 'Cooldown window between repeat alerts for the same person'),
            ('alert_sound_enabled', 'true', 'ALERTS', 'Enable audio chime for critical/high violations'),
            ('alert_browser_notification', 'true', 'ALERTS', 'Trigger browser native notifications'),
            ('evidence_auto_capture', 'true', 'SYSTEM', 'Automatically save evidence snapshots for high/critical incidents'),
            ('max_evidence_per_session', '50', 'SYSTEM', 'Maximum evidence frames stored per session')
        ]
        cursor.executemany('''
            INSERT INTO system_settings (key, value, category, description)
            VALUES (?, ?, ?, ?)
        ''', default_settings)

    conn.commit()
    conn.close()

def log_audit(user_id, username, action, details='', ip_address='127.0.0.1'):
    try:
        conn = get_db()
        conn.execute('''
            INSERT INTO audit_logs (user_id, username, action, details, ip_address)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, username, action, details, ip_address))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging audit: {e}")

def get_setting(key, default=None):
    try:
        conn = get_db()
        row = conn.execute('SELECT value FROM system_settings WHERE key = ?', (key,)).fetchone()
        conn.close()
        if row:
            return row['value']
    except Exception:
        pass
    return default
