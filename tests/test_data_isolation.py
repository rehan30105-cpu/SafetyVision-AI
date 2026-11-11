"""Security regression tests for session, incident, analytics and account views."""
import os
import tempfile
import base64
import secrets

import cv2
import numpy as np

import pytest
from werkzeug.security import generate_password_hash

from database import db
from services.analytics_service import get_analytics_metrics


@pytest.fixture
def isolation_app(monkeypatch):
    fd, path = tempfile.mkstemp(prefix='safetyvision-isolation-', suffix='.db')
    os.close(fd)
    os.remove(path)
    monkeypatch.setattr(db, 'DB_PATH', path)
    import app as application
    import services.report_service as report_service
    report_dir = tempfile.mkdtemp(prefix='safetyvision-isolation-reports-')
    os.makedirs(report_dir, exist_ok=True)
    monkeypatch.setattr(report_service, 'REPORTS_DIR', report_dir)
    monkeypatch.setattr(application, 'REPORTS_FOLDER', report_dir)
    static_root = tempfile.mkdtemp(prefix='safetyvision-isolation-static-')
    os.makedirs(os.path.join(static_root, 'reports'), exist_ok=True)
    monkeypatch.setattr(application.app, 'static_folder', static_root)
    monkeypatch.setattr(report_service, 'REPORTS_DIR', os.path.join(static_root, 'reports'))
    monkeypatch.setattr(application, 'REPORTS_FOLDER', os.path.join(static_root, 'reports'))
    application.init_db()
    conn = db.get_db()
    user_ids = {}
    for username, role in [('scope-admin', 'ADMIN'), ('scope-a', 'OPERATOR'), ('scope-b', 'OPERATOR')]:
        user_ids[username] = conn.execute(
            'INSERT INTO users(username,email,password_hash,role,full_name,is_active) VALUES(?,?,?,?,?,1)',
            (username, username + '@example.test', generate_password_hash('isolation-test-password'), role, username)).lastrowid
    conn.commit()
    conn.close()

    session_ids = {}
    for username in ('scope-a', 'scope-b'):
        application.pipeline.reset_tracker(user_id=user_ids[username])
        session_ids[username], _ = application.session_manager.create_session('IMAGE', username, user_id=user_ids[username])
        results = [{'person_id': 1, 'compliance': 0.0, 'hardhat': 'MISSING', 'mask': 'PRESENT', 'vest': 'PRESENT',
                    'missing_items': ['Hardhat'], 'active_violations': ['Missing Hardhat'],
                    'risk': 'HIGH', 'risk_score': 60.0}]
        application.session_manager.complete_session(
            session_ids[username], results, None, user_id=user_ids[username])

    yield application, application.app.test_client(), user_ids, session_ids
    try:
        os.remove(path)
    except OSError:
        pass
    import shutil
    shutil.rmtree(report_dir, ignore_errors=True)
    shutil.rmtree(static_root, ignore_errors=True)


def _sign_in(client, user_id, role):
    with client.session_transaction() as sess:
        sess.update(user_id=user_id, role=role, _csrf_token=secrets.token_urlsafe(24))


def _csrf_headers(client):
    with client.session_transaction() as sess:
        return {'X-CSRF-Token': sess['_csrf_token']}


def _csrf_headers(client):
    with client.session_transaction() as sess:
        return {'X-CSRF-Token': sess['_csrf_token']}


@pytest.mark.parametrize(('owner', 'other'), [('scope-a', 'scope-b'), ('scope-b', 'scope-a')])
def test_user_scope_is_consistent_for_history_incidents_exports_and_reports(isolation_app, owner, other):
    _, client, user_ids, session_ids = isolation_app
    _sign_in(client, user_ids[owner], 'OPERATOR')

    history = client.get('/api/detection-history').get_json()
    assert [item['id'] for item in history] == [session_ids[owner]]
    assert client.get('/session/' + session_ids[owner]).status_code == 200
    assert client.get('/session/' + session_ids[other]).status_code == 302
    assert client.get('/reports/session/' + session_ids[other] + '/download').status_code == 404

    incidents_page = client.get('/incidents').get_data(as_text=True)
    assert 'INC-' + session_ids[owner] in incidents_page
    assert 'INC-' + session_ids[other] not in incidents_page
    exported = client.get('/api/incidents/export-csv').get_data(as_text=True)
    assert 'INC-' + session_ids[owner] in exported
    assert 'INC-' + session_ids[other] not in exported

    page = client.get('/reports').get_data(as_text=True)
    assert session_ids[owner] in page
    assert session_ids[other] not in page


def test_analytics_dashboard_and_admin_account_view_use_their_declared_scope(isolation_app):
    _, client, user_ids, session_ids = isolation_app
    data_a = get_analytics_metrics('all', user_id=user_ids['scope-a'])
    data_b = get_analytics_metrics('all', user_id=user_ids['scope-b'])
    data_admin = get_analytics_metrics('all', is_admin=True)
    assert data_a['stats']['people_assessed'] == 1
    assert data_b['stats']['people_assessed'] == 1
    assert data_admin['stats']['people_assessed'] == 2
    assert data_a['stats']['total_incidents'] == 1
    assert data_b['stats']['total_incidents'] == 1
    assert data_admin['stats']['total_incidents'] == 2

    _sign_in(client, user_ids['scope-a'], 'OPERATOR')
    assert b'Your account' in client.get('/dashboard').data
    assert client.get('/admin/users/{}/activity'.format(user_ids['scope-b'])).status_code == 403

    _sign_in(client, user_ids['scope-admin'], 'ADMIN')
    assert b'System-wide' in client.get('/dashboard').data
    account_page = client.get('/admin/users/{}/activity'.format(user_ids['scope-a']))
    assert account_page.status_code == 200
    assert session_ids['scope-a'].encode() in account_page.data
    assert session_ids['scope-b'].encode() not in account_page.data


def test_legacy_unowned_detection_rows_are_preserved_but_hidden_from_normal_users(isolation_app):
    _, client, user_ids, _ = isolation_app
    conn = db.get_db()
    conn.execute("INSERT INTO detection_sessions(session_uid,session_type,source_name,status) VALUES('LEGACY-UNOWNED','IMAGE','legacy','COMPLETED')")
    conn.execute("INSERT INTO incidents(incident_uid,violation_type,risk_score,severity) VALUES('INC-LEGACY-UNOWNED','Legacy',10,'LOW')")
    conn.commit()
    conn.close()

    _sign_in(client, user_ids['scope-a'], 'OPERATOR')
    assert client.get('/session/LEGACY-UNOWNED').status_code == 302
    assert 'INC-LEGACY-UNOWNED' not in client.get('/incidents').get_data(as_text=True)

    _sign_in(client, user_ids['scope-admin'], 'ADMIN')
    assert client.get('/session/LEGACY-UNOWNED').status_code == 200
    assert 'INC-LEGACY-UNOWNED' in client.get('/incidents').get_data(as_text=True)


def test_video_session_id_cannot_be_used_by_another_account(isolation_app):
    application, client, user_ids, _ = isolation_app
    video_uid = application.video_service.start_session('owned-test.mp4', user_id=user_ids['scope-a'])
    ok, encoded = cv2.imencode('.jpg', np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    image_b64 = base64.b64encode(encoded.tobytes()).decode('ascii')
    _sign_in(client, user_ids['scope-b'], 'OPERATOR')
    response = client.post('/api/video/analyze_frame', json={
        'session_uid': video_uid, 'image_data': image_b64, 'video_time': 0.1,
    }, headers=_csrf_headers(client))
    assert response.status_code == 404


def test_video_session_completion_is_owner_checked(isolation_app):
    application, client, user_ids, _ = isolation_app
    video_uid = application.video_service.start_session('owned-test.mp4', user_id=user_ids['scope-a'])
    _sign_in(client, user_ids['scope-b'], 'OPERATOR')
    response = client.post('/api/video/complete', json={
        'session_uid': video_uid, 'duration_seconds': 0.1,
    }, headers=_csrf_headers(client))
    assert response.status_code == 404


def test_live_preview_is_transient_and_final_counts_ignore_client_values(isolation_app, monkeypatch):
    application, client, user_ids, _ = isolation_app
    _sign_in(client, user_ids['scope-a'], 'OPERATOR')
    frame = np.zeros((36, 36, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode('.jpg', frame)
    assert ok
    payload = base64.b64encode(encoded.tobytes()).decode('ascii')
    violation_detections = [
        {'class_name': 'Person', 'confidence': 0.95, 'bbox': [50, 30, 150, 220]},
        {'class_name': 'NO-Hardhat', 'confidence': 0.90, 'bbox': [70, 25, 110, 70]},
        {'class_name': 'Mask', 'confidence': 0.90, 'bbox': [70, 75, 110, 100]},
        {'class_name': 'Safety Vest', 'confidence': 0.90, 'bbox': [55, 100, 145, 190]},
    ]
    monkeypatch.setattr(application.pipeline.model_mgr, 'predict', lambda *args, **kwargs: (violation_detections, None))
    conn = db.get_db()
    baseline_incidents = conn.execute('SELECT COUNT(*) FROM incidents WHERE user_id=? AND session_uid IS NOT NULL', (user_ids['scope-a'],)).fetchone()[0]
    conn.close()

    for _ in range(3):
        response = client.post('/api/detect/frame', json={'image_data': payload}, headers=_csrf_headers(client))
        assert response.status_code == 200
    live_uid = response.get_json()['session_uid']
    assert response.get_json()['person_results']
    conn = db.get_db()
    saved_incidents_before_stop = conn.execute('SELECT COUNT(*) FROM incidents WHERE user_id=? AND session_uid IS NOT NULL', (user_ids['scope-a'],)).fetchone()[0]
    conn.close()
    assert saved_incidents_before_stop == baseline_incidents

    monkeypatch.setattr(application, '_telegram_notify', lambda *args, **kwargs: None)
    completed = client.post('/api/session/live_complete', json={
        'duration_seconds': 1.0, 'person_results': [{'person_id': 999, 'risk': 'CRITICAL'}],
    }, headers=_csrf_headers(client))
    assert completed.get_json()['session_uid'] == live_uid
    conn = db.get_db()
    row = conn.execute('SELECT user_id,status,people_count FROM detection_sessions WHERE session_uid=?', (live_uid,)).fetchone()
    assert (row['user_id'], row['status'], row['people_count']) == (user_ids['scope-a'], 'COMPLETED', 1)
    live_incident = conn.execute('SELECT COUNT(*) FROM incidents WHERE user_id=? AND session_uid=?',
                                 (user_ids['scope-a'], live_uid)).fetchone()[0]
    assert live_incident == 1
    assert conn.execute('SELECT COUNT(*) FROM notifications WHERE user_id=? AND incident_id IN (SELECT id FROM incidents WHERE session_uid=?)',
                        (user_ids['scope-a'], live_uid)).fetchone()[0] == 1
    conn.close()
    application.pipeline.reset_tracker(user_id=user_ids['scope-a'])


def test_session_and_overall_reports_generate_with_owner_and_download_authorization(isolation_app):
    application, client, user_ids, session_ids = isolation_app
    _sign_in(client, user_ids['scope-a'], 'OPERATOR')
    session_response = client.get('/reports/session/{}/download'.format(session_ids['scope-a']))
    assert session_response.status_code == 200
    assert session_response.data.startswith(b'%PDF-')
    conn = db.get_db()
    report = conn.execute('SELECT file_path,owner_id FROM generated_reports WHERE session_uid=?',
                          (session_ids['scope-a'],)).fetchone()
    report_path = report['file_path']
    assert report and report['owner_id'] == user_ids['scope-a']
    assert client.get('/static/reports/' + report_path.split('/')[-1]).status_code == 200
    conn.close()

    overall_response = client.get('/reports/overall/download')
    assert overall_response.status_code == 200
    assert overall_response.data.startswith(b'%PDF-')
    conn = db.get_db()
    overall = conn.execute("SELECT file_path,owner_id FROM generated_reports WHERE report_type='OVERALL' AND owner_id=? ORDER BY id DESC LIMIT 1",
                           (user_ids['scope-a'],)).fetchone()
    assert overall and overall['owner_id'] == user_ids['scope-a']
    conn.close()

    _sign_in(client, user_ids['scope-b'], 'OPERATOR')
    assert client.get('/static/reports/' + report_path.split('/')[-1]).status_code == 404