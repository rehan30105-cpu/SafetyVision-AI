"""Offline history and owner-scoped session detail route tests."""
import os
import tempfile

import pytest
from werkzeug.security import generate_password_hash

from database import db


@pytest.fixture
def history_app(monkeypatch):
    fd, path = tempfile.mkstemp(prefix='safetyvision-history-', suffix='.db')
    os.close(fd)
    os.remove(path)
    monkeypatch.setattr(db, 'DB_PATH', path)
    import app as application
    application.init_db()
    conn = db.get_db()
    users = {}
    for username, role in [('history-admin', 'ADMIN'), ('history-a', 'OPERATOR'), ('history-b', 'OPERATOR')]:
        users[username] = conn.execute(
            'INSERT INTO users (username,email,password_hash,role,full_name,is_active) VALUES (?,?,?,?,?,1)',
            (username, username + '@example.test', generate_password_hash('long-password-test'), role, username)).lastrowid
    conn.commit()
    conn.close()
    sessions = {}
    for username in ('history-a', 'history-b'):
        sessions[username], _ = application.session_manager.create_session('IMAGE', username, user_id=users[username])
        application.session_manager.complete_session(
            sessions[username], [{'person_id': 1, 'compliance': 100.0, 'hardhat': 'PRESENT', 'mask': 'PRESENT',
                                  'vest': 'PRESENT', 'risk': 'LOW', 'risk_score': 0.0}],
            'static/evidence/{}.jpg'.format(username), user_id=users[username])
    yield application, application.app.test_client(), users, sessions
    try:
        os.remove(path)
    except OSError:
        pass


def _sign_in(client, user_id, role):
    with client.session_transaction() as sess:
        sess.update(user_id=user_id, role=role)


def test_history_lists_only_owned_sessions_and_rejects_tampered_session_id(history_app):
    _, client, users, sessions = history_app
    _sign_in(client, users['history-a'], 'OPERATOR')
    response = client.get('/history')
    assert response.status_code == 200
    assert sessions['history-a'].encode() in response.data
    assert sessions['history-b'].encode() not in response.data
    assert client.get('/session/' + sessions['history-a']).status_code == 200
    assert client.get('/session/' + sessions['history-b']).status_code == 302
    assert client.get('/reports/session/' + sessions['history-b'] + '/download').status_code == 404


def test_admin_history_includes_owned_and_legacy_unowned_rows(history_app):
    _, client, users, sessions = history_app
    conn = db.get_db()
    conn.execute("INSERT INTO detection_sessions(session_uid,session_type,source_name,status) VALUES('LEGACY-SESSION','IMAGE','legacy','COMPLETED')")
    conn.commit()
    conn.close()
    _sign_in(client, users['history-admin'], 'ADMIN')
    response = client.get('/history')
    assert response.status_code == 200
    for session_uid in (*sessions.values(), 'LEGACY-SESSION'):
        assert session_uid.encode() in response.data