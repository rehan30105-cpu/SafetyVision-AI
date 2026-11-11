"""Offline authentication and dashboard route checks using an isolated DB."""
import os
import secrets
import tempfile

import pytest
from werkzeug.security import generate_password_hash

from database import db


@pytest.fixture
def login_app(monkeypatch):
    fd, path = tempfile.mkstemp(prefix='safetyvision-login-', suffix='.db')
    os.close(fd)
    os.remove(path)
    monkeypatch.setattr(db, 'DB_PATH', path)
    import app as application
    application.init_db()
    conn = db.get_db()
    users = {}
    for username, role in [('dashboard-admin', 'ADMIN'), ('dashboard-user', 'OPERATOR')]:
        users[username] = conn.execute(
            'INSERT INTO users (username,email,password_hash,role,full_name,is_active) VALUES (?,?,?,?,?,1)',
            (username, username + '@example.test', generate_password_hash('long-password-test'), role, username)).lastrowid
    conn.commit()
    conn.close()
    yield application, application.app.test_client(), users
    try:
        os.remove(path)
    except OSError:
        pass


def _auth(client, user_id, stale_role):
    with client.session_transaction() as sess:
        sess.update(user_id=user_id, role=stale_role, _csrf_token=secrets.token_urlsafe(24))


def test_unauthenticated_dashboard_redirects_to_login(login_app):
    _, client, _ = login_app
    response = client.get('/dashboard')
    assert response.status_code == 302
    assert '/login' in response.headers['Location']


def test_admin_dashboard_direct_and_sidebar_route_to_dashboard(login_app):
    _, client, users = login_app
    _auth(client, users['dashboard-admin'], 'OPERATOR')
    response = client.get('/dashboard')
    assert response.status_code == 200
    assert b'System-wide' in response.data
    assert b'Safety Overview Dashboard' in response.data
    assert b'href="/dashboard"' in response.data


def test_normal_dashboard_is_account_scoped_and_admin_pages_are_forbidden(login_app):
    _, client, users = login_app
    _auth(client, users['dashboard-user'], 'ADMIN')
    response = client.get('/dashboard')
    assert response.status_code == 200
    assert b'Your account' in response.data
    assert client.get('/admin/demo-requests').status_code == 403
    assert client.get('/users').status_code == 403