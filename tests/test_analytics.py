"""Analytics range and account-scope tests using Flask's offline test client."""
import os
import tempfile

import pytest
from werkzeug.security import generate_password_hash

from database import db


@pytest.fixture
def analytics_app(monkeypatch):
    fd, path = tempfile.mkstemp(prefix='safetyvision-analytics-', suffix='.db')
    os.close(fd)
    os.remove(path)
    monkeypatch.setattr(db, 'DB_PATH', path)
    import app as application
    application.init_db()
    conn = db.get_db()
    users = {}
    for username, role in [('analytics-admin', 'ADMIN'), ('analytics-a', 'OPERATOR'), ('analytics-b', 'OPERATOR')]:
        users[username] = conn.execute(
            'INSERT INTO users (username,email,password_hash,role,full_name,is_active) VALUES (?,?,?,?,?,1)',
            (username, username + '@example.test', generate_password_hash('long-password-test'), role, username)).lastrowid
    conn.commit()
    conn.close()
    for username in ('analytics-a', 'analytics-b'):
        uid, _ = application.session_manager.create_session('IMAGE', username, user_id=users[username])
        application.session_manager.complete_session(
            uid, [{'person_id': 1, 'compliance': 100.0, 'hardhat': 'PRESENT', 'mask': 'PRESENT',
                  'vest': 'PRESENT', 'risk': 'LOW', 'risk_score': 0.0}],
            'static/evidence/{}.jpg'.format(username), user_id=users[username])
    yield application, application.app.test_client(), users
    try:
        os.remove(path)
    except OSError:
        pass


def _sign_in(client, user_id, role):
    with client.session_transaction() as sess:
        sess.update(user_id=user_id, role=role)


@pytest.mark.parametrize('time_range', ['today', 'yesterday', '7days', '30days', 'all'])
def test_analytics_ranges_render(time_range, analytics_app):
    _, client, users = analytics_app
    _sign_in(client, users['analytics-a'], 'OPERATOR')
    response = client.get('/analytics?range=' + time_range)
    assert response.status_code == 200
    assert b'Analytics' in response.data


def test_analytics_is_owner_scoped_and_admin_global(analytics_app):
    _, client, users = analytics_app
    _sign_in(client, users['analytics-a'], 'OPERATOR')
    data_a = client.get('/analytics?range=all').get_data(as_text=True)
    _sign_in(client, users['analytics-b'], 'OPERATOR')
    data_b = client.get('/analytics?range=all').get_data(as_text=True)
    assert 'analytics-a' not in data_b
    assert 'analytics-b' not in data_a
    _sign_in(client, users['analytics-admin'], 'ADMIN')
    assert client.get('/dashboard').status_code == 200
    assert client.get('/analytics?range=all').status_code == 200