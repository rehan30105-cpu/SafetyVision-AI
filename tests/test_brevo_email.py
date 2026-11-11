"""Brevo request and post-commit access-decision tests with mocked HTTP."""

import os
import tempfile

import pytest

from database import db
from services import brevo_email
from werkzeug.security import generate_password_hash


TEST_CONFIG = {
    'BREVO_API_KEY': 'unit-test-secret-key',
    'MAIL_FROM_EMAIL': 'verified@example.test',
    'MAIL_FROM_NAME': 'SafetyVision Test',
    'ADMIN_EMAIL': 'admin@example.test',
    'PUBLIC_BASE_URL': 'https://safety.example.test',
}


class FakeResponse:
    def __init__(self, status_code=201, body=None):
        self.status_code = status_code
        self.body = {'messageId': '<accepted-id@example.test>'} if body is None else body

    def json(self):
        return self.body


@pytest.mark.parametrize('decision', ['APPROVED', 'REJECTED'])
def test_decision_email_payload_and_acceptance(monkeypatch, decision):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse()

    monkeypatch.setattr(brevo_email.requests, 'post', fake_post)
    result = brevo_email.send_access_decision_email(
        TEST_CONFIG, decision=decision, recipient_email='applicant@example.test',
        recipient_name='Taylor Applicant')

    assert result == {'ok': True, 'message_id': '<accepted-id@example.test>', 'error': None}
    assert captured['url'] == brevo_email.BREVO_SEND_URL
    assert captured['headers']['api-key'] == TEST_CONFIG['BREVO_API_KEY']
    assert captured['timeout'] == 10
    payload = captured['json']
    assert payload['sender'] == {'email': TEST_CONFIG['MAIL_FROM_EMAIL'], 'name': TEST_CONFIG['MAIL_FROM_NAME']}
    assert payload['to'] == [{'email': 'applicant@example.test', 'name': 'Taylor Applicant'}]
    assert payload['replyTo']['email'] == TEST_CONFIG['ADMIN_EMAIL']
    if decision == 'APPROVED':
        assert 'approved' in payload['subject'].lower()
        assert 'password you chose' in payload['textContent']
        assert 'https://safety.example.test/login' in payload['textContent']
    else:
        assert 'not approved' in payload['textContent']
        assert 'https://safety.example.test' in payload['textContent']


def test_legacy_approval_sends_one_time_setup_link(monkeypatch):
    captured = {}
    monkeypatch.setattr(brevo_email.requests, 'post',
                        lambda url, **kwargs: (captured.update(kwargs) or FakeResponse()))
    result = brevo_email.send_access_decision_email(
        TEST_CONFIG, decision='APPROVED', recipient_email='legacy@example.test',
        recipient_name='Legacy Applicant',
        activation_url='https://safety.example.test/demo/activate/one-time-token')
    assert result['ok']
    assert 'expires in 24 hours' in captured['json']['textContent']
    assert 'https://safety.example.test/demo/activate/one-time-token' in captured['json']['textContent']


@pytest.mark.parametrize('response', [FakeResponse(400, {'message': 'contains sensitive provider detail'}),
                                      FakeResponse(500, {'message': 'contains sensitive provider detail'}),
                                      FakeResponse(201, {})])
def test_provider_errors_are_sanitized(monkeypatch, caplog, response):
    monkeypatch.setattr(brevo_email.requests, 'post', lambda *args, **kwargs: response)
    result = brevo_email.send_access_decision_email(
        TEST_CONFIG, decision='REJECTED', recipient_email='applicant@example.test', recipient_name='Taylor')
    assert not result['ok']
    assert TEST_CONFIG['BREVO_API_KEY'] not in str(result)
    assert 'contains sensitive provider detail' not in str(result)
    assert TEST_CONFIG['BREVO_API_KEY'] not in caplog.text


@pytest.mark.parametrize('status', [401, 403])
def test_auth_rejection_identifies_credential_category_without_exposing_key(monkeypatch, caplog, status):
    monkeypatch.setattr(brevo_email.requests, 'post', lambda *args, **kwargs: FakeResponse(status, {'message': 'private provider body'}))
    result = brevo_email.send_access_decision_email(
        TEST_CONFIG, decision='REJECTED', recipient_email='applicant@example.test', recipient_name='Taylor')
    assert not result['ok']
    assert 'API credential' in result['error']
    assert str(status) in result['error']
    assert TEST_CONFIG['BREVO_API_KEY'] not in result['error']
    assert 'private provider body' not in result['error']
    assert TEST_CONFIG['BREVO_API_KEY'] not in caplog.text
    assert 'contains sensitive provider detail' not in caplog.text


def test_request_exception_does_not_leak_secret(monkeypatch, caplog):
    class SecretException(brevo_email.requests.RequestException):
        pass
    def fail(*args, **kwargs):
        raise SecretException('request failed with ' + TEST_CONFIG['BREVO_API_KEY'])
    monkeypatch.setattr(brevo_email.requests, 'post', fail)
    result = brevo_email.send_access_decision_email(
        TEST_CONFIG, decision='APPROVED', recipient_email='applicant@example.test', recipient_name='Taylor')
    assert not result['ok']
    assert TEST_CONFIG['BREVO_API_KEY'] not in str(result)
    assert TEST_CONFIG['BREVO_API_KEY'] not in caplog.text


@pytest.fixture
def safetyvision_client(monkeypatch):
    fd, db_path = tempfile.mkstemp(prefix='safetyvision-brevo-mail-', suffix='.db')
    os.close(fd)
    monkeypatch.setattr(db, 'DB_PATH', db_path)
    import app as application
    application.init_db()
    conn = db.get_db()
    conn.execute(
        'INSERT INTO users (username,email,password_hash,role,full_name,is_active) VALUES (?,?,?,?,?,1)',
        ('brevo_admin', 'brevo-admin@example.test', generate_password_hash('brevo-admin-password'), 'ADMIN', 'Mail Admin'))
    conn.commit()
    admin_id = conn.execute("SELECT id FROM users WHERE username='brevo_admin'").fetchone()[0]
    conn.close()
    yield application, application.app.test_client(), admin_id
    try:
        os.remove(db_path)
    except OSError:
        pass


@pytest.mark.parametrize('decision', ['APPROVED', 'REJECTED'])
def test_admin_review_emails_only_after_commit(safetyvision_client, monkeypatch, decision):
    application, client, admin_id = safetyvision_client
    conn = db.get_db()
    cursor = conn.execute(
        "INSERT INTO demo_requests (name,email,company,role,message,status,password_hash) "
        "VALUES (?,?,?,?,?,'PENDING',?)",
        ('Mail Test Applicant', 'mail-applicant@example.test', 'Test Works', 'Safety lead',
         'Email integration test', None if decision == 'APPROVED' else 'legacy-no-password'))
    request_id = cursor.lastrowid
    conn.commit()
    conn.close()

    application.app.config.update(TEST_CONFIG)
    with client.session_transaction() as session:
        session.update(user_id=admin_id, role='ADMIN', _csrf_token='test-csrf-token')
    captured = {}

    def fake_post(url, **kwargs):
        # The status transition is committed before an external send is attempted.
        check = db.get_db()
        status = check.execute('SELECT status FROM demo_requests WHERE id=?', (request_id,)).fetchone()[0]
        check.close()
        assert status == decision
        captured.update(url=url, **kwargs)
        return FakeResponse()

    monkeypatch.setattr(brevo_email.requests, 'post', fake_post)
    response = client.post('/admin/demo-requests/{}/review'.format(request_id), data={
        'csrf_token': 'test-csrf-token',
        'form_token': application.form_token('demo-review').dumps({'id': request_id}),
        'decision': decision,
        'review_notes': 'Test note',
    })
    assert response.status_code == 302
    assert captured['json']['to'][0]['email'] == 'mail-applicant@example.test'
    conn = db.get_db()
    row = conn.execute('SELECT status,email_status,email_attempts,email_sent_at FROM demo_requests WHERE id=?',
                       (request_id,)).fetchone()
    assert row['status'] == decision
    assert row['email_status'] == 'SENT'
    assert row['email_attempts'] == 1
    assert row['email_sent_at']
    conn.close()


def test_review_failure_is_visible_and_decision_remains_committed(safetyvision_client, monkeypatch):
    application, client, admin_id = safetyvision_client
    conn = db.get_db()
    cursor = conn.execute(
        "INSERT INTO demo_requests (name,email,company,role,message,status,password_hash) "
        "VALUES (?,?,?,?,?,'PENDING',?)",
        ('Failed Mail Applicant', 'failed-mail@example.test', 'Test Works', 'Safety lead', 'Test', 'password-hash'))
    request_id = cursor.lastrowid
    conn.commit()
    conn.close()
    application.app.config.update(TEST_CONFIG)
    with client.session_transaction() as session:
        session.update(user_id=admin_id, _csrf_token='test-csrf-token')

    monkeypatch.setattr(brevo_email.requests, 'post', lambda *a, **k: FakeResponse(401, {'message': 'secret'}))
    response = client.post('/admin/demo-requests/{}/review'.format(request_id), data={
        'csrf_token': 'test-csrf-token',
        'form_token': application.form_token('demo-review').dumps({'id': request_id}),
        'decision': 'APPROVED',
    }, follow_redirects=True)
    assert response.status_code == 200
    assert 'not accepted' in response.get_data(as_text=True)
    conn = db.get_db()
    row = conn.execute('SELECT status,email_status,email_error,email_attempts FROM demo_requests WHERE id=?',
                       (request_id,)).fetchone()
    assert row['status'] == 'APPROVED'
    assert row['email_status'] == 'FAILED'
    assert row['email_attempts'] == 1
    assert 'secret' not in (row['email_error'] or '')
    conn.close()


def test_email_review_get_is_read_only_and_requires_active_admin(safetyvision_client, monkeypatch):
    application, client, admin_id = safetyvision_client
    application.app.config.update(TEST_CONFIG)
    conn = db.get_db()
    cursor = conn.execute("INSERT INTO demo_requests (name,email,company,role,status) VALUES ('Private Person','private@example.test','Private Co','Safety','PENDING')")
    request_id = cursor.lastrowid
    conn.commit()
    conn.close()
    links = []
    monkeypatch.setattr(application, '_send_access_request_email', lambda **kwargs: (links.extend(kwargs.get('action_buttons') or []) or {'ok': True, 'message_id': 'mock', 'error': None}))
    # Mint link-shaped tokens exactly as submission does, storing only nonce hashes.
    token, nonce_hash = application._create_admin_review_token(request_id, 'APPROVED')
    conn = db.get_db()
    conn.execute('UPDATE demo_requests SET email_approval_nonce_hash=? WHERE id=?', (nonce_hash, request_id))
    conn.commit()
    conn.close()
    url = '/admin/demo-requests/email-review/{}?decision=APPROVED'.format(token)
    assert client.get(url).status_code == 302
    assert 'Private Person' not in client.get(url).get_data(as_text=True)
    with client.session_transaction() as sess:
        sess.update(user_id=admin_id, _csrf_token='test-csrf-token')
    view = client.get(url)
    assert view.status_code == 200
    assert 'Private Person' in view.get_data(as_text=True)
    changed = '/admin/demo-requests/email-review/{}?decision=REJECTED'.format(token)
    assert client.get(changed).status_code == 400
    conn = db.get_db()
    assert conn.execute('SELECT status FROM demo_requests WHERE id=?', (request_id,)).fetchone()[0] == 'PENDING'
    conn.close()


def test_email_review_post_requires_csrf_and_nonce_is_single_use(safetyvision_client, monkeypatch):
    application, client, admin_id = safetyvision_client
    application.app.config.update(TEST_CONFIG)
    conn = db.get_db()
    cursor = conn.execute("INSERT INTO demo_requests (name,email,company,role,status,password_hash) VALUES ('Applicant','a@example.test','Co','Safety','PENDING','test-hash')")
    request_id = cursor.lastrowid
    conn.commit()
    conn.close()
    signed, nonce_hash = application._create_admin_review_token(request_id, 'REJECTED')
    payload = application.form_token('admin-email-review').loads(signed)
    conn = db.get_db()
    conn.execute('UPDATE demo_requests SET email_rejection_nonce_hash=? WHERE id=?', (nonce_hash, request_id))
    conn.commit()
    conn.close()
    with client.session_transaction() as sess:
        sess.update(user_id=admin_id, _csrf_token='csrf-ok')
    target = '/admin/demo-requests/{}/review'.format(request_id)
    data = {'form_token': application.form_token('demo-review').dumps({'id': request_id}),
            'email_action_token': signed, 'decision': 'REJECTED'}
    assert client.post(target, data=data).status_code == 400
    # Invalid CSRF is rejected before the review handler; it does not clear auth.
    with client.session_transaction() as sess:
        sess['_csrf_token'] = 'csrf-ok'
    data['csrf_token'] = 'csrf-ok'
    monkeypatch.setattr(brevo_email.requests, 'post', lambda *a, **k: FakeResponse())
    assert client.post(target, data=data).status_code == 302
    assert client.post(target, data=data).status_code == 302
    conn = db.get_db()
    assert conn.execute('SELECT status,email_rejection_nonce_hash FROM demo_requests WHERE id=?', (request_id,)).fetchone()[:] == ('REJECTED', None)
    conn.close()


def test_admin_only_retry_records_result(safetyvision_client, monkeypatch):
    application, client, admin_id = safetyvision_client
    application.app.config.update(TEST_CONFIG)
    conn = db.get_db()
    cursor = conn.execute("INSERT INTO demo_requests (name,email,company,role,status,request_email_status) VALUES ('Applicant','retry@example.test','Co','Safety','PENDING','FAILED')")
    request_id = cursor.lastrowid
    conn.commit()
    conn.close()
    assert client.post('/admin/demo-requests/{}/email-retry'.format(request_id), data={'csrf_token':'x','category':'REQUEST'}).status_code == 400
    with client.session_transaction() as sess:
        sess.update(user_id=admin_id, _csrf_token='csrf-ok')
    monkeypatch.setattr(brevo_email.requests, 'post', lambda *a, **k: FakeResponse())
    response = client.post('/admin/demo-requests/{}/email-retry'.format(request_id), data={'csrf_token':'csrf-ok','category':'REQUEST'})
    assert response.status_code == 302
    conn = db.get_db()
    assert conn.execute('SELECT request_email_status FROM demo_requests WHERE id=?', (request_id,)).fetchone()[0] == 'ACCEPTED'
    conn.close()