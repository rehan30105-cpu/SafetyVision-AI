"""Offline tests for private Telegram demo approval and webhook callbacks."""
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pytest
from werkzeug.security import generate_password_hash

from database import db
from services.telegram_approval import TelegramDemoApprovalService


@pytest.fixture
def approval_client(monkeypatch):
    fd, db_path = tempfile.mkstemp(prefix='safetyvision-telegram-approval-', suffix='.db')
    os.close(fd)
    monkeypatch.setattr(db, 'DB_PATH', db_path)
    import app as application
    application.init_db()
    conn = db.get_db()
    cursor = conn.execute(
        'INSERT INTO users (username,email,password_hash,role,full_name,is_active) VALUES (?,?,?,?,?,1)',
        ('approval-admin', 'approval-admin@example.test', generate_password_hash('admin-test-password'), 'ADMIN', 'Approval Admin'))
    admin_id = cursor.lastrowid
    conn.commit()
    conn.close()
    application.app.config.update({
        'PUBLIC_BASE_URL': 'https://safetyvision.example.test',
        'TELEGRAM_BOT_TOKEN': 'TESTTOKEN',
        'TELEGRAM_APPROVAL_ENABLED': True,
        'TELEGRAM_APPROVAL_CHAT_ID': '70001',
        'TELEGRAM_APPROVAL_ADMIN_USER_ID': '80001',
        'TELEGRAM_APPROVAL_WEBHOOK_SECRET': 'webhook-secret-for-tests',
        'BREVO_API_KEY': '', 'MAIL_FROM_EMAIL': '', 'MAIL_FROM_NAME': '', 'ADMIN_EMAIL': '',
    })
    application.telegram_approval = application.TelegramDemoApprovalService(application.app.config)
    yield application, application.app.test_client(), admin_id
    try:
        os.remove(db_path)
    except OSError:
        pass


def _insert_request():
    conn = db.get_db()
    cursor = conn.execute(
        "INSERT INTO demo_requests (name,email,company,role,message,status,password_hash) VALUES (?,?,?,?,?,'PENDING',?)",
        ('Taylor Applicant', 'taylor@example.test', 'Example Works', 'Safety lead', 'Test request', generate_password_hash('long-test-password')))
    request_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return request_id


def _callback(application, client, data, *, sender_id=80001, chat_id=70001, chat_type='private', callback_id='callback-1'):
    payload = {'callback_query': {
        'id': callback_id,
        'from': {'id': sender_id},
        'message': {'message_id': 91, 'chat': {'id': chat_id, 'type': chat_type}},
        'data': data,
    }}
    return client.post('/telegram/demo-approval/webhook', json=payload,
                       headers={'X-Telegram-Bot-Api-Secret-Token': 'webhook-secret-for-tests'})


def test_new_request_notification_is_after_commit_private_and_text_only(approval_client, monkeypatch):
    application, client, _ = approval_client
    calls = []

    def fake_post(method, data):
        if method == 'getChat':
            return {'ok': True, 'result': {'type': 'private'}}
        conn = db.get_db()
        saved = conn.execute("SELECT status FROM demo_requests WHERE email='new-telegram@example.test'").fetchone()
        conn.close()
        assert saved and saved['status'] == 'PENDING'
        calls.append((method, data))
        return {'ok': True, 'result': {'message_id': 12}}

    monkeypatch.setattr(application.telegram_approval, '_post', fake_post)
    monkeypatch.setattr(application, '_send_access_request_email', lambda **kwargs: {'ok': True, 'message_id': 'mail', 'error': None})
    page = client.get('/')
    token = re.search(r'name="form_token" value="([^"]+)', page.get_data(as_text=True)).group(1)
    with client.session_transaction() as session:
        csrf = session['_csrf_token']
    response = client.post('/demo/request', data={
        'csrf_token': csrf, 'form_token': token, 'name': 'Taylor Applicant', 'email': 'new-telegram@example.test',
        'company': 'Example Works', 'role': 'Safety lead', 'message': 'Test',
        'password': 'a-long-test-password', 'confirm_password': 'a-long-test-password',
    })
    assert response.status_code == 201
    assert len(calls) == 1 and calls[0][0] == 'sendMessage'
    assert 'NEW DEMO REQUEST' in calls[0][1]['text']
    assert 'Request ID: 1' in calls[0][1]['text']
    assert [button['text'] for button in calls[0][1]['reply_markup']['inline_keyboard'][0]] == ['✅ Approve', '❌ Reject']
    assert 'sendDocument' not in [method for method, _ in calls]


def test_labeled_diagnostic_sends_only_plain_text_to_verified_private_chat(approval_client, monkeypatch):
    application, _, _ = approval_client
    calls = []

    def fake_post(method, data):
        calls.append((method, data))
        if method == 'getChat':
            return {'ok': True, 'result': {'type': 'private'}}
        return {'ok': True, 'result': {'message_id': 123}}

    monkeypatch.setattr(application.telegram_approval, '_post', fake_post)
    result = application.telegram_approval.send_diagnostic_test_message()
    assert result['ok'] is True
    assert calls == [
        ('getChat', {'chat_id': '70001'}),
        ('sendMessage', {'chat_id': '70001', 'text': 'SafetyVision AI demo approval test'}),
    ]


def test_telegram_http_error_returns_status_and_sanitized_description(monkeypatch, caplog):
    import services.telegram_approval as service_module

    class FakeResponse:
        status_code = 401

        @staticmethod
        def json():
            return {'ok': False, 'description': 'Unauthorized token-should-not-leak'}

    monkeypatch.setattr(service_module.requests, 'post', lambda *args, **kwargs: FakeResponse())
    service = TelegramDemoApprovalService({
        'TELEGRAM_BOT_TOKEN': 'token-should-not-leak',
        'TELEGRAM_APPROVAL_CHAT_ID': '70001',
        'TELEGRAM_APPROVAL_ADMIN_USER_ID': '80001',
        'TELEGRAM_APPROVAL_WEBHOOK_SECRET': 'webhook-secret',
        'TELEGRAM_APPROVAL_ENABLED': True,
    })
    result = service._post('getMe', {})
    assert result['ok'] is False
    assert result['http_status'] == 401
    assert 'token-should-not-leak' not in caplog.text
    assert 'token-should-not-leak' not in str(result)


def test_telegram_failure_keeps_saved_request(approval_client, monkeypatch):
    application, client, _ = approval_client
    monkeypatch.setattr(application.telegram_approval, '_post', lambda *args, **kwargs: {'ok': False, 'error': 'safe'})
    monkeypatch.setattr(application, '_send_access_request_email', lambda **kwargs: {'ok': True, 'message_id': 'mail', 'error': None})
    page = client.get('/')
    token = re.search(r'name="form_token" value="([^"]+)', page.get_data(as_text=True)).group(1)
    with client.session_transaction() as session:
        csrf = session['_csrf_token']
    response = client.post('/demo/request', data={
        'csrf_token': csrf, 'form_token': token, 'name': 'Taylor Applicant', 'email': 'saved@example.test',
        'company': 'Example Works', 'role': 'Safety lead', 'password': 'a-long-test-password',
        'confirm_password': 'a-long-test-password',
    })
    assert response.status_code == 201
    conn = db.get_db()
    assert conn.execute('SELECT status FROM demo_requests WHERE email=?', ('saved@example.test',)).fetchone()[0] == 'PENDING'
    conn.close()


def test_telegram_notification_failure_is_retryable_once_and_not_repeated(approval_client, monkeypatch):
    application, _, _ = approval_client
    request_id = _insert_request()
    attempts = []

    def send(row, callback_data):
        attempts.append(row['id'])
        return {'ok': len(attempts) > 1, 'result': {'message_id': 42} if len(attempts) > 1 else None,
                'error': None if len(attempts) > 1 else 'Telegram rejected the request.'}

    monkeypatch.setattr(application.telegram_approval, 'send_new_request', send)
    failed = application._send_telegram_demo_notification(request_id)
    assert failed['ok'] is False
    conn = db.get_db()
    assert conn.execute('SELECT telegram_approval_status FROM demo_requests WHERE id=?', (request_id,)).fetchone()[0] == 'FAILED'
    conn.close()

    retried = application._send_telegram_demo_notification(request_id, retry_failed=True)
    assert retried['ok'] is True
    already_sent = application._send_telegram_demo_notification(request_id, retry_failed=True)
    assert already_sent['skipped'] is True
    assert attempts == [request_id, request_id]
    conn = db.get_db()
    status, message_id = conn.execute(
        'SELECT telegram_approval_status,telegram_approval_message_id FROM demo_requests WHERE id=?', (request_id,)).fetchone()
    conn.close()
    assert (status, message_id) == ('SENT', 42)


@pytest.mark.parametrize(('action', 'decision'), [('approve', 'APPROVED'), ('reject', 'REJECTED')])
def test_authorized_approval_and_rejection_commit_then_email_and_edit_message(approval_client, monkeypatch, action, decision):
    application, client, _ = approval_client
    request_id = _insert_request()
    callback_calls = []

    def capture_email(rid, requested_decision, row, token):
        conn = db.get_db()
        committed_status = conn.execute('SELECT status FROM demo_requests WHERE id=?', (rid,)).fetchone()[0]
        conn.close()
        callback_calls.append((rid, requested_decision, row['status'], committed_status))
        return {'ok': True}

    monkeypatch.setattr(application, '_notify_decision_applicant', capture_email)
    monkeypatch.setattr(application.telegram_approval, 'edit_decision_message',
        lambda *args: callback_calls.append(('edit',) + args) or {'ok': True})
    monkeypatch.setattr(application.telegram_approval, 'answer_callback',
        lambda *args: callback_calls.append(('answer',) + args) or {'ok': True})
    response = _callback(application, client, application._telegram_demo_callback_data(action, request_id))
    assert response.status_code == 200
    conn = db.get_db()
    status = conn.execute('SELECT status FROM demo_requests WHERE id=?', (request_id,)).fetchone()[0]
    account_count = conn.execute('SELECT COUNT(*) FROM users WHERE email=?', ('taylor@example.test',)).fetchone()[0]
    conn.close()
    assert status == decision
    assert account_count == (1 if decision == 'APPROVED' else 0)
    assert callback_calls[0] == ('answer', 'callback-1', 'Request #{} {}'.format(request_id, decision.lower()), False)
    assert callback_calls[1] == (request_id, decision, 'PENDING', decision)
    assert callback_calls[2][0] == 'edit'


def test_unauthorized_callback_cannot_change_request_or_send_email(approval_client, monkeypatch):
    application, client, _ = approval_client
    request_id = _insert_request()
    calls = []
    monkeypatch.setattr(application, '_notify_decision_applicant', lambda *args: calls.append(args))
    monkeypatch.setattr(application.telegram_approval, 'answer_callback', lambda *args: calls.append(args) or {'ok': True})
    response = _callback(application, client, application._telegram_demo_callback_data('approve', request_id), sender_id=99999)
    assert response.status_code == 200
    conn = db.get_db()
    assert conn.execute('SELECT status FROM demo_requests WHERE id=?', (request_id,)).fetchone()[0] == 'PENDING'
    conn.close()
    assert len(calls) == 1 and 'not authorized' in calls[0][1].lower()


def test_duplicate_callback_preserves_first_decision_and_reports_actual_status(approval_client, monkeypatch):
    application, client, _ = approval_client
    request_id = _insert_request()
    edits, answers = [], []
    monkeypatch.setattr(application, '_notify_decision_applicant', lambda *args: {
        'ok': True, 'message_id': 'mock-accepted', 'error': None,
    })
    monkeypatch.setattr(application.telegram_approval, 'edit_decision_message', lambda *args: edits.append(args) or {'ok': True})
    monkeypatch.setattr(application.telegram_approval, 'answer_callback', lambda *args: answers.append(args) or {'ok': True})
    approve = application._telegram_demo_callback_data('approve', request_id)
    reject = application._telegram_demo_callback_data('reject', request_id)
    _callback(application, client, approve, callback_id='first')
    _callback(application, client, reject, callback_id='duplicate')
    conn = db.get_db()
    assert conn.execute('SELECT status FROM demo_requests WHERE id=?', (request_id,)).fetchone()[0] == 'APPROVED'
    conn.close()
    assert len(edits) == 2 and edits[-1][-1] == 'APPROVED'
    assert 'already approved' in answers[-1][1].lower()


def test_concurrent_opposite_decisions_only_finalize_once(approval_client):
    application, _, _ = approval_client
    request_id = _insert_request()
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(
            lambda decision: application._apply_demo_request_decision(
                request_id, decision, reviewed_by=None, review_notes='Concurrent test.'),
            ('APPROVED', 'REJECTED')))
    assert sum(1 for outcome in outcomes if outcome['ok']) == 1
    assert sum(1 for outcome in outcomes if outcome.get('reason') == 'already_reviewed') == 1
    conn = db.get_db()
    status = conn.execute('SELECT status FROM demo_requests WHERE id=?', (request_id,)).fetchone()[0]
    conn.close()
    assert status in ('APPROVED', 'REJECTED')


def test_missing_request_and_invalid_or_missing_webhook_auth(approval_client, monkeypatch):
    application, client, _ = approval_client
    answers = []
    monkeypatch.setattr(application.telegram_approval, 'answer_callback', lambda *args: answers.append(args) or {'ok': True})
    missing_data = application._telegram_demo_callback_data('approve', 999999)
    assert _callback(application, client, missing_data).status_code == 200
    assert 'no longer exists' in answers[-1][1]
    assert client.post('/telegram/demo-approval/webhook', json={}).status_code == 403
    assert client.post('/telegram/demo-approval/webhook', json={}, headers={
        'X-Telegram-Bot-Api-Secret-Token': 'wrong'}).status_code == 403
    assert application._parse_telegram_demo_callback_data('approve:1:000000000000000000000000') is None


def test_email_failure_does_not_reverse_decision_and_is_retry_visible(approval_client, monkeypatch):
    application, client, _ = approval_client
    request_id = _insert_request()
    monkeypatch.setattr(application, 'send_access_decision_email',
        lambda *args, **kwargs: {'ok': False, 'message_id': None, 'error': 'Email provider request failed.'})
    monkeypatch.setattr(application.telegram_approval, 'edit_decision_message', lambda *args: {'ok': True})
    monkeypatch.setattr(application.telegram_approval, 'answer_callback', lambda *args: {'ok': True})
    response = _callback(application, client, application._telegram_demo_callback_data('approve', request_id))
    assert response.status_code == 200
    conn = db.get_db()
    row = conn.execute('SELECT status,decision_email_status,email_status FROM demo_requests WHERE id=?', (request_id,)).fetchone()
    conn.close()
    assert tuple(row) == ('APPROVED', 'FAILED', 'FAILED')


@pytest.mark.parametrize(('decision', 'expected'), [('APPROVED', 'APPROVED'), ('REJECTED', 'REJECTED')])
def test_website_review_action_persists_decision_and_renders_queue(approval_client, monkeypatch, decision, expected):
    application, client, admin_id = approval_client
    request_id = _insert_request()
    monkeypatch.setattr(application, '_notify_decision_applicant', lambda *args: {'ok': True, 'message_id': 'mock-accepted', 'error': None})
    monkeypatch.setattr(application.telegram_approval, 'edit_decision_message', lambda *args: {'ok': True})
    with client.session_transaction() as sess:
        sess.update(user_id=admin_id, role='ADMIN', _csrf_token='queue-csrf')
    page = client.get('/admin/demo-requests')
    assert page.status_code == 200
    assert b'Demo requests' in page.data
    assert b'Pending' in page.data and b'Approve' in page.data and b'Reject' in page.data
    response = client.post('/admin/demo-requests/{}/review'.format(request_id), data={
        'csrf_token': 'queue-csrf',
        'form_token': application.form_token('demo-review').dumps({'id': request_id}),
        'decision': decision,
        'review_notes': 'Isolated website action test',
    })
    assert response.status_code == 302
    conn = db.get_db()
    row = conn.execute('SELECT status FROM demo_requests WHERE id=?', (request_id,)).fetchone()
    accounts = conn.execute('SELECT COUNT(*) FROM users WHERE email=? COLLATE NOCASE', ('taylor@example.test',)).fetchone()[0]
    conn.close()
    assert row['status'] == expected
    assert accounts == (1 if expected == 'APPROVED' else 0)


def test_telegram_report_notifier_unchanged_and_sms_module_removed(approval_client):
    application, _, _ = approval_client
    import services.telegram_service as reports
    assert callable(reports.notify_session_completion)
    assert callable(reports.get_notifier().send_document)
    assert not os.path.exists(os.path.join(os.path.dirname(application.__file__), 'services', 'fast2sms.py'))
    assert not any('fast2sms' in name.lower() for name in __import__('sys').modules)


def test_database_failure_during_callback(approval_client, monkeypatch):
    import sqlite3
    application, client, _ = approval_client
    request_id = _insert_request()
    answers = []
    monkeypatch.setattr(application.telegram_approval, 'answer_callback', lambda *args: answers.append(args) or {'ok': True})

    def fail_decision(*args, **kwargs):
        raise sqlite3.OperationalError('Simulated DB disk I/O error')

    monkeypatch.setattr(application, '_apply_demo_request_decision', fail_decision)
    response = _callback(application, client, application._telegram_demo_callback_data('approve', request_id))
    assert response.status_code == 200
    assert len(answers) == 1
    assert 'could not be reviewed' in answers[0][1].lower()


def test_telegram_api_timeout_handled_gracefully(approval_client, monkeypatch):
    application, client, _ = approval_client
    request_id = _insert_request()
    import requests

    def timeout_post(*args, **kwargs):
        raise requests.exceptions.Timeout('Connection timed out')

    monkeypatch.setattr(application.telegram_approval, '_post', timeout_post)
    response = _callback(application, client, application._telegram_demo_callback_data('approve', request_id))
    assert response.status_code == 200
    conn = db.get_db()
    status = conn.execute('SELECT status FROM demo_requests WHERE id=?', (request_id,)).fetchone()[0]
    conn.close()
    assert status == 'APPROVED'


def test_telegram_message_edit_failure_preserves_decision(approval_client, monkeypatch):
    application, client, _ = approval_client
    request_id = _insert_request()
    answers = []
    monkeypatch.setattr(application.telegram_approval, 'answer_callback', lambda *args: answers.append(args) or {'ok': True})
    monkeypatch.setattr(application.telegram_approval, 'edit_decision_message', lambda *args: {'ok': False, 'error': 'Telegram 502 Bad Gateway'})
    response = _callback(application, client, application._telegram_demo_callback_data('approve', request_id))
    assert response.status_code == 200
    conn = db.get_db()
    status = conn.execute('SELECT status FROM demo_requests WHERE id=?', (request_id,)).fetchone()[0]
    conn.close()
    assert status == 'APPROVED'
