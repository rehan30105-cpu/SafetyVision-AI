"""Isolated tests for public demo-request submission and email outcomes."""
import os
import re
import tempfile

import pytest
from werkzeug.security import generate_password_hash

from database import db


@pytest.fixture
def client(monkeypatch):
    fd, db_path = tempfile.mkstemp(prefix="safetyvision-request-submit-", suffix=".db")
    os.close(fd)
    monkeypatch.setattr(db, "DB_PATH", db_path)
    import app as application
    application.init_db()
    conn = db.get_db()
    conn.execute(
        "INSERT INTO users (username,email,password_hash,role,full_name,is_active) VALUES (?,?,?,?,?,1)",
        ("test_admin", "admin@example.test", generate_password_hash("admin-test-password"), "ADMIN", "Test Admin"),
    )
    conn.commit()
    conn.close()
    application.app.config["PUBLIC_BASE_URL"] = "https://safetyvision.example.test"
    yield application, application.app.test_client()
    try:
        os.remove(db_path)
    except OSError:
        pass


def _submit(application, client, email):
    page = client.get("/")
    token = re.search(r'name="form_token" value="([^"]+)', page.get_data(as_text=True)).group(1)
    with client.session_transaction() as session:
        csrf = session["_csrf_token"]
    return client.post("/demo/request", data={
        "csrf_token": csrf,
        "form_token": token,
        "name": "Test Applicant",
        "email": email,
        "company": "Example Works",
        "role": "Safety lead",
        "message": "Automated test",
        "password": "a-long-test-password",
        "confirm_password": "a-long-test-password",
    })


def test_duplicate_pending_request_returns_status_without_resending(client, monkeypatch):
    application, test_client = client
    conn = db.get_db()
    conn.execute(
        "INSERT INTO demo_requests (name,email,company,role,status) VALUES (?,?,?,?, 'PENDING')",
        ("Existing Applicant", "duplicate@example.test", "Example", "Safety lead"),
    )
    conn.commit()
    conn.close()
    calls = []
    monkeypatch.setattr(application, "_send_access_request_email", lambda **kwargs: calls.append(kwargs))

    response = _submit(application, test_client, "DUPLICATE@example.test")

    assert response.status_code == 409
    assert "PENDING" in response.get_data(as_text=True)
    assert "do not need to submit again" in response.get_data(as_text=True)
    assert calls == []


def test_demo_requests_admin_queue_renders_database_counts_and_search(client):
    application, test_client = client
    conn = db.get_db()
    conn.execute("INSERT INTO demo_requests (name,email,company,role,status) VALUES ('Queued Person','queued@example.test','Example','Safety','PENDING')")
    conn.execute("INSERT INTO demo_requests (name,email,company,role,status) VALUES ('Approved Person','approved@example.test','Example','Safety','APPROVED')")
    conn.commit()
    admin_id = conn.execute("SELECT id FROM users WHERE username='test_admin'").fetchone()[0]
    conn.close()
    with test_client.session_transaction() as session:
        session.update(user_id=admin_id, role='ADMIN')
    response = test_client.get('/admin/demo-requests')
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'Queued Person' in html and 'Approved Person' in html
    assert 'Pending' in html and 'Approved' in html and 'Total requests' in html
    filtered = test_client.get('/admin/demo-requests?q=Queued&status=PENDING').get_data(as_text=True)
    assert 'Queued Person' in filtered and 'Approved Person' not in filtered


def test_new_request_commits_before_applicant_and_admin_emails(client, monkeypatch):
    application, test_client = client
    calls = []
    monkeypatch.setattr(application.telegram_approval, 'send_new_request', lambda *args: {'ok': True})

    def mock_email(**kwargs):
        conn = db.get_db()
        saved = conn.execute("SELECT status FROM demo_requests WHERE email=? COLLATE NOCASE",
                             ("new@example.test",)).fetchone()
        conn.close()
        assert saved is not None and saved["status"] == "PENDING"
        calls.append(kwargs)
        return {"ok": True, "message_id": "mock-accepted", "error": None}

    monkeypatch.setattr(application, "_send_access_request_email", mock_email)
    response = _submit(application, test_client, "new@example.test")

    assert response.status_code == 201
    assert "accepted by Brevo" in response.get_data(as_text=True)
    assert len(calls) == 2
    assert calls[0]["recipient_email"] == "new@example.test"
    assert calls[0]["subject"] == "We received your SafetyVision AI access request"
    assert calls[1]["recipient_email"] == application.app.config["ADMIN_EMAIL"]
    assert calls[1]["subject"] == "New SafetyVision AI access request"
    conn = db.get_db()
    outcome = conn.execute(
        "SELECT status,request_email_status,admin_email_status FROM demo_requests WHERE email=?",
        ("new@example.test",),
    ).fetchone()
    conn.close()
    assert tuple(outcome) == ("PENDING", "ACCEPTED", "ACCEPTED")


def test_email_failures_keep_request_and_record_failure(client, monkeypatch, caplog):
    application, test_client = client
    caplog.set_level("INFO", logger="safetyvision.access")
    calls = []
    monkeypatch.setattr(application.telegram_approval, 'send_new_request', lambda *args: {'ok': True})

    def failed_email(**kwargs):
        calls.append(kwargs)
        return {"ok": False, "message_id": None, "error": "Email provider rejected the message (HTTP 503)."}

    monkeypatch.setattr(application, "_send_access_request_email", failed_email)
    response = _submit(application, test_client, "failed@example.test")

    assert response.status_code == 201
    assert "confirmation email was not accepted" in response.get_data(as_text=True)
    assert len(calls) == 2
    conn = db.get_db()
    outcome = conn.execute(
        "SELECT status,request_email_status,admin_email_status FROM demo_requests WHERE email=?",
        ("failed@example.test",),
    ).fetchone()
    conn.close()
    assert tuple(outcome) == ("PENDING", "FAILED", "FAILED")
    assert "HTTP 503" in caplog.text
    assert "failed@example.test" not in caplog.text


def test_provider_is_not_called_when_public_review_url_missing(client, monkeypatch):
    application, test_client = client
    application.app.config["PUBLIC_BASE_URL"] = ""
    calls = []
    monkeypatch.setattr(application.telegram_approval, 'send_new_request', lambda *args: {'ok': True})
    monkeypatch.setattr(application, "_send_access_request_email", lambda **kwargs: calls.append(kwargs) or {
        "ok": True, "message_id": "mock-accepted", "error": None,
    })

    response = _submit(application, test_client, "no-url@example.test")

    assert response.status_code == 201
    assert len(calls) == 1
    assert calls[0]["recipient_email"] == "no-url@example.test"
    conn = db.get_db()
    outcome = conn.execute(
        "SELECT status,request_email_status,admin_email_status FROM demo_requests WHERE email=?",
        ("no-url@example.test",),
    ).fetchone()
    conn.close()
    assert tuple(outcome) == ("PENDING", "ACCEPTED", "FAILED")