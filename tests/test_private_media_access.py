"""Private evidence/report URL authorization using isolated SQLite and files."""

import os
import tempfile

import pytest
from werkzeug.security import generate_password_hash

from database import db


@pytest.fixture
def media_client(monkeypatch):
    db_fd, db_path = tempfile.mkstemp(prefix="safetyvision-private-media-", suffix=".db")
    os.close(db_fd)
    os.remove(db_path)
    monkeypatch.setattr(db, "DB_PATH", db_path)

    import app as application

    application.init_db()
    conn = db.get_db()
    cursor = conn.execute(
        "INSERT INTO users (username,email,password_hash,role,full_name,is_active) "
        "VALUES (?,?,?,?,?,1)",
        ("media_test_user", "media@example.test", generate_password_hash("media-test-password"), "OPERATOR", "Media Test"),
    )
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()

    media_root = tempfile.mkdtemp(prefix="safetyvision-private-files-")
    static_root = os.path.join(media_root, "static")
    for folder in ("uploads", "evidence", "reports", "assets"):
        os.makedirs(os.path.join(static_root, folder))
    fixtures = {
        ("uploads", "analysis.jpg"): b"private fixture: upload image",
        ("evidence", "incident.jpg"): b"private fixture: incident evidence",
        ("reports", "session.pdf"): b"private fixture: session report",
        ("assets", "public-test.txt"): b"public stylesheet or marketing asset",
    }
    for (folder, filename), content in fixtures.items():
        with open(os.path.join(static_root, folder, filename), "wb") as media:
            media.write(content)

    conn = db.get_db()
    conn.execute("INSERT INTO detection_sessions (session_uid,session_type,user_id,evidence_image,status) VALUES ('MEDIA-OWNED','IMAGE',?,'static/uploads/analysis.jpg','COMPLETED')", (user_id,))
    conn.execute("INSERT INTO incidents (incident_uid,violation_type,risk_score,severity,user_id,evidence_image) VALUES ('MEDIA-INCIDENT','Missing Hardhat',50,'HIGH',?,'static/evidence/incident.jpg')", (user_id,))
    conn.execute("INSERT INTO generated_reports (owner_id,file_path,report_type) VALUES (?, 'static/reports/session.pdf','SESSION')", (user_id,))
    conn.commit()
    conn.close()

    monkeypatch.setattr(application, "UPLOAD_FOLDER", os.path.join(static_root, "uploads"))
    monkeypatch.setattr(application, "EVIDENCE_FOLDER", os.path.join(static_root, "evidence"))
    monkeypatch.setattr(application, "REPORTS_FOLDER", os.path.join(static_root, "reports"))
    monkeypatch.setattr(application.app, "static_folder", static_root)
    application.app.config["TESTING"] = True

    yield application, application.app.test_client(), user_id, media_root

    import shutil
    shutil.rmtree(media_root, ignore_errors=True)
    try:
        os.remove(db_path)
    except OSError:
        pass


def test_saved_upload_evidence_and_pdf_require_active_login(media_client):
    application, client, user_id, media_root = media_client
    private_urls = {
        "/static/uploads/analysis.jpg": b"private fixture: upload image",
        "/static/evidence/incident.jpg": b"private fixture: incident evidence",
        "/static/reports/session.pdf": b"private fixture: session report",
    }

    for url, expected_content in private_urls.items():
        anonymous = client.get(url)
        assert anonymous.status_code == 302
        assert "/login" in anonymous.headers["Location"]

        with client.session_transaction() as session:
            session["user_id"] = user_id
        authorized = client.get(url)
        assert authorized.status_code == 200
        assert authorized.data == expected_content

        with client.session_transaction() as session:
            session.clear()

    assert {
        folder: set(os.listdir(os.path.join(media_root, "static", folder)))
        for folder in ("uploads", "evidence", "reports")
    } == {
        "uploads": {"analysis.jpg"},
        "evidence": {"incident.jpg"},
        "reports": {"session.pdf"},
    }


def test_disabled_account_cannot_download_private_files(media_client):
    application, client, user_id, _ = media_client
    with client.session_transaction() as session:
        session["user_id"] = user_id

    conn = db.get_db()
    conn.execute("UPDATE users SET is_active=0 WHERE id=?", (user_id,))
    conn.commit()
    conn.close()

    response = client.get("/static/reports/session.pdf")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]
    with client.session_transaction() as session:
        assert "user_id" not in session


def test_public_static_assets_remain_public(media_client):
    _, client, _, _ = media_client

    response = client.get("/static/assets/public-test.txt")

    assert response.status_code == 200
    assert response.data == b"public stylesheet or marketing asset"


def test_normal_account_cannot_read_unowned_or_other_accounts_media(media_client):
    application, client, user_id, media_root = media_client
    conn = db.get_db()
    other_id = conn.execute(
        "INSERT INTO users (username,email,password_hash,role,full_name,is_active) VALUES (?,?,?,?,?,1)",
        ('media_other', 'media-other@example.test', generate_password_hash('media-other-password'), 'OPERATOR', 'Other')).lastrowid
    conn.execute("INSERT INTO detection_sessions (session_uid,session_type,user_id,evidence_image,status) VALUES ('MEDIA-OTHER','IMAGE',?,'static/uploads/analysis.jpg','COMPLETED')", (other_id,))
    conn.commit()
    conn.close()
    with client.session_transaction() as session:
        session['user_id'] = user_id
    # Same file may have one valid owner relation; an unowned path is still not public.
    assert client.get('/static/uploads/analysis.jpg').status_code == 200
    conn = db.get_db()
    conn.execute('UPDATE detection_sessions SET evidence_image=NULL WHERE session_uid=?', ('MEDIA-OWNED',))
    conn.execute('UPDATE detection_sessions SET evidence_image=NULL WHERE session_uid=?', ('MEDIA-OTHER',))
    conn.commit()
    conn.close()
    assert client.get('/static/uploads/analysis.jpg').status_code == 404


def test_normal_account_cannot_read_media_owned_only_by_another_user(media_client):
    application, client, user_id, _ = media_client
    conn = db.get_db()
    other_id = conn.execute(
        "INSERT INTO users (username,email,password_hash,role,full_name,is_active) VALUES (?,?,?,?,?,1)",
        ('media_other_owner', 'media-other-owner@example.test', generate_password_hash('another-long-password'), 'OPERATOR', 'Other Owner')).lastrowid
    conn.execute('UPDATE detection_sessions SET user_id=? WHERE session_uid=?', (other_id, 'MEDIA-OWNED'))
    conn.commit()
    conn.close()
    with client.session_transaction() as session:
        session['user_id'] = user_id
    assert client.get('/static/uploads/analysis.jpg').status_code == 404