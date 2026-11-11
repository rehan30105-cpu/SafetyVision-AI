"""Role separation and self-service settings checks using an isolated SQLite DB."""
import os
import secrets
import tempfile

import pytest
from werkzeug.security import check_password_hash, generate_password_hash

from database import db


@pytest.fixture
def settings_app(monkeypatch):
    fd, db_path = tempfile.mkstemp(prefix="safetyvision-user-settings-", suffix=".db")
    os.close(fd)
    monkeypatch.setattr(db, "DB_PATH", db_path)
    import app as application

    application.init_db()
    conn = db.get_db()
    conn.execute(
        "INSERT INTO users (username,email,password_hash,role,full_name,is_active) VALUES (?,?,?,?,?,1)",
        ("settings_admin", "settings-admin@example.test", generate_password_hash("admin-password-123"), "ADMIN", "Settings Admin"),
    )
    conn.execute(
        "INSERT INTO users (username,email,password_hash,role,full_name,is_active) VALUES (?,?,?,?,?,1)",
        ("settings_user", "settings-user@example.test", generate_password_hash("user-password-123"), "OPERATOR", "Original Name"),
    )
    ids = {row["username"]: row["id"] for row in conn.execute("SELECT id,username FROM users WHERE username LIKE 'settings_%'")}
    conn.commit()
    conn.close()
    yield application, application.app.test_client(), ids
    try:
        os.remove(db_path)
    except OSError:
        pass


def sign_in(client, user_id, role="OPERATOR"):
    with client.session_transaction() as session:
        session.update(user_id=user_id, role=role, _csrf_token=secrets.token_urlsafe(32))


def csrf(client):
    with client.session_transaction() as session:
        return session.get("_csrf_token")


def test_role_aware_settings_and_admin_protection(settings_app):
    application, client, ids = settings_app
    assert client.get("/settings").status_code == 302
    assert "/login" in client.get("/settings").headers["Location"]

    sign_in(client, ids["settings_user"], "ADMIN")  # DB role overrides stale session role.
    user_page = client.get("/settings")
    assert user_page.status_code == 200
    html = user_page.get_data(as_text=True)
    assert "Your settings" in html
    assert "settings-user@example.test" in html
    assert "System Settings &amp; Calibration" not in html
    assert "/user-settings/profile" in html
    assert 'data-theme-mode="system"' in html
    assert 'href="/settings"' in html

    assert client.get("/api/settings/update").status_code == 405
    assert client.post("/api/settings/update", data={"csrf_token": csrf(client)}).status_code == 403
    assert client.get("/admin/demo-requests").status_code == 403
    assert client.get("/admin/settings").status_code == 403

    client.post("/logout", data={"csrf_token": csrf(client)})
    sign_in(client, ids["settings_admin"], "OPERATOR")
    admin_page = client.get("/settings")
    assert admin_page.status_code == 302
    assert admin_page.headers["Location"].endswith("/admin/settings")
    full_admin_page = client.get("/admin/settings")
    assert full_admin_page.status_code == 200
    assert "System Settings &amp; Calibration" in full_admin_page.get_data(as_text=True)
    assert "Danger Zone" in full_admin_page.get_data(as_text=True)


def test_profile_updates_are_self_scoped_and_reject_privilege_fields(settings_app):
    application, client, ids = settings_app
    sign_in(client, ids["settings_user"])
    assert client.post("/user-settings/profile", data={"full_name": "Changed Name"}).status_code == 400
    response = client.post("/user-settings/profile", data={
        "csrf_token": csrf(client), "full_name": "Changed Name",
        "user_id": ids["settings_admin"], "role": "ADMIN", "is_active": "0",
    })
    assert response.status_code == 302
    conn = db.get_db()
    user = conn.execute("SELECT full_name,role,is_active FROM users WHERE id=?", (ids["settings_user"],)).fetchone()
    admin = conn.execute("SELECT full_name,role,is_active FROM users WHERE id=?", (ids["settings_admin"],)).fetchone()
    assert tuple(user) == ("Changed Name", "OPERATOR", 1)
    assert tuple(admin) == ("Settings Admin", "ADMIN", 1)
    conn.close()


def test_password_change_requires_current_password_and_logout_uses_csrf(settings_app):
    application, client, ids = settings_app
    sign_in(client, ids["settings_user"])
    base = {"csrf_token": csrf(client), "new_password": "new-user-password-456", "confirm_password": "new-user-password-456"}
    failed = client.post("/user-settings/password", data={**base, "current_password": "wrong-password"})
    assert failed.status_code == 302
    conn = db.get_db()
    old_hash = conn.execute("SELECT password_hash FROM users WHERE id=?", (ids["settings_user"],)).fetchone()[0]
    assert check_password_hash(old_hash, "user-password-123")
    conn.close()

    changed = client.post("/user-settings/password", data={**base, "current_password": "user-password-123"})
    assert changed.status_code == 302
    conn = db.get_db()
    new_hash = conn.execute("SELECT password_hash FROM users WHERE id=?", (ids["settings_user"],)).fetchone()[0]
    assert check_password_hash(new_hash, "new-user-password-456")
    conn.close()

    assert client.post("/logout").status_code == 400
    logout = client.post("/logout", data={"csrf_token": csrf(client)})
    assert logout.status_code == 302
    assert logout.headers["Location"].endswith("/")
    assert client.get("/settings").status_code == 302