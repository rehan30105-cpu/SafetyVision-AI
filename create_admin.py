"""SafetyVision AI — Initial Admin Account Provisioning CLI.

Run this script to create or reset the primary administrator account:
    python create_admin.py --username admin --email admin@company.com --password <secure-password> --fullname "System Admin"
"""
import argparse
import os
import re
import sys
from werkzeug.security import generate_password_hash

from database.db import get_db, init_db


def main():
    parser = argparse.ArgumentParser(description="Create or update an administrator account.")
    parser.add_argument("--username", required=True, help="Admin username (3-40 alphanumeric chars)")
    parser.add_argument("--email", required=True, help="Admin email address")
    parser.add_argument("--password", required=True, help="Admin password (at least 12 chars)")
    parser.add_argument("--fullname", default="System Administrator", help="Admin display name")
    args = parser.parse_args()

    username = args.username.strip()
    email = args.email.strip().lower()
    password = args.password
    fullname = args.fullname.strip()

    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,40}", username):
        print("Error: Username must be 3-40 alphanumeric characters (._- allowed).", file=sys.stderr)
        sys.exit(1)

    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        print("Error: Invalid email format.", file=sys.stderr)
        sys.exit(1)

    if len(password) < 12:
        print("Error: Password must be at least 12 characters long.", file=sys.stderr)
        sys.exit(1)

    init_db()
    conn = get_db()
    cursor = conn.cursor()

    existing = cursor.execute("SELECT id, username, email FROM users WHERE username=? OR email=?", (username, email)).fetchone()
    pw_hash = generate_password_hash(password)

    if existing:
        cursor.execute(
            "UPDATE users SET password_hash=?, role='ADMIN', full_name=?, is_active=1 WHERE id=?",
            (pw_hash, fullname, existing["id"])
        )
        print(f"Successfully updated existing user '{existing['username']}' (ID: {existing['id']}) as active ADMIN.")
    else:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role, full_name, is_active) VALUES (?, ?, ?, 'ADMIN', ?, 1)",
            (username, email, pw_hash, fullname)
        )
        print(f"Successfully created new ADMIN user '{username}' ({email}).")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
