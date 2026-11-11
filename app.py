import os
import io
import cv2
import json
import base64
import logging
import re
import secrets
import hashlib
import hmac
import html
from urllib.parse import urlsplit
from datetime import timezone
import numpy as np
from datetime import datetime, timedelta
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
except Exception:
    pass
from flask import (
    Flask, render_template, request, jsonify, redirect, url_for, session, send_file, send_from_directory, flash, abort
)
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps

from database.db import get_db, init_db, log_audit, get_setting
from core.model_manager import ModelManager
from core.annotator import draw_person_annotations
from core.tracker import TrackedPerson, PersonTracker
from services.pipeline import SafetyVisionPipeline
from services.session_service import SessionManager
from core.ppe_metrics import summarize_ppe
from services.report_service import generate_pdf_report, generate_session_pdf_report, generate_overall_pdf_report
from services.recommendation_engine import generate_recommendations
from services.video_analysis_service import VideoAnalysisService
from services.analytics_service import get_dashboard_metrics, get_analytics_metrics
from services.telegram_service import notify_session_completion as _telegram_notify
from services.brevo_email import send_access_decision_email, send_transactional_email
from services.telegram_approval import TelegramDemoApprovalService
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Dedicated logger for live-session lifecycle diagnostics. Never logs secrets.
logger = logging.getLogger("safetyvision.live")
telegram_approval_logger = logging.getLogger('safetyvision.telegram_approval')
telegram_approval_logger.setLevel(logging.INFO)
telegram_approval_logger.propagate = False
if not any(getattr(handler, '_safetyvision_telegram_approval', False)
           for handler in telegram_approval_logger.handlers):
    diagnostic_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
    os.makedirs(diagnostic_dir, exist_ok=True)
    diagnostic_handler = logging.FileHandler(
        os.path.join(diagnostic_dir, 'telegram-approval-callback.log'), encoding='utf-8')
    diagnostic_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s: %(message)s'))
    diagnostic_handler._safetyvision_telegram_approval = True
    telegram_approval_logger.addHandler(diagnostic_handler)


app = Flask(__name__)
configured_secret = os.environ.get('SECRET_KEY')
if os.environ.get('APP_ENV', '').strip().lower() == 'production' and not configured_secret:
    raise RuntimeError('SECRET_KEY must be configured when APP_ENV=production.')
app.secret_key = configured_secret or secrets.token_hex(32)
if not configured_secret:
    logging.getLogger("safetyvision.security").warning("SECRET_KEY is not configured; sessions will reset on restart")
app.config['DEMO_ACTIVATION_MAX_AGE'] = int(os.environ.get('DEMO_ACTIVATION_MAX_AGE', '86400'))
app.config['BREVO_API_KEY'] = os.environ.get('BREVO_API_KEY', '').strip()
app.config['MAIL_FROM_EMAIL'] = os.environ.get('MAIL_FROM_EMAIL', '').strip()
app.config['MAIL_FROM_NAME'] = os.environ.get('MAIL_FROM_NAME', '').strip()
app.config['ADMIN_EMAIL'] = os.environ.get('ADMIN_EMAIL', '').strip()
app.config['PUBLIC_BASE_URL'] = os.environ.get('PUBLIC_BASE_URL', '').strip().rstrip('/')
# Demo approvals use a dedicated bot; retain the legacy token fallback for
# deployments that have not yet configured TELEGRAM_APPROVAL_BOT_TOKEN.
app.config['TELEGRAM_BOT_TOKEN'] = (os.environ.get('TELEGRAM_APPROVAL_BOT_TOKEN') or
                                    os.environ.get('TELEGRAM_BOT_TOKEN', '')).strip()
app.config['TELEGRAM_APPROVAL_ENABLED'] = os.environ.get('TELEGRAM_APPROVAL_ENABLED', 'false').strip().lower() in ('1', 'true', 'yes', 'on')
app.config['TELEGRAM_APPROVAL_CHAT_ID'] = os.environ.get('TELEGRAM_APPROVAL_CHAT_ID', '').strip()
app.config['TELEGRAM_APPROVAL_ADMIN_USER_ID'] = os.environ.get('TELEGRAM_APPROVAL_ADMIN_USER_ID', '').strip()
app.config['TELEGRAM_APPROVAL_WEBHOOK_SECRET'] = os.environ.get('TELEGRAM_APPROVAL_WEBHOOK_SECRET', '').strip()
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE', 'false').strip().lower() in ('1', 'true', 'yes', 'on')
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_DOMAIN'] = None
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 # 32 MB max upload
# Reload Jinja templates automatically on change so UI edits appear without a
# full server restart (dev convenience; safe because it only re-reads templates).
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# --- Rate limiting (Flask-Limiter, in-memory store for a single process) ---
# Limits are per-client-IP. For multi-worker production deployments, swap
# storage_uri for Redis (e.g. storage_uri="redis://localhost:6379") and run
# one shared limiter instance behind the WSGI server.
_RATELIMIT_STORAGE = os.environ.get('RATELIMIT_STORAGE_URI', 'memory://')
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri=_RATELIMIT_STORAGE,
    default_limits=[],  # opt-in limits only, so existing tests are not surprised
)
# Concurrency cap for long-running video inference jobs (protects GPU memory).
MAX_CONCURRENT_VIDEO_JOBS = int(os.environ.get('MAX_CONCURRENT_VIDEO_JOBS', '2'))
_video_job_semaphore = __import__('threading').Semaphore(MAX_CONCURRENT_VIDEO_JOBS)

# Force secure cookies in production regardless of the env flag.
if os.environ.get('APP_ENV', '').strip().lower() == 'production':
    app.config['SESSION_COOKIE_SECURE'] = True

# --- Upload validation ---------------------------------------------------
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}


def _allowed_extension(filename, allowed):
    return bool(filename) and '.' in filename and \
        os.path.splitext(filename)[1].lower() in allowed


def _upload_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')

# In-memory live-preview lifecycle map. The user ID is always from the
# authenticated Flask session; browser-supplied ownership/session IDs are not
# used for authorization. One process owns each active preview session.
_active_live_sessions = {}
_active_live_track_ids = {}
_active_live_sessions_lock = __import__('threading').RLock()

# Ensure directories exist
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
EVIDENCE_FOLDER = os.path.join(app.root_path, 'static', 'evidence')
REPORTS_FOLDER = os.path.join(app.root_path, 'static', 'reports')
for folder in [UPLOAD_FOLDER, EVIDENCE_FOLDER, REPORTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Initialize Database and Pipeline Singleton
init_db()
pipeline = SafetyVisionPipeline()
session_manager = SessionManager.get_instance()
video_service = VideoAnalysisService.get_instance()
telegram_approval = TelegramDemoApprovalService(app.config)
if app.config['TELEGRAM_APPROVAL_ENABLED'] and not telegram_approval.is_configured():
    logging.getLogger('safetyvision.telegram_approval').warning(
        'Telegram demo approvals are enabled but configuration is incomplete; notifications and callbacks are disabled.')

# Jinja Template Filter for JSON parsing
@app.template_filter('fromjson')
def fromjson_filter(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return []
    return value or []

# Context processor to inject global variables
@app.context_processor
def inject_globals():
    try:
        conn = get_db()
        is_admin = session.get('role') == 'ADMIN'
        if is_admin:
            crit_count = conn.execute('SELECT COUNT(*) FROM incidents WHERE severity = "CRITICAL" AND status != "RESOLVED"').fetchone()[0]
            unread_notifs = conn.execute('SELECT COUNT(*) FROM notifications WHERE is_read = 0').fetchone()[0]
        elif session.get('user_id'):
            crit_count = conn.execute('SELECT COUNT(*) FROM incidents WHERE severity = "CRITICAL" AND status != "RESOLVED" AND user_id=?', (session['user_id'],)).fetchone()[0]
            unread_notifs = conn.execute('SELECT COUNT(*) FROM notifications WHERE is_read = 0 AND user_id=?', (session['user_id'],)).fetchone()[0]
        else:
            crit_count = unread_notifs = 0
        conn.close()
    except Exception:
        crit_count = 0
        unread_notifs = 0
    return {
        'critical_count': crit_count,
        'unread_notifs_count': unread_notifs,
        'csrf_token': csrf_token_value()
    }

# Authentication Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        uid = session.get('user_id')
        if not uid:
            # If unauthenticated, redirect to the login page instead of granting access
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Authentication required.'}), 401
            return redirect(url_for('login', next=request.path))
        conn = get_db()
        user = conn.execute('SELECT id, username, role, full_name, is_active FROM users WHERE id=?', (uid,)).fetchone()
        conn.close()
        if user is None or not user['is_active']:
            session.clear()
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Authentication required. Sign in with an active account.'}), 401
            return redirect(url_for('login', next=request.path, access='disabled'))
        # Never authorize from client-controlled/stale session role data.
        session['username'] = user['username']
        session['role'] = user['role']
        session['full_name'] = user['full_name']
        return f(*args, **kwargs)
    return decorated_function


@login_required
def _authorize_private_static_access():
    """Reuse normal active-account checks before serving private static files."""
    return None


@app.before_request
def protect_private_static_files():
    """Keep saved evidence and generated reports behind active-account auth."""
    private_prefixes = ('/static/uploads/', '/static/evidence/', '/static/reports/')
    if request.path.startswith(private_prefixes):
        denied = _authorize_private_static_access()
        if denied:
            return denied
        if session.get('role') == 'ADMIN':
            return None
        relative_path = 'static/' + request.path[len('/static/'):]
        owned_path_candidates = [relative_path]
        relative_dir, relative_name = os.path.split(relative_path)
        if relative_name.startswith('annotated_'):
            owned_path_candidates.append(os.path.join(relative_dir, 'evidence_' + relative_name[len('annotated_'):]))
        conn = get_db()
        ownership_query = '''
            SELECT 1 FROM detection_sessions WHERE evidence_image=? AND user_id=?
            UNION ALL SELECT 1 FROM incidents WHERE evidence_image=? AND user_id=?
            UNION ALL SELECT 1 FROM evidence WHERE image_path=? AND user_id=?
            UNION ALL SELECT 1 FROM generated_reports WHERE file_path=? AND owner_id=?
            LIMIT 1
        '''
        owned = None
        for candidate in owned_path_candidates:
            owned = conn.execute(ownership_query, (candidate, session.get('user_id'), candidate, session.get('user_id'),
                                                    candidate, session.get('user_id'), candidate, session.get('user_id'))).fetchone()
            if owned:
                break
        conn.close()
        if not owned:
            abort(404)


def form_token(purpose):
    return URLSafeTimedSerializer(app.secret_key, salt='safetyvision-' + purpose)


def safe_local_redirect(target, role=None):
    """Accept only same-origin relative paths for post-login redirects."""
    if not target or not isinstance(target, str) or '\\' in target or any(ord(ch) < 32 for ch in target):
        return None
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith('/') or parsed.path.startswith('//'):
        return None
    # Reject protocol-relative/backslash tricks and network-path references.
    if parsed.path.startswith('///') or parsed.path.startswith('/\\'):
        return None
    # Return destinations are an experience preference, never an authorization
    # mechanism. Keep normal users out of administrator workspace URLs.
    privileged_paths = {
        '/admin/demo-requests', '/users', '/settings', '/audit-logs',
        '/api/admin/reset-detection-data', '/api/settings/update',
        '/api/users/create', '/api/cameras/create', '/api/zones/create',
        '/api/incidents/update-status', '/api/notifications/mark-all-read',
    }
    if role in ('OPERATOR', 'SAFETY_OFFICER', 'USER', 'user') and (
            parsed.path in privileged_paths or
            parsed.path.startswith(('/admin/', '/api/admin/', '/api/cameras/', '/api/zones/'))):
        return None
    return target


def valid_form_token(purpose, token, max_age=7200):
    try:
        form_token(purpose).loads(token, max_age=max_age)
        return True
    except (BadSignature, SignatureExpired, TypeError):
        return False


def csrf_token_value():
    token = session.get('_csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['_csrf_token'] = token
    return token


def _create_admin_review_token(request_id, decision):
    nonce = secrets.token_urlsafe(32)
    token = form_token('admin-email-review').dumps(
        {'id': request_id, 'decision': decision, 'nonce': nonce})
    return token, hashlib.sha256(nonce.encode('utf-8')).hexdigest()


def _send_access_request_email(*, recipient_email, recipient_name, subject, intro=None,
                               text_action=None, html_action=None, text=None, html_content=None,
                               action_buttons=None):
    if action_buttons:
        approve_url, reject_url = action_buttons
        review_url = approve_url
        text = (text or '') + '\n\nApprove Request (review page): ' + approve_url + '\nReject Request (review page): ' + reject_url
        html_content = (html_content or '') + ('<p><a style="display:inline-block;padding:12px 20px;background:#16834a;color:white;text-decoration:none" href="{}">Approve Request</a> '
                        '<a style="display:inline-block;padding:12px 20px;background:#b42318;color:white;text-decoration:none" href="{}">Reject Request</a></p>'
                        .format(html.escape(approve_url, quote=True), html.escape(reject_url, quote=True)))
    else:
        text = 'Hello {},\n\n{}\n{}'.format(recipient_name or 'there', intro or '', text_action or '') if text is None else text
        html_content = '<p>Hello {},</p><p>{}</p><p>{}</p>'.format(
            html.escape(recipient_name or 'there'), html.escape(intro or ''), html.escape(html_action or '')) if html_content is None else html_content
    try:
        return send_transactional_email(app.config, recipient_email=recipient_email,
            recipient_name=recipient_name, subject=subject, text_content=text, html_content=html_content)
    except Exception as exc:
        logging.getLogger('safetyvision.access').warning('Request email failed (error_type=%s).', type(exc).__name__)
        return {'ok': False, 'message_id': None, 'error': 'Email provider request failed.'}


def _log_request_email_outcome(request_id, category, result):
    """Log request-email outcomes without addresses, message bodies, or credentials."""
    logging.getLogger('safetyvision.access').info(
        'Demo request email outcome (request_id=%s category=%s function_called=true accepted=%s error=%s).',
        request_id, category, bool(result.get('ok')),
        result.get('error') or 'none')


def _record_email_outcome(request_id, category, result):
    # Existing email columns are retained; category-specific fields are additive migration.
    column = {'REQUEST': 'request_email_status', 'ADMIN': 'admin_email_status', 'DECISION': 'decision_email_status'}[category]
    error_column = column.replace('_status', '_error')
    conn = get_db()
    try:
        conn.execute('UPDATE demo_requests SET {}=?, {}=?, email_attempts=COALESCE(email_attempts,0)+1, email_sent_at=CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE email_sent_at END WHERE id=?'.format(column, error_column),
                     ('ACCEPTED' if result.get('ok') else 'FAILED', None if result.get('ok') else result.get('error'), int(bool(result.get('ok'))), request_id))
        if category == 'DECISION':
            conn.execute('UPDATE demo_requests SET email_status=?,email_error=? WHERE id=?',
                         ('SENT' if result.get('ok') else 'FAILED', None if result.get('ok') else result.get('error'), request_id))
        conn.commit()
    finally:
        conn.close()


def _apply_demo_request_decision(request_id, decision, *, reviewed_by=None, review_notes='', nonce=None):
    """Atomically apply one review using the same account policy for web/Telegram."""
    if decision not in ('APPROVED', 'REJECTED'):
        return {'ok': False, 'reason': 'invalid_decision'}
    conn = get_db()
    activation_token = None
    try:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute("SELECT * FROM demo_requests WHERE id=? AND status='PENDING'", (request_id,)).fetchone()
        if row is None:
            exists = conn.execute('SELECT 1 FROM demo_requests WHERE id=?', (request_id,)).fetchone()
            conn.rollback()
            return {'ok': False, 'reason': 'already_reviewed' if exists else 'not_found'}
        if nonce:
            nonce_column = 'email_approval_nonce_hash' if decision == 'APPROVED' else 'email_rejection_nonce_hash'
            nonce_row = conn.execute('SELECT {} AS nonce_hash FROM demo_requests WHERE id=?'.format(nonce_column), (request_id,)).fetchone()
            expected_hash = hashlib.sha256(nonce.encode('utf-8')).hexdigest()
            if not nonce_row or not nonce_row['nonce_hash'] or not hmac.compare_digest(nonce_row['nonce_hash'], expected_hash):
                conn.rollback()
                return {'ok': False, 'reason': 'invalid_token'}
        if decision == 'APPROVED':
            if conn.execute('SELECT 1 FROM users WHERE email=? COLLATE NOCASE', (row['email'],)).fetchone():
                conn.rollback()
                return {'ok': False, 'reason': 'account_exists'}
            if row['password_hash']:
                base_username = re.sub(r'[^A-Za-z0-9_.-]+', '_', row['email'].split('@', 1)[0]).strip('._-')[:32] or 'user'
                username = base_username
                suffix = 1
                while conn.execute('SELECT 1 FROM users WHERE username=? COLLATE NOCASE', (username,)).fetchone():
                    suffix += 1
                    username = '{}{}'.format(base_username[:max(1, 40-len(str(suffix)))], suffix)
                conn.execute('INSERT INTO users (username,email,password_hash,role,full_name,is_active) VALUES (?,?,?,?,?,1)',
                             (username, row['email'].strip().lower(), row['password_hash'], 'OPERATOR', row['name']))
            else:
                activation_token = secrets.token_urlsafe(32)
                activation_hash = hashlib.sha256(activation_token.encode('utf-8')).hexdigest()
                conn.execute("UPDATE demo_requests SET activation_token_hash=?, activation_expires_at=datetime('now', '+24 hours') WHERE id=?",
                             (activation_hash, request_id))
        changed = conn.execute(
            "UPDATE demo_requests SET status=?,review_notes=?,reviewed_by=?,reviewed_at=CURRENT_TIMESTAMP,"
            "email_approval_nonce_hash=NULL,email_rejection_nonce_hash=NULL WHERE id=? AND status='PENDING'",
            (decision, (review_notes or '')[:1000], reviewed_by, request_id)).rowcount
        if changed != 1:
            conn.rollback()
            return {'ok': False, 'reason': 'already_reviewed'}
        conn.commit()
        return {'ok': True, 'row': dict(row), 'activation_token': activation_token}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _notify_decision_applicant(request_id, decision, row, activation_token=None):
    activation_url = None
    if decision == 'APPROVED' and activation_token:
        public_base = app.config.get('PUBLIC_BASE_URL', '').rstrip('/')
        if public_base:
            activation_url = public_base + url_for('demo_activate', token=activation_token)
    try:
        result = send_access_decision_email(app.config, decision=decision,
            recipient_email=row['email'].strip(), recipient_name=row['name'], activation_url=activation_url)
    except Exception as exc:
        logging.getLogger('safetyvision.access').warning(
            'Access decision email integration failed safely (request_id=%s, error_type=%s).',
            request_id, type(exc).__name__)
        result = {'ok': False, 'message_id': None, 'error': 'Email provider request failed.'}
    _record_email_outcome(request_id, 'DECISION', result)
    if not result.get('ok'):
        logging.getLogger('safetyvision.access').warning(
            'Access decision saved but email was not confirmed (request_id=%s, decision=%s).', request_id, decision)
    return result


def _telegram_demo_callback_data(action, request_id):
    """Create a compact signed payload; Telegram callback_data is untrusted."""
    payload = '{}:{}'.format(action, request_id)
    # Approval buttons can be sent by the Flask process or a maintenance/test
    # process. Use the stable configured webhook secret so both verify the same
    # payload even when development SECRET_KEY is absent and Flask generated a
    # process-local fallback key.
    callback_key = app.config.get('TELEGRAM_APPROVAL_WEBHOOK_SECRET') or app.secret_key
    signing_key = hashlib.sha256(('safetyvision-telegram-callback:' + callback_key).encode('utf-8')).digest()
    signature = hmac.new(signing_key, payload.encode('ascii'), hashlib.sha256).hexdigest()[:24]
    return '{}:{}'.format(payload, signature)


def _send_telegram_demo_notification(request_id, retry_failed=False):
    """Atomically claim and send one pending request's Telegram approval alert."""
    allowed_statuses = ('UNKNOWN', 'FAILED') if retry_failed else ('UNKNOWN',)
    conn = get_db()
    try:
        conn.execute('BEGIN IMMEDIATE')
        placeholders = ','.join('?' for _ in allowed_statuses)
        row = conn.execute(
            'SELECT id,name,email,company,created_at,status FROM demo_requests '
            'WHERE id=? AND status=\'PENDING\' AND telegram_approval_status IN ({})'.format(placeholders),
            (request_id,) + allowed_statuses).fetchone()
        if row is None:
            conn.rollback()
            return {'ok': False, 'skipped': True, 'reason': 'already_sent_or_in_progress'}
        claimed = conn.execute(
            'UPDATE demo_requests SET telegram_approval_status=\'SENDING\', telegram_approval_error=NULL '
            'WHERE id=? AND status=\'PENDING\' AND telegram_approval_status IN ({})'.format(placeholders),
            (request_id,) + allowed_statuses).rowcount
        if not claimed:
            conn.rollback()
            return {'ok': False, 'skipped': True, 'reason': 'already_sent_or_in_progress'}
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    try:
        result = telegram_approval.send_new_request(dict(row), _telegram_demo_callback_data)
    except Exception as exc:
        logger.warning('Demo request Telegram notification failed safely (request_id=%s, error_type=%s).',
                       request_id, type(exc).__name__)
        result = {'ok': False, 'error': 'Telegram notification failed.'}

    conn = get_db()
    try:
        if result.get('ok'):
            conn.execute(
                "UPDATE demo_requests SET telegram_approval_status='SENT', telegram_approval_error=NULL, "
                'telegram_approval_message_id=?, telegram_approval_sent_at=CURRENT_TIMESTAMP '
                "WHERE id=? AND telegram_approval_status='SENDING'",
                ((result.get('result') or {}).get('message_id'), request_id))
        else:
            error = str(result.get('error') or result.get('description') or 'Telegram did not accept the notification.')[:240]
            conn.execute(
                "UPDATE demo_requests SET telegram_approval_status='FAILED', telegram_approval_error=? "
                "WHERE id=? AND telegram_approval_status='SENDING'", (error, request_id))
        conn.commit()
    finally:
        conn.close()
    logger.info('Demo request Telegram notification outcome (request_id=%s, accepted=%s).',
                request_id, bool(result.get('ok')))
    return result


def _parse_telegram_demo_callback_data(value):
    match = re.fullmatch(r'(approve|reject):([1-9][0-9]{0,9}):([0-9a-f]{24})', str(value or ''))
    if not match:
        return None
    payload = '{}:{}'.format(match.group(1), match.group(2))
    callback_key = app.config.get('TELEGRAM_APPROVAL_WEBHOOK_SECRET') or app.secret_key
    signing_key = hashlib.sha256(('safetyvision-telegram-callback:' + callback_key).encode('utf-8')).digest()
    expected = hmac.new(signing_key, payload.encode('ascii'), hashlib.sha256).hexdigest()[:24]
    if not hmac.compare_digest(match.group(3), expected):
        return None
    return ('APPROVED' if match.group(1) == 'approve' else 'REJECTED', int(match.group(2)))


def _send_decision_notification(request_id, decision):
    conn = get_db()
    row = conn.execute('SELECT * FROM demo_requests WHERE id=?', (request_id,)).fetchone()
    conn.close()
    if row is None:
        return {'ok': False, 'message_id': None, 'error': 'Request no longer exists.'}
    activation_url = None
    if decision == 'APPROVED' and not row['password_hash'] and not row['activated_at']:
        raw_token = secrets.token_urlsafe(32)
        conn = get_db()
        try:
            conn.execute('UPDATE demo_requests SET activation_token_hash=?, activation_expires_at=datetime(\'now\', \'+24 hours\') WHERE id=? AND status=\'APPROVED\'',
                         (hashlib.sha256(raw_token.encode('utf-8')).hexdigest(), request_id))
            conn.commit()
        finally:
            conn.close()
        activation_url = app.config.get('PUBLIC_BASE_URL', '').rstrip('/') + url_for('demo_activate', token=raw_token)
    result = send_access_decision_email(app.config, decision=decision,
        recipient_email=row['email'], recipient_name=row['name'], activation_url=activation_url)
    _record_email_outcome(request_id, 'DECISION', result)
    return result


def _retry_request_notification(request_id, category):
    conn = get_db()
    row = conn.execute('SELECT * FROM demo_requests WHERE id=?', (request_id,)).fetchone()
    conn.close()
    if row is None:
        return {'ok': False, 'message_id': None, 'error': 'Request no longer exists.'}
    if category == 'DECISION':
        return _send_decision_notification(request_id, row['status']) if row['status'] in ('APPROVED', 'REJECTED') else {'ok': False, 'message_id': None, 'error': 'Request is still pending.'}
    if category == 'REQUEST':
        result = _send_access_request_email(recipient_email=row['email'], recipient_name=row['name'],
            subject='We received your SafetyVision AI access request',
            intro='Your request is pending administrator review. You cannot sign in until it is approved.',
            text_action='We will email you again when an administrator makes a decision.')
        return result
    if category == 'ADMIN' and row['status'] == 'PENDING':
        base = app.config.get('PUBLIC_BASE_URL', '').rstrip('/')
        if not base:
            return {'ok': False, 'message_id': None, 'error': 'Public website URL is not configured.'}
        urls = []
        for decision in ('APPROVED', 'REJECTED'):
            signed, nonce_hash = _create_admin_review_token(request_id, decision)
            column = 'email_approval_nonce_hash' if decision == 'APPROVED' else 'email_rejection_nonce_hash'
            conn = get_db()
            try:
                conn.execute('UPDATE demo_requests SET {}=? WHERE id=? AND status=\'PENDING\''.format(column), (nonce_hash, request_id))
                conn.commit()
            finally:
                conn.close()
            urls.append(base + url_for('admin_demo_request_review_page', token=signed, decision=decision))
        body = '<p>Request from {} ({}) at {}.</p><p>{}</p>'.format(
            html.escape(row['name']), html.escape(row['email']), html.escape(row['company']), html.escape(row['message'] or ''))
        return _send_access_request_email(recipient_email=app.config.get('ADMIN_EMAIL'),
            recipient_name='SafetyVision administrator', subject='New SafetyVision AI access request',
            text='New request: {} / {} / {}\nApprove: {}\nReject: {}'.format(row['name'], row['email'], row['company'], *urls),
            html_content=body + '<p><a href="{}">Approve Request</a> &nbsp; <a href="{}">Reject Request</a></p>'.format(
                html.escape(urls[0], quote=True), html.escape(urls[1], quote=True)))
    return {'ok': False, 'message_id': None, 'error': 'This notification cannot be retried for the current request status.'}


@app.route('/admin/demo-requests/email-review/<token>')
def admin_demo_request_review_page(token):
    # Validate the logged-in principal before even looking up request details.
    user_id = session.get('user_id')
    if user_id is None:
        return redirect(url_for('login', next=url_for('admin_demo_request_review_page', token=token,
                            decision=request.args.get('decision'))))
    auth_conn = get_db()
    admin = auth_conn.execute('SELECT role,is_active FROM users WHERE id=?', (user_id,)).fetchone()
    auth_conn.close()
    if not admin or admin['role'] != 'ADMIN' or not admin['is_active']:
        abort(403)
    try:
        data = form_token('admin-email-review').loads(token, max_age=86400)
    except (BadSignature, SignatureExpired, TypeError):
        abort(400)
    conn = get_db()
    row = conn.execute('SELECT id,name,email,company,role,message,status,created_at FROM demo_requests WHERE id=?', (data.get('id'),)).fetchone()
    conn.close()
    if row is None or row['status'] != 'PENDING':
        abort(404)
    decision = request.args.get('decision', '').upper()
    if decision not in ('APPROVED', 'REJECTED') or data.get('decision') != decision or not data.get('nonce'):
        abort(400)
    nonce_column = 'email_approval_nonce_hash' if decision == 'APPROVED' else 'email_rejection_nonce_hash'
    nonce_conn = get_db()
    nonce_row = nonce_conn.execute('SELECT {} AS nonce_hash FROM demo_requests WHERE id=?'.format(nonce_column), (data['id'],)).fetchone()
    nonce_conn.close()
    if not nonce_row or not nonce_row['nonce_hash'] or not hmac.compare_digest(
            nonce_row['nonce_hash'], hashlib.sha256(data['nonce'].encode('utf-8')).hexdigest()):
        abort(400)
    return render_template('admin_email_review.html', item=dict(row), token=token,
        decision=decision, csrf_token=csrf_token_value(),
        form_token=form_token('demo-review').dumps({'id': row['id']}), email_action_token=token)


@app.before_request
def protect_state_changes():
    """Validate purpose-bound CSRF tokens for every browser state change."""
    if request.endpoint == 'telegram_demo_approval_webhook':
        # Telegram authenticates webhook updates with its dedicated secret header.
        return None
    if request.method not in ('POST', 'PUT', 'PATCH', 'DELETE'):
        return None
    expected = session.get('_csrf_token')
    supplied = request.form.get('csrf_token') or request.headers.get('X-CSRF-Token')
    if not expected or not supplied or not hmac.compare_digest(expected, supplied):
        abort(400, description='The form security token is missing or invalid. Reload and try again.')
    # Retain the existing purpose-bound signed tokens for the public request,
    # admin decision, and legacy activation workflows as an additional check.
    signed_purposes = {'login': 'login', 'demo_request_submit': 'demo-request',
                       'admin_demo_request_review': 'demo-review', 'demo_activate': 'demo-activate'}
    purpose = signed_purposes.get(request.endpoint)
    if purpose:
        try:
            payload = form_token(purpose).loads(request.form.get('form_token'), max_age=7200)
        except (BadSignature, SignatureExpired, TypeError):
            abort(400, description='The form security token is missing, invalid, or expired.')
        if request.endpoint == 'admin_demo_request_review' and payload.get('id') != request.view_args.get('request_id'):
            abort(400, description='The review token does not match this request.')
        if request.endpoint == 'demo_activate':
            token_hash = hashlib.sha256(request.view_args['token'].encode('utf-8')).hexdigest()
            if payload.get('token_hash') != token_hash:
                abort(400, description='The activation token does not match this form.')


@app.after_request
def add_security_headers(response):
    # Prevent browsing/back-button exposure of authenticated pages
    response.cache_control.no_store = True
    response.cache_control.no_cache = True
    response.cache_control.must_revalidate = True
    response.cache_control.max_age = 0
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    # Enforce HTTPS in production (HSTS). Harmless in development.
    if os.environ.get('APP_ENV', '').strip().lower() == 'production':
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    # Content Security Policy: same-origin first-party resources, inline
    # styles/scripts required by the theme bootstrap, Chart.js CDN, and data:
    # image URIs used by the app. No unsafe-eval.
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self' https://fonts.gstatic.com https://fonts.googleapis.com; "
        "connect-src 'self'; "
        "frame-ancestors 'self'; "
        "base-uri 'self';"
    )
    return response


@app.errorhandler(403)
def forbidden_page(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'You do not have permission to access this resource.'}), 403
    return render_template('error.html', code=403, title='Access denied', message='Your account does not have permission to view this page.'), 403


@app.errorhandler(404)
def not_found_page(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Resource not found.'}), 404
    return render_template('error.html', code=404, title='Page not found', message='The page may have moved or is not available.'), 404


@app.errorhandler(405)
def method_not_allowed_page(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'This method is not allowed for this resource.'}), 405
    return render_template('error.html', code=405, title='Action not available', message='Use the controls provided on the page to complete this action.'), 405


@app.errorhandler(500)
def internal_error_page(error):
    app.logger.exception('Unhandled application error', exc_info=error)
    if request.path.startswith('/api/'):
        return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500
    return render_template('error.html', code=500, title='Something went wrong', message='The request could not be completed. Please try again later.'), 500


@app.errorhandler(429)
def ratelimit_handler(error):
    """Friendly rate-limit responses so clients know to slow down."""
    description = getattr(error, 'description', None) or 'Too many requests. Please slow down.'
    retry_after = None
    try:
        retry_after = getattr(error, 'retry_after', None)
    except Exception:
        retry_after = None
    if request.path.startswith('/api/'):
        response = jsonify({'error': description, 'code': 429})
        response.status_code = 429
    else:
        response = render_template('error.html', code=429, title='Too many requests',
                                   message=description), 429
    if retry_after:
        try:
            response.headers['Retry-After'] = str(int(retry_after))
        except Exception:
            pass
    return response


# Role Authorization Helper
def role_required(min_role):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Re-read the current role from the database; do not trust cookie state.
            conn = get_db()
            user = conn.execute('SELECT role,is_active FROM users WHERE id=?', (session.get('user_id'),)).fetchone()
            conn.close()
            user_role = user['role'] if user and user['is_active'] else None
            if min_role == 'ADMIN' and user_role != 'ADMIN':
                if request.path.startswith('/api/'):
                    return jsonify({'error': 'Administrator privileges required.'}), 403
                abort(403)
            return f(*args, **kwargs)
        return wrapper
    return decorator


# ========================================================
# 1. AUTHENTICATION & SESSION ROUTES
# ========================================================

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("20 per minute; 100 per hour", methods=["POST"])
def login():
    # A signed-in visitor still needs a live, approved account.
    if 'user_id' in session:
        conn = get_db()
        user = conn.execute('SELECT role,is_active FROM users WHERE id=?', (session['user_id'],)).fetchone()
        conn.close()
        if user and user['is_active']:
            destination = safe_local_redirect(request.args.get('next'), user['role'])
            return redirect(destination or url_for('dashboard'))
        session.clear()

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        next_url = request.form.get('next') or request.args.get('next')

        if not username or not password:
            return render_template('login.html', form_token=form_token('login').dumps({'form': 'login'}), csrf_token=csrf_token_value(), error='Username and password are required.'), 400

        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ? OR email = ? COLLATE NOCASE', (username, username)).fetchone()
        access_request = None if user else conn.execute(
            'SELECT status FROM demo_requests WHERE email=? COLLATE NOCASE ORDER BY created_at DESC LIMIT 1',
            (username.strip().lower(),)).fetchone()
        conn.close()

        if user and check_password_hash(user['password_hash'], password):
            if not user['is_active']:
                return render_template('login.html', form_token=form_token('login').dumps({'form': 'login'}), csrf_token=csrf_token_value(), error='This account is disabled. Contact an administrator.'), 403
            # Renew the session to prevent fixation; keep role-based data in sync
            session.clear()
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            session['full_name'] = user['full_name']
            session.permanent = True
            csrf_token_value()

            log_audit(user['id'], user['username'], 'LOGIN', f"User logged in with role {user['role']}")

            # Redirect to the originally requested page (validated) or the dashboard
            # Never allow an admin to be redirected into a normal-user-only
            # workspace or accept arbitrary caller-controlled redirect targets.
            destination = safe_local_redirect(next_url, user['role'])
            return redirect(destination or url_for('dashboard'))

        # Never reveal whether the username or password was wrong
        if access_request and access_request['status'] == 'PENDING':
            return render_template('login.html', form_token=form_token('login').dumps({'form': 'login'}), csrf_token=csrf_token_value(), error='Your access request is pending administrator approval. You cannot sign in until it is approved.'), 403
        if access_request and access_request['status'] == 'REJECTED':
            return render_template('login.html', form_token=form_token('login').dumps({'form': 'login'}), csrf_token=csrf_token_value(), error='Your latest access request was rejected. You may submit a new request or contact an administrator.'), 403
        return render_template('login.html', form_token=form_token('login').dumps({'form': 'login'}), csrf_token=csrf_token_value(), error='Invalid username or password credentials.'), 401

    return render_template('login.html', form_token=form_token('login').dumps({'form': 'login'}), csrf_token=csrf_token_value(),
                           next_url=safe_local_redirect(request.args.get('next'), session.get('role')) or '', access_message=(
        'This account is disabled or awaiting administrator approval.' if request.args.get('access') == 'disabled' else None))

@app.route('/logout', methods=['POST'])
def logout():
    uid = session.get('user_id')
    uname = session.get('username', 'Anonymous')
    if uid:
        log_audit(uid, uname, 'LOGOUT', 'User signed out')
    session.clear()
    response = redirect(url_for('landing'))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response


@app.route('/explore')
def explore_platform():
    """Public entry point; resolve access only through authenticated login."""
    uid = session.get('user_id')
    if uid:
        conn = get_db()
        user = conn.execute('SELECT role,is_active FROM users WHERE id=?', (uid,)).fetchone()
        conn.close()
        if user and user['is_active']:
            return redirect(url_for('dashboard'))
        session.clear()
    return redirect(url_for('login', next=url_for('dashboard')))


# ========================================================
# 2. MAIN NAVIGATION PAGES
# ========================================================

@app.route('/dashboard')
@login_required
def dashboard():
    is_admin = session.get('role') == 'ADMIN'
    data = get_dashboard_metrics(user_id=session.get('user_id'), is_admin=is_admin)
    kpis = data['kpis']
    charts = data['charts']

    # Active alerts (unread notifications — separate from KPIs, sidebar badge)
    conn = get_db()
    if is_admin:
        active_alerts = conn.execute('SELECT COUNT(*) FROM notifications WHERE is_read=0').fetchone()[0]
    else:
        active_alerts = conn.execute('SELECT COUNT(*) FROM notifications WHERE is_read=0 AND user_id=?',
                                     (session.get('user_id'),)).fetchone()[0]
    conn.close()
    kpis['active_alerts'] = active_alerts

    return render_template(
        'dashboard.html',
        active_page='dashboard',
        kpis=kpis,
        recent_incidents=data['recent_incidents'],
        charts_json=charts,
        has_sessions=data['has_sessions'],
        has_incidents=data['has_incidents'],
        dashboard_scope='System-wide' if is_admin else 'Your account',
    )


@app.route('/')
def landing():
    # Public marketing landing page. No authentication required.
    # Authenticated users reach the working dashboard via /dashboard (or the app nav).
    return render_template('landing.html', demo_form_token=form_token('demo-request').dumps({'form': 'demo'}),
                           csrf_token=csrf_token_value(), login_form_token=form_token('login').dumps({'form': 'login'}))


@app.route('/demo/request', methods=['POST'])
@limiter.limit("15 per minute; 100 per hour", methods=["POST"])
def demo_request_submit():
    if not valid_form_token('demo-request', request.form.get('form_token')):
        return render_template('landing.html', demo_error='Your form expired. Please reload the page and try again.', demo_form_token=form_token('demo-request').dumps({'form': 'demo'}), csrf_token=csrf_token_value()), 400

    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip().lower()
    company = request.form.get('company', '').strip()
    # This field is descriptive only; privilege is always assigned by the
    # approval handler from the server-side role policy.
    role = request.form.get('role', '').strip()
    message = request.form.get('message', '').strip()
    password = request.form.get('password', '')
    confirmation = request.form.get('confirm_password', '')
    if 'user_id' in session:
        conn = get_db()
        account = conn.execute('SELECT id,is_active FROM users WHERE id=?', (session['user_id'],)).fetchone()
        conn.close()
        if account and account['is_active']:
            return render_template('landing.html', demo_error='Sign out before submitting an access request.', demo_form_token=form_token('demo-request').dumps({'form': 'demo'}), csrf_token=csrf_token_value()), 403
        session.clear()
    if (not name or len(name) > 120 or not company or len(company) > 160 or not role or len(role) > 120
            or len(message) > 2000 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email) or len(email) > 254):
        return render_template('landing.html', demo_error='Enter a valid name, work email, company, and role. Notes must be under 2,000 characters.', demo_form_token=form_token('demo-request').dumps({'form': 'demo'}), csrf_token=csrf_token_value()), 400
    if len(password) < 12 or len(password) > 256 or password != confirmation:
        return render_template('landing.html', demo_error='Choose a password of at least 12 characters and confirm it.', demo_form_token=form_token('demo-request').dumps({'form': 'demo'}), csrf_token=csrf_token_value()), 400
    conn = get_db()
    try:
        conn.execute('BEGIN IMMEDIATE')
        if conn.execute('SELECT 1 FROM users WHERE email=? COLLATE NOCASE', (email,)).fetchone():
            conn.rollback()
            conn.close()
            return render_template('landing.html', demo_error='An account already exists for this email. Use sign in or contact an administrator.', demo_form_token=form_token('demo-request').dumps({'form': 'demo'}), csrf_token=csrf_token_value()), 409
        pending = conn.execute("SELECT id,telegram_approval_status FROM demo_requests WHERE email=? COLLATE NOCASE AND status='PENDING' ORDER BY created_at DESC LIMIT 1", (email,)).fetchone()
        if pending:
            conn.rollback()
            conn.close()
            logging.getLogger('safetyvision.access').info('Demo request not saved (reason=duplicate_pending).')
            return render_template('landing.html', demo_error='An existing request is PENDING and awaiting administrator review. Check your email for updates; you do not need to submit again.', demo_form_token=form_token('demo-request').dumps({'form': 'demo'}), csrf_token=csrf_token_value()), 409
        approved = conn.execute("SELECT status FROM demo_requests WHERE email=? COLLATE NOCASE AND status='APPROVED' ORDER BY created_at DESC LIMIT 1", (email,)).fetchone()
        if approved:
            conn.rollback()
            conn.close()
            logging.getLogger('safetyvision.access').info('Demo request not saved (reason=duplicate_approved).')
            return render_template('landing.html', demo_error='An existing request is APPROVED. Sign in with this email or contact an administrator for help.', demo_form_token=form_token('demo-request').dumps({'form': 'demo'}), csrf_token=csrf_token_value()), 409
        conn.execute('INSERT INTO demo_requests (name, email, company, role, message, password_hash) VALUES (?, ?, ?, ?, ?, ?)',
                     (name, email, company, role, message, generate_password_hash(password)))
        request_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        logging.getLogger('safetyvision.access').exception('Access request submission failed')
        return render_template('landing.html', demo_error='Your request could not be saved. Please try again.', demo_form_token=form_token('demo-request').dumps({'form': 'demo'}), csrf_token=csrf_token_value()), 500
    conn.close()
    logging.getLogger('safetyvision.access').info(
        'Demo request saved (request_id=%s status=PENDING).', request_id)
    # Both notifications are sent only after the request transaction commits.
    admin_base_url = app.config.get('PUBLIC_BASE_URL', '').rstrip('/')
    admin_action_urls = None
    if admin_base_url:
        action_links = []
        for decision in ('APPROVED', 'REJECTED'):
            review_token, nonce_hash = _create_admin_review_token(request_id, decision)
            nonce_column = 'email_approval_nonce_hash' if decision == 'APPROVED' else 'email_rejection_nonce_hash'
            token_conn = get_db()
            try:
                token_conn.execute('UPDATE demo_requests SET {}=? WHERE id=? AND status=\'PENDING\''.format(nonce_column),
                                   (nonce_hash, request_id))
                token_conn.commit()
            finally:
                token_conn.close()
            action_links.append(admin_base_url + url_for('admin_demo_request_review_page',
                token=review_token, decision=decision))
        admin_action_urls = tuple(action_links)
    admin_review_url = admin_action_urls[0] if admin_action_urls else None
    created_at_conn = get_db()
    created_at = created_at_conn.execute('SELECT created_at FROM demo_requests WHERE id=?', (request_id,)).fetchone()[0]
    created_at_conn.close()
    _send_telegram_demo_notification(request_id)
    applicant_email = _send_access_request_email(
        recipient_email=email, recipient_name=name, subject='We received your SafetyVision AI access request',
        intro='Your request is pending administrator review. You cannot sign in until it is approved.',
        text_action='We will email you again when an administrator makes a decision.',
        html_action='We will email you again when an administrator makes a decision.')
    _log_request_email_outcome(request_id, 'REQUEST', applicant_email)
    admin_details_text = ('New demo/access request\nName: {}\nEmail: {}\nOrganization: {}\nRole/purpose: {}\nDetails: {}\nRequest date: {}'
                          .format(name, email, company, role, message or '(none)', created_at,
                                  ))
    admin_details_html = ('<p>A new demo/access request is pending.</p><dl>'
                          '<dt>Applicant</dt><dd>{}</dd><dt>Email</dt><dd>{}</dd>'
                          '<dt>Organization</dt><dd>{}</dd><dt>Role/purpose</dt><dd>{}</dd>'
                          '<dt>Details</dt><dd>{}</dd><dt>Request date</dt><dd>{}</dd></dl>{}'
                          .format(html.escape(name), html.escape(email), html.escape(company),
                                  html.escape(role), html.escape(message or '(none)'), html.escape(str(created_at)),
                                  ('<p><a href="{}">Review request</a></p>'.format(html.escape(admin_review_url, quote=True))
                                   if admin_review_url else '')))
    if admin_action_urls:
        admin_email = _send_access_request_email(
            recipient_email=app.config.get('ADMIN_EMAIL', ''), recipient_name='SafetyVision administrator',
            subject='New SafetyVision AI access request', text=admin_details_text, html_content=admin_details_html,
            action_buttons=admin_action_urls)
    else:
        admin_email = {'ok': False, 'message_id': None,
                       'error': 'Public website URL is not configured; administrator action links were not created.'}
        logging.getLogger('safetyvision.access').warning(
            'Demo request email outcome (request_id=%s category=ADMIN function_called=false accepted=false error=public_url_not_configured).',
            request_id)
    if admin_action_urls:
        _log_request_email_outcome(request_id, 'ADMIN', admin_email)
    _record_email_outcome(request_id, 'REQUEST', applicant_email)
    _record_email_outcome(request_id, 'ADMIN', admin_email)
    return render_template('landing.html', demo_success='Your request was saved as PENDING. An administrator must approve it before you can sign in.' +
                           (' A confirmation email was accepted by Brevo.' if applicant_email.get('ok') else
                            ' The confirmation email was not accepted; the administrator will still review your saved request.'),
                            demo_form_token=form_token('demo-request').dumps({'form': 'demo'}), csrf_token=csrf_token_value()), 201
@app.route('/admin/demo-requests')
@login_required
@role_required('ADMIN')
def admin_demo_requests():
    conn = get_db()
    query = '''
        SELECT dr.id,dr.name,dr.email,dr.company,dr.role,dr.message,dr.status,
               dr.review_notes,dr.reviewed_at,dr.activated_at,dr.email_status,dr.email_error,
                dr.request_email_status,dr.request_email_error,dr.admin_email_status,dr.admin_email_error,
                dr.telegram_approval_status,dr.telegram_approval_error,
               dr.decision_email_status,dr.decision_email_error,
               dr.created_at,u.username AS reviewer_name,
               (SELECT COUNT(*) FROM users account WHERE account.email=dr.email COLLATE NOCASE) AS account_count,
               (dr.password_hash IS NOT NULL AND dr.password_hash != '') AS has_password
        FROM demo_requests dr LEFT JOIN users u ON u.id=dr.reviewed_by
        WHERE (? = '' OR dr.name LIKE ? OR dr.email LIKE ? OR dr.company LIKE ? OR dr.role LIKE ? OR dr.message LIKE ?)
          AND (? = '' OR dr.status = ?)
        ORDER BY CASE dr.status WHEN 'PENDING' THEN 0 ELSE 1 END, dr.created_at DESC
    '''
    search = request.args.get('q', '').strip()[:120]
    status_filter = request.args.get('status', '').strip().upper()
    if status_filter not in ('', 'PENDING', 'APPROVED', 'REJECTED'):
        status_filter = ''
    like = '%{}%'.format(search)
    requests_list = [dict(row) for row in conn.execute(
        query, (search, like, like, like, like, like, status_filter, status_filter)).fetchall()]
    request_counts = {row['status']: row['count'] for row in conn.execute(
        "SELECT status, COUNT(*) AS count FROM demo_requests GROUP BY status").fetchall()}
    request_counts['TOTAL'] = sum(request_counts.values())
    conn.close()
    return render_template('demo_requests.html', active_page='demo_requests', requests_list=requests_list,
                           request_counts=request_counts, query_q=search, query_status=status_filter,
                           decision_tokens={str(item['id']): form_token('demo-review').dumps({'id': item['id']}) for item in requests_list},
                           activation_tokens={str(item['id']): form_token('demo-activation-issue').dumps({'id': item['id']}) for item in requests_list})


@app.route('/telegram/demo-approval/webhook', methods=['POST'])
def telegram_demo_approval_webhook():
    """Receive secret-protected callback updates forwarded by the local poller."""
    approval_logger = logging.getLogger('safetyvision.telegram_approval')
    configured_secret = app.config.get('TELEGRAM_APPROVAL_WEBHOOK_SECRET', '')
    supplied_secret = request.headers.get('X-Telegram-Bot-Api-Secret-Token', '')
    if not telegram_approval.is_active() or not configured_secret or not hmac.compare_digest(supplied_secret, configured_secret):
        approval_logger.warning('Telegram approval update rejected (reason=inactive_or_invalid_forward_secret).')
        abort(403)
    update = request.get_json(silent=True)
    callback = update.get('callback_query') if isinstance(update, dict) else None
    if not isinstance(callback, dict):
        approval_logger.info('Stage 1: Telegram callback_query received: PASS (non-callback update).')
        return jsonify({'ok': True})
    callback_id = callback.get('id')
    sender = callback.get('from') or {}
    message = callback.get('message') or {}
    chat = message.get('chat') or {}
    callback_data = callback.get('data')
    approval_logger.info('Stage 1: Telegram callback_query received: PASS (callback_id=%s).', callback_id or 'unknown')
    parsed = _parse_telegram_demo_callback_data(callback_data)
    approval_logger.info('Telegram approval callback received (callback_present=true, callback_data_valid=%s).',
                         parsed is not None)
    answer_text = 'Invalid approval action.'
    answer_alert = False
    if str(sender.get('id', '')) != telegram_approval.admin_user_id:
        approval_logger.warning('Stage 4: Admin authorization verified: FAIL (reason=admin_user_mismatch).')
        approval_logger.warning('Telegram approval authorization denied (reason=admin_user_mismatch).')
        answer_text, answer_alert = 'You are not authorized to review demo requests.', True
    elif (str(chat.get('id', '')) != telegram_approval.chat_id or chat.get('type') != 'private'
          or not message.get('message_id')):
        approval_logger.warning('Stage 4: Admin authorization verified: FAIL (reason=chat_not_admin_private).')
        answer_text, answer_alert = 'This approval message is not from the configured administrator chat.', True
    elif parsed is None:
        approval_logger.warning('Stage 2: Callback data parsed: FAIL (reason=invalid_or_expired_payload).')
        approval_logger.warning('Stage 3: Request ID extracted: FAIL (reason=invalid_or_expired_payload).')
        approval_logger.warning('Telegram approval callback rejected (reason=invalid_or_expired_payload).')
        answer_text, answer_alert = 'This approval button is invalid or has expired.', True
    else:
        decision, request_id = parsed
        approval_logger.info('Stage 2: Callback data parsed: PASS (action=%s).', decision)
        approval_logger.info('Stage 3: Request ID extracted: PASS (request_id=%s, action=%s).', request_id, decision)
        approval_logger.info('Telegram approval callback parsed (request_id=%s, action=%s).', request_id, decision)
        approval_logger.info('Stage 4: Admin authorization verified: PASS (request_id=%s, action=%s).', request_id, decision)
        approval_logger.info('Telegram approval authorization passed (request_id=%s, action=%s).', request_id, decision)
        approval_logger.info('Stage 6: Database update initiated: PASS (request_id=%s, action=%s).', request_id, decision)
        try:
            outcome = _apply_demo_request_decision(request_id, decision, reviewed_by=None,
                review_notes='Reviewed via Telegram by configured administrator.')
        except Exception as exc:
            approval_logger.exception('Telegram demo decision failed safely (request_id=%s, action=%s, error_type=%s).',
                                      request_id, decision, type(exc).__name__)
            outcome = {'ok': False, 'reason': 'database_error'}
        if outcome['ok']:
            approval_logger.info('Stage 7: Database commit completed: PASS (request_id=%s, status=%s).', request_id, decision)
            approval_logger.info('Telegram demo decision committed (request_id=%s, status=%s).', request_id, decision)
            answer_text = 'Request #{} {}'.format(request_id, decision.lower())
            # Acknowledge Telegram before email/API work so its callback spinner
            # is cleared promptly even when Brevo or Telegram is slow.
            if callback_id:
                try:
                    callback_answer = telegram_approval.answer_callback(callback_id, answer_text, False)
                except Exception as exc:
                    approval_logger.warning('Telegram answerCallbackQuery failed safely (error_type=%s).', type(exc).__name__)
                    callback_answer = {'ok': False, 'error': type(exc).__name__}
                if callback_answer.get('ok'):
                    approval_logger.info('Stage 5: answerCallbackQuery executed: PASS (request_id=%s).', request_id)
                else:
                    approval_logger.warning('Stage 5: answerCallbackQuery executed: FAIL (request_id=%s, error=%s).',
                                            request_id, callback_answer.get('error') or 'rejected')
                approval_logger.info('Telegram callback answer result (request_id=%s, ok=%s, http_status=%s).',
                                     request_id, bool(callback_answer.get('ok')), callback_answer.get('http_status'))
                callback_id = None

            try:
                email_result = _notify_decision_applicant(request_id, decision, outcome['row'], outcome['activation_token'])
                if email_result.get('ok'):
                    approval_logger.info('Stage 10: Applicant email notification attempted: PASS (request_id=%s).', request_id)
                else:
                    approval_logger.warning('Stage 10: Applicant email notification attempted: FAIL (request_id=%s, error=%s).',
                                            request_id, email_result.get('error') or 'unconfirmed')
            except Exception as exc:
                approval_logger.warning('Stage 10: Applicant email notification attempted: FAIL (request_id=%s, error_type=%s).',
                                        request_id, type(exc).__name__)

            try:
                edit_result = telegram_approval.edit_decision_message(chat['id'], message['message_id'], request_id, decision)
            except Exception as exc:
                approval_logger.warning('Telegram editMessageText failed safely (error_type=%s).', type(exc).__name__)
                edit_result = {'ok': False, 'error': type(exc).__name__}
            if edit_result.get('ok'):
                approval_logger.info('Stage 8: Original Telegram message edited: PASS (request_id=%s, status=%s).', request_id, decision)
                approval_logger.info('Stage 9: Inline buttons removed: PASS (request_id=%s).', request_id)
            else:
                approval_logger.warning('Stage 8: Original Telegram message edited: FAIL (request_id=%s, error=%s).',
                                        request_id, edit_result.get('error') or 'rejected')
                approval_logger.warning('Stage 9: Inline buttons removed: FAIL (request_id=%s).', request_id)
            approval_logger.info('Telegram approval message edit result (request_id=%s, status=%s, ok=%s, http_status=%s).',
                                 request_id, decision, bool(edit_result.get('ok')), edit_result.get('http_status'))
        elif outcome['reason'] == 'already_reviewed':
            approval_logger.warning('Stage 7: Database commit completed: FAIL (request_id=%s, reason=already_reviewed).', request_id)
            conn = get_db()
            reviewed = conn.execute('SELECT status FROM demo_requests WHERE id=?', (request_id,)).fetchone()
            conn.close()
            if reviewed and reviewed['status'] in ('APPROVED', 'REJECTED'):
                try:
                    edit_result = telegram_approval.edit_decision_message(chat['id'], message['message_id'], request_id, reviewed['status'])
                except Exception as exc:
                    approval_logger.warning('Telegram editMessageText failed safely (error_type=%s).', type(exc).__name__)
                    edit_result = {'ok': False, 'error': type(exc).__name__}
                if edit_result.get('ok'):
                    approval_logger.info('Stage 8: Original Telegram message edited: PASS (request_id=%s, status=%s).', request_id, reviewed['status'])
                    approval_logger.info('Stage 9: Inline buttons removed: PASS (request_id=%s).', request_id)
                else:
                    approval_logger.warning('Stage 8: Original Telegram message edited: FAIL (request_id=%s).', request_id)
                    approval_logger.warning('Stage 9: Inline buttons removed: FAIL (request_id=%s).', request_id)
                approval_logger.info('Telegram approval message edit result (request_id=%s, status=%s, ok=%s, http_status=%s).',
                                     request_id, reviewed['status'], bool(edit_result.get('ok')), edit_result.get('http_status'))
                answer_text = 'Request #{} was already {}.'.format(request_id, reviewed['status'].lower())
            else:
                answer_text = 'Request #{} has already been reviewed.'.format(request_id)
        elif outcome['reason'] == 'not_found':
            approval_logger.warning('Stage 7: Database commit completed: FAIL (request_id=%s, reason=not_found).', request_id)
            answer_text = 'Request #{} no longer exists.'.format(request_id)
        elif outcome['reason'] == 'account_exists':
            approval_logger.warning('Stage 7: Database commit completed: FAIL (request_id=%s, reason=account_exists).', request_id)
            answer_text, answer_alert = 'Request #{} cannot be approved because an account already exists.'.format(request_id), True
        else:
            approval_logger.warning('Stage 7: Database commit completed: FAIL (request_id=%s, reason=%s).', request_id, outcome.get('reason'))
            answer_text, answer_alert = 'Request #{} could not be reviewed.'.format(request_id), True
    if callback_id:
        try:
            callback_answer = telegram_approval.answer_callback(callback_id, answer_text, answer_alert)
        except Exception as exc:
            approval_logger.warning('Telegram answerCallbackQuery failed safely (error_type=%s).', type(exc).__name__)
            callback_answer = {'ok': False, 'error': type(exc).__name__}
        if callback_answer.get('ok'):
            approval_logger.info('Stage 5: answerCallbackQuery executed: PASS.')
        else:
            approval_logger.warning('Stage 5: answerCallbackQuery executed: FAIL (error=%s).', callback_answer.get('error') or 'rejected')
        approval_logger.info('Telegram callback answer result (ok=%s, http_status=%s).',
                             bool(callback_answer.get('ok')), callback_answer.get('http_status'))
    return jsonify({'ok': True})


@app.route('/admin/users/<int:user_id>/toggle-active', methods=['POST'])
@login_required
@role_required('ADMIN')
def admin_user_toggle_active(user_id):
    enabled = request.form.get('is_active') == '1'
    conn = get_db()
    user = conn.execute('SELECT id,username,role,is_active FROM users WHERE id=?', (user_id,)).fetchone()
    if user is None:
        conn.close()
        abort(404)
    if user['role'] == 'ADMIN' and not enabled:
        active_admins = conn.execute("SELECT COUNT(*) FROM users WHERE role='ADMIN' AND is_active=1").fetchone()[0]
        if active_admins <= 1:
            conn.close()
            flash('Cannot disable the last active administrator.', 'error')
            return redirect(url_for('user_management'))
    conn.execute('UPDATE users SET is_active=? WHERE id=?', (int(enabled), user_id))
    conn.commit()
    conn.close()
    log_audit(session['user_id'], session.get('username', ''), 'USER_STATUS_CHANGED',
              'User {} changed to {}'.format(user['username'], 'ACTIVE' if enabled else 'DISABLED'),
              request.remote_addr or '127.0.0.1')
    flash('User {} is now {}.'.format(user['username'], 'ACTIVE' if enabled else 'DISABLED'), 'success')
    return redirect(url_for('user_management'))


@app.route('/admin/demo-requests/<int:request_id>/review', methods=['POST'])
@login_required
@role_required('ADMIN')
def admin_demo_request_review(request_id):
    try:
        review_token_data = form_token('demo-review').loads(request.form.get('form_token'), max_age=7200)
    except (BadSignature, SignatureExpired, TypeError):
        abort(400)
    if review_token_data.get('id') != request_id:
        abort(400)
    decision = request.form.get('decision', '').strip().upper()
    notes = request.form.get('review_notes', '').strip()[:1000]
    if decision not in ('APPROVED', 'REJECTED'):
        abort(400)
    email_action_token = request.form.get('email_action_token')
    email_nonce = None
    if email_action_token:
        try:
            email_payload = form_token('admin-email-review').loads(email_action_token, max_age=86400)
        except (BadSignature, SignatureExpired, TypeError):
            abort(400)
        if email_payload.get('id') != request_id or email_payload.get('decision') != decision or not email_payload.get('nonce'):
            abort(400)
        email_nonce = email_payload['nonce']
    try:
        outcome = _apply_demo_request_decision(request_id, decision, reviewed_by=session['user_id'],
                                               review_notes=notes, nonce=email_nonce)
    except Exception:
        logging.getLogger('safetyvision.access').exception('Could not review access request %s', request_id)
        flash('The access request could not be updated. No account was activated.', 'error')
        return redirect(url_for('admin_demo_requests'))
    if not outcome['ok']:
        if outcome['reason'] == 'not_found':
            abort(404)
        if outcome['reason'] == 'invalid_token':
            abort(400)
        if outcome['reason'] == 'account_exists':
            flash('Cannot approve: an account already uses this email address.', 'error')
        else:
            flash('This request has already been reviewed.', 'warning')
        return redirect(url_for('admin_demo_requests'))
    row = outcome['row']
    activation_token = outcome['activation_token']
    log_audit(session['user_id'], session.get('username', ''), 'DEMO_REQUEST_' + decision,
              'Demo request {} reviewed'.format(request_id), request.remote_addr or '127.0.0.1')

    email_result = _notify_decision_applicant(request_id, decision, row, activation_token)
    if email_result['ok']:
        logging.getLogger('safetyvision.access').info(
            'Access decision email accepted by Brevo (request_id=%s, decision=%s, message_id=%s).',
            request_id, decision, email_result['message_id'])
    else:
        logging.getLogger('safetyvision.access').warning(
            'Access decision saved but email was not confirmed (request_id=%s, decision=%s, reason=%s).',
            request_id, decision, email_result['error'])

    if decision == 'APPROVED':
        if activation_token:
            flash('Request approved. The applicant must set a password through the one-time link sent by email.' +
                  (' Brevo accepted the email request.' if email_result['ok'] else ' The email failed; retry it from this queue. (one-time password-setup link email failed)'),
                  'success' if email_result['ok'] else 'warning')
        else:
            flash('Request approved and active normal-user account created. ' +
                  ('Brevo accepted the sign-in email for the requester.' if email_result['ok'] else
                   'Email was not accepted by the provider; the requester can still sign in with the password set during the request.'),
                  'success' if email_result['ok'] else 'warning')
    else:
        flash('Request rejected. Historical request retained; the requester may submit a new request. ' +
              ('A notification email was accepted by the email provider.' if email_result['ok'] else
               'The rejection email was not accepted by the provider.'), 'success' if email_result['ok'] else 'warning')
    return redirect(url_for('admin_demo_requests'))


@app.route('/admin/demo-requests/<int:request_id>/activation-link', methods=['POST'])
@login_required
@role_required('ADMIN')
def admin_demo_request_activation_link(request_id):
    try:
        token_data = form_token('demo-activation-issue').loads(request.form.get('form_token'), max_age=7200)
    except (BadSignature, SignatureExpired, TypeError):
        abort(400)
    if token_data.get('id') != request_id:
        abort(400)

    conn = get_db()
    try:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute("SELECT * FROM demo_requests WHERE id=? AND status='APPROVED'", (request_id,)).fetchone()
        if row is None or row['activated_at']:
            conn.rollback()
            conn.close()
            flash('Only approved requests without an activated account can receive a password-setup link.', 'warning')
            return redirect(url_for('admin_demo_requests'))
        if conn.execute('SELECT 1 FROM users WHERE email=? COLLATE NOCASE', (row['email'].strip().lower(),)).fetchone():
            conn.rollback()
            conn.close()
            flash('An account already exists for this request; no activation link was issued.', 'warning')
            return redirect(url_for('admin_demo_requests'))
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode('utf-8')).hexdigest()
        conn.execute("UPDATE demo_requests SET activation_token_hash=?, activation_expires_at=datetime('now', '+24 hours') WHERE id=?",
                     (token_hash, request_id))
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        logging.getLogger('safetyvision.access').exception('Could not issue legacy activation link for request %s', request_id)
        flash('The password-setup link could not be issued.', 'error')
        return redirect(url_for('admin_demo_requests'))
    conn.close()
    public_base = app.config.get('PUBLIC_BASE_URL', '').rstrip('/')
    if not public_base:
        flash('The password-setup link was created but cannot be emailed until PUBLIC_BASE_URL is configured.', 'warning')
        return redirect(url_for('admin_demo_requests'))
    activation_url = public_base + url_for('demo_activate', token=raw_token)
    result = send_access_decision_email(app.config, decision='APPROVED', recipient_email=row['email'],
        recipient_name=row['name'], activation_url=activation_url)
    _record_email_outcome(request_id, 'DECISION', result)
    flash('One-time password-setup email ' + ('accepted by Brevo.' if result.get('ok') else 'failed; retry from the request queue.'),
          'success' if result.get('ok') else 'warning')
    return redirect(url_for('admin_demo_requests'))


@app.route('/admin/demo-requests/<int:request_id>/email-retry', methods=['POST'])
@login_required
@role_required('ADMIN')
def admin_demo_request_email_retry(request_id):
    category = request.form.get('category', '').strip().upper()
    if category not in ('REQUEST', 'ADMIN', 'DECISION'):
        abort(400)
    conn = get_db()
    row = conn.execute('SELECT * FROM demo_requests WHERE id=?', (request_id,)).fetchone()
    conn.close()
    if row is None:
        abort(404)
    status_column = {'REQUEST': 'request_email_status', 'ADMIN': 'admin_email_status', 'DECISION': 'decision_email_status'}[category]
    error_column = status_column.replace('_status', '_error')
    if row[status_column] != 'FAILED':
        flash('Only failed email notifications can be retried.', 'warning')
        return redirect(url_for('admin_demo_requests'))
    result = _retry_request_notification(request_id, category)
    _record_email_outcome(request_id, category, result)
    flash(('Brevo accepted the retried email.' if result.get('ok') else 'Email retry failed: ' + (result.get('error') or 'provider request failed')),
          'success' if result.get('ok') else 'warning')
    return redirect(url_for('admin_demo_requests'))


@app.route('/admin/demo-requests/<int:request_id>/telegram-retry', methods=['POST'])
@login_required
@role_required('ADMIN')
def admin_demo_request_telegram_retry(request_id):
    conn = get_db()
    row = conn.execute('SELECT status,telegram_approval_status FROM demo_requests WHERE id=?', (request_id,)).fetchone()
    conn.close()
    if row is None:
        abort(404)
    if row['status'] != 'PENDING' or row['telegram_approval_status'] not in ('UNKNOWN', 'FAILED'):
        flash('Only a pending request with an unknown or failed Telegram notification can be retried.', 'warning')
        return redirect(url_for('admin_demo_requests'))
    result = _send_telegram_demo_notification(request_id, retry_failed=True)
    flash(('Telegram accepted the demo approval notification.' if result.get('ok') else
           'Telegram notification retry failed: ' + str(result.get('error') or result.get('reason') or 'provider request failed')),
          'success' if result.get('ok') else 'warning')
    return redirect(url_for('admin_demo_requests'))


@app.route('/demo/activate/<token>', methods=['GET', 'POST'])
@limiter.limit("10 per minute; 60 per hour", methods=["POST"])
def demo_activate(token):
    token_hash = hashlib.sha256(token.encode('utf-8')).hexdigest()
    conn = get_db()
    row = conn.execute('SELECT * FROM demo_requests WHERE activation_token_hash=? AND status=\'APPROVED\' AND activated_at IS NULL AND activation_expires_at > CURRENT_TIMESTAMP', (token_hash,)).fetchone()
    if row is None:
        conn.close()
        return render_template('activate_demo.html', error='This activation link is invalid, expired, or already used.'), 400
    if request.method == 'POST':
        try:
            submitted_token_data = form_token('demo-activate').loads(request.form.get('form_token'), max_age=7200)
        except (BadSignature, SignatureExpired, TypeError):
            conn.close()
            abort(400)
        if submitted_token_data.get('token_hash') != token_hash:
            conn.close()
            abort(400)
    if request.method == 'GET':
        conn.close()
        return render_template('activate_demo.html', activation_token=token, form_token=form_token('demo-activate').dumps({'token_hash': token_hash}), csrf_token=csrf_token_value())

    if not valid_form_token('demo-activate', request.form.get('form_token')):
        conn.close()
        return render_template('activate_demo.html', error='Your form expired. Reload the page and try again.', activation_token=token, form_token=form_token('demo-activate').dumps({'token_hash': token_hash})), 400
    password = request.form.get('password', '')
    confirmation = request.form.get('confirm_password', '')
    username = request.form.get('username', '').strip()
    if not re.fullmatch(r'[A-Za-z0-9_.-]{3,40}', username) or len(password) < 12 or len(password) > 256 or password != confirmation:
        conn.close()
        return render_template('activate_demo.html', error='Choose a username (3-40 letters, numbers, dots, dashes or underscores), a password of at least 12 characters, and matching confirmation.', activation_token=token, form_token=form_token('demo-activate').dumps({'token_hash': token_hash})), 400
    email = row['email'].strip().lower()
    if conn.execute('SELECT 1 FROM users WHERE username=? COLLATE NOCASE OR email=? COLLATE NOCASE', (username, email)).fetchone():
        conn.close()
        return render_template('activate_demo.html', error='That username or email already has an account. Contact an administrator.', activation_token=token, form_token=form_token('demo-activate').dumps({'token_hash': token_hash})), 409
    try:
        cursor = conn.execute('INSERT INTO users (username,email,password_hash,role,full_name,is_active) VALUES (?,?,?,?,?,1)',
                              (username, email, generate_password_hash(password), 'OPERATOR', row['name']))
    except Exception:
        conn.rollback()
        conn.close()
        return render_template('activate_demo.html', error='The account could not be created. Please contact an administrator.', activation_token=token, form_token=form_token('demo-activate').dumps({'token_hash': token_hash})), 409
    changed = conn.execute('UPDATE demo_requests SET activated_at=CURRENT_TIMESTAMP, activation_token_hash=NULL WHERE id=? AND activated_at IS NULL AND activation_token_hash=?', (row['id'], token_hash)).rowcount
    if changed != 1:
        conn.rollback()
        conn.close()
        abort(409)
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    log_audit(user_id, username, 'DEMO_ACCOUNT_ACTIVATED', 'Approved demo account activated', request.remote_addr or '127.0.0.1')
    return render_template('activate_demo.html', success='Your account is active. You can now sign in.', activation_token=None)

@app.route('/live')
@app.route('/webcam')
@login_required
def live_monitoring():
    conn = get_db()
    zones = [dict(z) for z in conn.execute('SELECT * FROM zones WHERE is_active = 1').fetchall()]
    conn.close()

    model_info = pipeline.model_mgr.get_stats()
    return render_template(
        'live_monitoring.html',
        active_page='live',
        zones_json=zones,
        model_info=model_info
    )

@app.route('/detection')
@app.route('/detect')
@login_required
def detection_center():
    model_info = pipeline.model_mgr.get_stats()
    return render_template(
        'detection_center.html',
        active_page='detection',
        model_info=model_info
    )


# ========================================================
# 3. MONITORING PAGES
# ========================================================

@app.route('/cameras')
@login_required
def cameras():
    conn = get_db()
    cams = [dict(c) for c in conn.execute('''
        SELECT c.*, z.name as zone_name
        FROM cameras c
        LEFT JOIN zones z ON c.assigned_zone_id = z.id
    ''').fetchall()]
    zones = [dict(z) for z in conn.execute('SELECT id, name, zone_type FROM zones').fetchall()]
    conn.close()
    model_info = pipeline.model_mgr.get_stats()
    return render_template('cameras.html', active_page='cameras', cameras=cams, zones=zones, model_info=model_info)

@app.route('/zones')
@login_required
def zones():
    conn = get_db()
    z_list = [dict(z) for z in conn.execute('SELECT * FROM zones ORDER BY id DESC').fetchall()]
    conn.close()
    return render_template('zones.html', active_page='zones', zones=z_list)

@app.route('/history')
@login_required
def detection_history():
    stype = request.args.get('type', '').strip().upper()
    q = request.args.get('q', '').strip()
    date = request.args.get('date', '').strip()
    sessions = session_manager.get_all_sessions(
        session_type=stype if stype else None,
        limit=200,
        search=q if q else None,
        date=date if date else None,
        user_id=None if session.get('role') == 'ADMIN' else session.get('user_id'),
        include_unowned=session.get('role') == 'ADMIN',
    )
    return render_template(
        'history.html',
        active_page='history',
        sessions=sessions,
        current_type=stype,
        query_q=q,
        query_date=date,
    )

@app.route('/session/<session_uid>')
@login_required
def session_detail(session_uid):
    is_admin = session.get('role') == 'ADMIN'
    sess = session_manager.get_session(session_uid, user_id=None if is_admin else session.get('user_id'), include_unowned=is_admin)
    if not sess:
        flash(f'Session {session_uid} not found.', 'error')
        return redirect(url_for('detection_history'))
    # Show the annotated evidence (real YOLO boxes/labels) on the page, matching
    # the PDF. Generated lazily from stored person_results; original preserved.
    try:
        from services.report_service import resolve_annotated_evidence
        annotated = resolve_annotated_evidence(sess.get('evidence_image'), sess.get('person_results'))
        if annotated:
            sess['evidence_image'] = annotated
    except Exception:
        pass
    return render_template('session_detail.html', active_page='history', session=sess)

@app.route('/incidents')
@login_required
def incidents():
    q = request.args.get('q', '').strip()
    status = request.args.get('status', '').strip()
    severity = request.args.get('severity', '').strip()

    is_admin = session.get('role') == 'ADMIN'
    query = 'SELECT * FROM incidents WHERE 1=1'
    params = []
    if not is_admin:
        query += ' AND user_id = ?'
        params.append(session.get('user_id'))
    if q:
        query += ' AND (incident_uid LIKE ? OR violation_type LIKE ? OR person_tracking_id LIKE ?)'
        params.extend([f"%{q}%", f"%{q}%", f"%{q}%"])
    if status:
        query += ' AND status = ?'
        params.append(status)
    if severity:
        query += ' AND severity = ?'
        params.append(severity)

    query += ' ORDER BY id DESC LIMIT 50'
    conn = get_db()
    inc_list = [dict(i) for i in conn.execute(query, params).fetchall()]
    conn.close()

    return render_template(
        'incidents.html',
        active_page='incidents',
        incidents=inc_list,
        query_q=q,
        query_status=status,
        query_severity=severity
    )

@app.route('/notifications')
@login_required
def notifications():
    conn = get_db()
    if session.get('role') == 'ADMIN':
        notifs = [dict(n) for n in conn.execute('SELECT * FROM notifications ORDER BY id DESC LIMIT 40').fetchall()]
    else:
        notifs = [dict(n) for n in conn.execute('SELECT * FROM notifications WHERE user_id=? ORDER BY id DESC LIMIT 40', (session.get('user_id'),)).fetchall()]
    conn.close()
    return render_template('notifications.html', active_page='notifications', notifications=notifs)


# ========================================================
# 4. ANALYTICS & AUDIT PAGES
# ========================================================

@app.route('/analytics')
@login_required
def analytics():
    time_range = request.args.get('range', '7days')
    if time_range not in ('today', 'yesterday', '7days', '30days', 'all'):
        time_range = '7days'
    data = get_analytics_metrics(time_range, user_id=session.get('user_id'), is_admin=session.get('role') == 'ADMIN')

    return render_template(
        'analytics.html',
        active_page='analytics',
        active_range=time_range,
        stats=data['stats'],
        analytics_json=data['charts'],
        range_label=data['range_label'],
        recent_incidents=data.get('recent_incidents', []),
        trends=data.get('trends', {}),
        has_incidents=data['has_incidents'],
        has_sessions=data['has_sessions'],
    )

@app.route('/heatmap')
@login_required
def heatmap():
    conn = get_db()
    # Fetch real incidents with associated zones to synthesize floor coordinates
    if session.get('role') == 'ADMIN':
        incidents = conn.execute('SELECT id, violation_type, risk_score, zone_name FROM incidents ORDER BY id DESC LIMIT 50').fetchall()
    else:
        incidents = conn.execute('SELECT id, violation_type, risk_score, zone_name FROM incidents WHERE user_id=? ORDER BY id DESC LIMIT 50', (session.get('user_id'),)).fetchall()
    conn.close()

    # Calculate real spatial coordinates based on zone centers or normalized locations
    zone_centers = {
        'Main Walkway': (0.25, 0.45),
        'Heavy Machinery Bay': (0.72, 0.35),
        'Vehicle Transit Path': (0.72, 0.80),
        'Restricted Substation': (0.25, 0.80)
    }

    heatmap_points = []
    for inc in incidents:
        z_name = inc['zone_name']
        center = zone_centers.get(z_name, (0.5, 0.5))
        jitter_x = round(center[0] + (np.sin(inc['id']) * 0.08), 3)
        jitter_y = round(center[1] + (np.cos(inc['id']) * 0.08), 3)
        heatmap_points.append({
            'x': min(0.95, max(0.05, jitter_x)),
            'y': min(0.95, max(0.05, jitter_y)),
            'risk': inc['risk_score'],
            'violation': inc['violation_type']
        })

    return render_template(
        'heatmap.html',
        active_page='heatmap',
        heatmap_data=heatmap_points,
        heatmap_json=heatmap_points
    )

@app.route('/reports')
@login_required
def reports():
    # Functional report-library filters over REAL session data. PDF downloads
    # are untouched; only the listed rows are narrowed. Parameterized; no new
    # endpoints, no schema change.
    q = request.args.get('q', '').strip()
    stype = request.args.get('type', '').strip().upper()
    date = request.args.get('date', '').strip()
    status = request.args.get('status', '').strip().upper()  # COMPLIANT | VIOLATION | ''
    sessions = session_manager.get_all_sessions(
        session_type=stype if stype else None,
        limit=200,
        search=q if q else None,
        date=date if date else None,
        user_id=None if session.get('role') == 'ADMIN' else session.get('user_id'),
        include_unowned=session.get('role') == 'ADMIN',
    )
    if status == 'COMPLIANT':
        sessions = [s for s in sessions if (s.get('display_violation_count', s.get('violation_count', 0)) or 0) == 0]
    elif status == 'VIOLATION':
        sessions = [s for s in sessions if (s.get('display_violation_count', s.get('violation_count', 0)) or 0) > 0]
    overall_stats = session_manager.get_overall_stats(user_id=None if session.get('role') == 'ADMIN' else session.get('user_id'),
                                                       include_unowned=session.get('role') == 'ADMIN')
    return render_template('reports.html', active_page='reports', sessions=sessions,
                           stats=overall_stats, query_q=q, query_type=stype,
                           query_date=date, query_status=status,
                           report_scope='System-wide' if session.get('role') == 'ADMIN' else 'Your account')

@app.route('/reports/session/<session_uid>/download')
@login_required
def download_session_report(session_uid):
    is_admin = session.get('role') == 'ADMIN'
    owned_session = session_manager.get_session(session_uid, user_id=None if is_admin else session.get('user_id'), include_unowned=is_admin)
    if not owned_session:
        abort(404)
    # Reuse an existing report only when its path is scoped to the requesting
    # account; report basenames are randomized to avoid cross-owner overwrite.
    conn = get_db()
    if is_admin:
        report_row = conn.execute('SELECT file_path FROM generated_reports WHERE session_uid=? ORDER BY id DESC LIMIT 1', (session_uid,)).fetchone()
    else:
        report_row = conn.execute('SELECT file_path FROM generated_reports WHERE session_uid=? AND owner_id=? ORDER BY id DESC LIMIT 1',
                                  (session_uid, session.get('user_id'))).fetchone()
    conn.close()
    pdf_rel_path = report_row['file_path'] if report_row else generate_session_pdf_report(
        session_uid, user_id=None if is_admin else session.get('user_id'), is_admin=is_admin)
    if not pdf_rel_path:
        abort(404)
    full_path = os.path.join(REPORTS_FOLDER, os.path.basename(pdf_rel_path))
    conn = get_db()
    conn.execute('INSERT OR IGNORE INTO generated_reports (owner_id,file_path,report_type,session_uid) VALUES (?,?,?,?)',
                 (owned_session.get('user_id'), pdf_rel_path, 'SESSION', session_uid))
    conn.commit()
    conn.close()
    return send_file(full_path, as_attachment=True, download_name=os.path.basename(full_path))

@app.route('/reports/overall/download')
@login_required
def download_overall_report():
    is_admin = session.get('role') == 'ADMIN'
    conn = get_db()
    if is_admin:
        report_row = conn.execute("SELECT file_path FROM generated_reports WHERE report_type='OVERALL' ORDER BY id DESC LIMIT 1").fetchone()
    else:
        report_row = conn.execute("SELECT file_path FROM generated_reports WHERE report_type='OVERALL' AND owner_id=? ORDER BY id DESC LIMIT 1",
                                  (session.get('user_id'),)).fetchone()
    conn.close()
    pdf_rel_path = report_row['file_path'] if report_row else generate_overall_pdf_report(
        user_id=None if is_admin else session.get('user_id'), is_admin=is_admin)
    if not report_row:
        conn = get_db()
        conn.execute('INSERT INTO generated_reports (owner_id,file_path,report_type) VALUES (?,?,?)',
                     (session.get('user_id'), pdf_rel_path, 'OVERALL'))
        conn.commit()
        conn.close()
    full_path = os.path.join(REPORTS_FOLDER, os.path.basename(pdf_rel_path))
    return send_file(full_path, as_attachment=True, download_name=os.path.basename(full_path))


# ========================================================
# 5. AI & ADMIN PAGES
# ========================================================

@app.route('/model-performance')
@login_required
def model_performance():
    stats = pipeline.model_mgr.get_stats()
    return render_template('model_performance.html', active_page='model_performance', stats=stats)

@app.route('/model-classes')
@login_required
def model_classes():
    return render_template('model_classes.html', active_page='model_classes')

@app.route('/users')
@login_required
@role_required('ADMIN')
def user_management():
    conn = get_db()
    users_list = [dict(u) for u in conn.execute('SELECT id, username, email, role, full_name, is_active, created_at FROM users').fetchall()]
    conn.close()
    return render_template('users.html', active_page='users', users=users_list)


@app.route('/admin/users/<int:user_id>/activity')
@login_required
@role_required('ADMIN')
def admin_user_activity(user_id):
    """Explicitly scoped account view for administrator investigations."""
    conn = get_db()
    user = conn.execute('SELECT id,username,email,full_name,role,is_active FROM users WHERE id=?', (user_id,)).fetchone()
    conn.close()
    if not user:
        abort(404)
    sessions = session_manager.get_all_sessions(limit=200, user_id=user_id, include_unowned=False)
    conn = get_db()
    incidents_list = [dict(row) for row in conn.execute(
        'SELECT * FROM incidents WHERE user_id=? ORDER BY id DESC LIMIT 100', (user_id,)).fetchall()]
    conn.close()
    stats = session_manager.get_overall_stats(user_id=user_id, include_unowned=False)
    return render_template('admin_user_activity.html', active_page='users', user=dict(user),
                           sessions=sessions, incidents=incidents_list, stats=stats)

@app.route('/audit-logs')
@login_required
@role_required('ADMIN')
def audit_logs():
    conn = get_db()
    logs_list = [dict(l) for l in conn.execute('SELECT * FROM audit_logs ORDER BY id DESC LIMIT 100').fetchall()]
    conn.close()
    return render_template('audit_logs.html', active_page='audit_logs', logs=logs_list)

@app.route('/settings')
@login_required
def settings():
    # login_required reloads the role and active state from the database. Only
    # administrators reach the system settings template; ordinary users get a
    # deliberately separate, limited personal-settings view.
    if session.get('role') != 'ADMIN':
        conn = get_db()
        user = conn.execute('SELECT full_name, username, email FROM users WHERE id=?',
                            (session['user_id'],)).fetchone()
        conn.close()
        if user is None:
            abort(403)
        return render_template('user_settings.html', active_page='settings', user=user)

    if session.get('role') == 'ADMIN':
        return redirect(url_for('admin_settings'))

    conn = get_db()
    user = conn.execute('SELECT full_name, username, email FROM users WHERE id=?',
                        (session['user_id'],)).fetchone()
    conn.close()
    if user is None:
        abort(403)
    return render_template('user_settings.html', active_page='settings', user=user)


@app.route('/admin/settings')
@login_required
@role_required('ADMIN')
def admin_settings():
    """Full system settings; always protected by the administrator guard."""
    conn = get_db()
    rows = conn.execute('SELECT key, value FROM system_settings').fetchall()
    settings_dict = {r['key']: r['value'] for r in rows}
    conn.close()
    return render_template('settings.html', active_page='settings', settings=settings_dict)


@app.route('/user-settings/profile', methods=['POST'])
@login_required
def update_user_settings_profile():
    # Identity is always taken from the authenticated server session, never a
    # form-supplied user ID. Do not permit profile actions to alter privileges.
    if session.get('role') == 'ADMIN':
        abort(403)
    full_name = request.form.get('full_name', '').strip()
    if not full_name or len(full_name) > 120 or any(ord(ch) < 32 for ch in full_name):
        flash('Enter a display name of 1–120 characters.')
        return redirect(url_for('settings'))
    conn = get_db()
    conn.execute('UPDATE users SET full_name=? WHERE id=?', (full_name, session['user_id']))
    conn.commit()
    conn.close()
    session['full_name'] = full_name
    flash('Display name updated.')
    return redirect(url_for('settings'))


@app.route('/user-settings/password', methods=['POST'])
@login_required
def update_user_settings_password():
    if session.get('role') == 'ADMIN':
        abort(403)
    current_password = request.form.get('current_password', '')
    new_password = request.form.get('new_password', '')
    confirm_password = request.form.get('confirm_password', '')
    conn = get_db()
    user = conn.execute('SELECT password_hash FROM users WHERE id=?', (session['user_id'],)).fetchone()
    if (not user or not check_password_hash(user['password_hash'], current_password)
            or len(new_password) < 12 or len(new_password) > 256
            or new_password != confirm_password):
        conn.close()
        flash('Password was not changed. Check your current password and enter matching new passwords (12–256 characters).')
        return redirect(url_for('settings'))
    conn.execute('UPDATE users SET password_hash=? WHERE id=?',
                 (generate_password_hash(new_password), session['user_id']))
    conn.commit()
    conn.close()
    flash('Password updated.')
    return redirect(url_for('settings'))

@app.route('/docs')
@login_required
def documentation():
    return render_template('docs.html', active_page='docs')

@app.route('/about')
@login_required
def about():
    return render_template('about.html', active_page='about')


# ========================================================
# 6. REST APIS & AJAX PIPELINE ENDPOINTS
# ========================================================

@app.route('/api/detect/frame', methods=['POST'])
@login_required
def api_detect_frame():
    """
    Receives base64 image data from live stream or upload, runs through unified pipeline,
    and returns full analysis dictionary.
    """
    data = request.get_json()
    if not data or 'image_data' not in data:
        return jsonify({'error': 'No image_data received'}), 400

    img_data = data['image_data']
    if ',' in img_data:
        _, img_data = img_data.split(',', 1)

    try:
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Failed to decode image frame'}), 400

        owner_id = session.get('user_id')
        with _active_live_sessions_lock:
            live_uid = _active_live_sessions.get(owner_id)
            if not live_uid:
                live_uid, _ = session_manager.create_session('LIVE', 'Laptop Webcam', user_id=owner_id)
                _active_live_sessions[owner_id] = live_uid
        analysis = pipeline.process_frame(frame, auto_persist=False, user_id=owner_id)
        with _active_live_sessions_lock:
            ids_by_user = getattr(app, '_live_track_ids', {})
            observed_ids = ids_by_user.setdefault(owner_id, set())
            observed_ids.update(int(item.get('track_id', item.get('person_id'))) for item in analysis.get('person_results', [])
                                if item.get('track_id', item.get('person_id')) is not None)
            app._live_track_ids = ids_by_user
            last_frame = getattr(app, '_live_last_frames', {})
            last_frame[owner_id] = frame.copy()
            app._live_last_frames = last_frame
        analysis['session_uid'] = live_uid

        return jsonify(analysis)
    except Exception as e:
        app.logger.exception('Frame inference failed')
        return jsonify({'error': 'Frame inference failed. Please retry or contact an administrator.'}), 500

@app.route('/api/detect/image', methods=['POST'])
@limiter.limit("20 per minute; 120 per hour", methods=["POST"])
@login_required
def api_detect_image():
    """
    Dedicated Image Detection Endpoint.
    Analyzes an uploaded image, associates PPE, creates an individual session,
    annotates the image stably, and saves primary evidence.
    """
    frame = None
    filename = "uploaded_image.jpg"

    file = request.files.get('image') or request.files.get('file')
    if not file and len(request.files) > 0:
        file = next(iter(request.files.values()), None)

    if file and file.filename != '':
        if not _allowed_extension(file.filename, IMAGE_EXTENSIONS):
            return jsonify({'error': 'Unsupported image file type. Allowed: JPG, JPEG, PNG, WEBP, BMP.'}), 415
        filename = file.filename
        raw_bytes = file.read()
        file_bytes = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            # Fallback: PIL supports AVIF (via pillow_avif_plugin), WebP, PNG, JPEG, TIFF, etc.
            try:
                import io
                import PIL.Image
                try:
                    import pillow_avif
                except ImportError:
                    pass
                pil_img = PIL.Image.open(io.BytesIO(raw_bytes)).convert('RGB')
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"[IMAGE_DECODE_ERROR] PIL failed to decode {filename}: {e}")

    # Fallback to base64 image data if form field or JSON payload provided
    if frame is None:
        img_data = None
        if 'image_data' in request.form:
            img_data = request.form['image_data']
        elif request.is_json:
            data = request.get_json(silent=True)
            if data and 'image_data' in data:
                img_data = data['image_data']
        
        if img_data:
            try:
                if ',' in img_data:
                    _, img_data = img_data.split(',', 1)
                img_bytes = base64.b64decode(img_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    import io, PIL.Image
                    pil_img = PIL.Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"[IMAGE_DECODE_ERROR] Base64 decode failed: {e}")

    if frame is None:
        print(f"[DETECTION] Image received: NO (Failed to decode format for '{filename}')")
        return jsonify({
            'success': False,
            'error': f"Unsupported image format or corrupt file for '{filename}'. Supported formats: JPG, JPEG, PNG, WEBP, AVIF."
        }), 400

    h, w = frame.shape[:2]
    session_uid, _ = session_manager.create_session('IMAGE', filename, user_id=session.get('user_id'))

    conf_thresh = float(get_setting('confidence_threshold', 0.35))
    iou_thresh = float(get_setting('iou_threshold', 0.45))
    detections, _ = pipeline.model_mgr.predict(frame, conf=conf_thresh, iou=iou_thresh)

    person_dets = [d for d in detections if d['class_name'] == 'Person']
    ppe_dets = [d for d in detections if d['class_name'] in [
        'Hardhat', 'Mask', 'Safety Vest', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'
    ]]
    hazard_dets = [d for d in detections if d['class_name'] in ['machinery', 'vehicle', 'Safety Cone']]

    tracked_persons = []
    for idx, pd in enumerate(person_dets, 1):
        tp = TrackedPerson(track_id=idx, bbox=pd['bbox'])
        tracked_persons.append(tp)

    tracked_persons = pipeline.ppe_engine.associate(tracked_persons, ppe_dets)
    context_data = pipeline.context_engine.analyze(tracked_persons, hazard_dets, frame_shape=(h, w))

    for p in tracked_persons:
        pipeline.risk_engine.evaluate_person_risk(p)

    person_results = [p.to_person_result() for p in tracked_persons]

    annotated_frame = draw_person_annotations(frame, person_results, detections)
    annotated_filename = f"annotated_{session_uid}.jpg"
    annotated_rel_path = f"static/uploads/{annotated_filename}"
    annotated_abs_path = os.path.join(UPLOAD_FOLDER, annotated_filename)
    cv2.imwrite(annotated_abs_path, annotated_frame)

    session_res = session_manager.complete_session(
        session_uid=session_uid,
        person_results=person_results,
        evidence_image=annotated_rel_path,
        duration_seconds=0.0,
        source_name=filename,
        notes=f"Single-image analysis ({w}x{h})",
        user_id=session.get('user_id')
    )

    # Fire-and-forget Telegram notifications (text alert for new PPE violations
    # + PDF report via sendDocument). Runs on a background thread and never
    # blocks persistence or the HTTP response.
    _telegram_notify(session_uid, session_res)

    # Detailed debugging output mandated in Phase 4 & Phase 24
    print("\n" + "="*50)
    print(f"[DETECTION] Image received: YES | Filename: {filename} | Dimensions: {w}x{h}")
    print(f"[DETECTION] Model loaded: YES | Conf Thresh: {conf_thresh} | IoU Thresh: {iou_thresh}")
    print(f"[DETECTION] Raw detections: {len(detections)}")
    for d in detections:
        print(f"  [RAW] class={d['class_name']} ({d['class_id']}) conf={d['confidence']:.3f} bbox={d['bbox']}")
    print(f"[DETECTION] Persons: {len(person_dets)} | PPE detections: {len(ppe_dets)} | Hazards: {len(hazard_dets)}")
    for p in person_results:
        print(f"[DETECTION] Person #{p['person_id']}: Compliance={p['compliance_score']}% | Status={p['status']} | Risk={p['risk']}")
        print(f"  Hardhat: {p['hardhat_label']} ({p['hardhat_conf']}%) | Mask: {p['mask_label']} ({p['mask_conf']}%) | Vest: {p['vest_label']} ({p['vest_conf']}%)")
    print(f"[DETECTION] Summary: People={session_res['people_count']}, Compliant={session_res['compliant_count']}, Violations={session_res['violation_count']}, Overall Compliance={session_res['overall_compliance']}%")
    print("="*50 + "\n")

    return jsonify({
        'success': True,
        'session_uid': session_uid,
        'session_id': session_uid,
        'annotated_image_url': f"/{annotated_rel_path}",
        'evidence_image': f"/{annotated_rel_path}",
        'person_results': person_results,
        'people_results': person_results,
        'people': person_results,
        'people_count': session_res['people_count'],
        'compliant_count': session_res['compliant_count'],
        'violation_count': session_res['violation_count'],
        'critical_count': session_res['critical_count'],
        'overall_compliance': session_res['overall_compliance'] if session_res['people_count'] > 0 else 'N/A',
        'safety_score': session_res['safety_score'],
        'summary': {
            'people_detected': session_res['people_count'],
            'compliant': session_res['compliant_count'],
            'violations': session_res['violation_count'],
            'critical': session_res['critical_count'],
            'overall_compliance': session_res['overall_compliance'] if session_res['people_count'] > 0 else 'N/A'
        },
        'report_download_url': f"/reports/session/{session_uid}/download"
    })

@app.route('/api/detect/video', methods=['POST'])
@limiter.limit("10 per minute; 60 per hour", methods=["POST"])
@login_required
def api_detect_video():
    """
    Dedicated Video Detection Endpoint.
    Analyzes an uploaded video file, samples frames at controlled rate,
    consolidates person results across the entire video, captures one representative evidence frame,
    and creates an atomic session.
    """
    file = request.files.get('video') or request.files.get('file')
    if not file and len(request.files) > 0:
        file = next(iter(request.files.values()), None)

    if not file or file.filename == '':
        return jsonify({'error': 'No video file provided.'}), 400
    if not _allowed_extension(file.filename, VIDEO_EXTENSIONS):
        return jsonify({'error': 'Unsupported video file type. Allowed: MP4, MOV, AVI, MKV, WEBM.'}), 415

    # Cap concurrent long-running video jobs to protect GPU memory and CPU.
    if not _video_job_semaphore.acquire(blocking=False):
        return jsonify({
            'error': 'Server is busy processing video jobs. Please try again shortly.',
            'code': 429,
        }), 429

    session_uid, _ = session_manager.create_session('VIDEO', file.filename, user_id=session.get('user_id'))
    upload_ext = os.path.splitext(file.filename)[1] or '.mp4'
    temp_video_filename = f"video_{session_uid}{upload_ext}"
    temp_video_path = os.path.join(UPLOAD_FOLDER, temp_video_filename)
    file.save(temp_video_path)

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        cap.release()
        _video_job_semaphore.release()
        try:
            os.remove(temp_video_path)
        except OSError:
            pass
        return jsonify({'error': 'Could not open video file.'}), 400

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_sec = total_frames / max(video_fps, 1.0)

    sample_interval = max(1, int(video_fps // 2))
    if total_frames > 0 and (total_frames // sample_interval) > 35:
        sample_interval = max(1, total_frames // 35)

    video_tracker = PersonTracker(max_lost_frames=15, min_iou=0.20)
    conf_thresh = float(get_setting('confidence_threshold', 0.35))
    iou_thresh = float(get_setting('iou_threshold', 0.45))

    frame_idx = 0
    sampled_count = 0
    best_evidence_frame = None
    best_evidence_results = []
    best_evidence_hazards = []
    max_person_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                sampled_count += 1
                h, w = frame.shape[:2]

                detections, _ = pipeline.model_mgr.predict(frame, conf=conf_thresh, iou=iou_thresh)
                person_dets = [d for d in detections if d['class_name'] == 'Person']
                ppe_dets = [d for d in detections if d['class_name'] in [
                    'Hardhat', 'Mask', 'Safety Vest', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'
                ]]
                hazard_dets = [d for d in detections if d['class_name'] in ['machinery', 'vehicle', 'Safety Cone']]

                tracked = video_tracker.update(person_dets, current_time=frame_idx / max(video_fps, 1.0))
                tracked = pipeline.ppe_engine.associate(tracked, ppe_dets)
                pipeline.context_engine.analyze(tracked, hazard_dets, frame_shape=(h, w))

                for p in tracked:
                    pipeline.risk_engine.evaluate_person_risk(p)

                current_results = [p.to_person_result() for p in tracked]

                if len(current_results) >= max_person_count:
                    max_person_count = len(current_results)
                    best_evidence_frame = frame.copy()
                    best_evidence_results = current_results
                    best_evidence_all_dets = detections

            frame_idx += 1
    finally:
        cap.release()
        _video_job_semaphore.release()
        try:
            os.remove(temp_video_path)
        except OSError:
            pass

    consolidated_persons = []
    all_tracks = list(video_tracker.tracks.values())
    if not all_tracks and best_evidence_results:
        consolidated_persons = best_evidence_results
    else:
        for t in all_tracks:
            consolidated_persons.append(t.to_person_result())

    evidence_filename = f"evidence_{session_uid}.jpg"
    evidence_rel_path = f"static/uploads/{evidence_filename}"
    evidence_abs_path = os.path.join(UPLOAD_FOLDER, evidence_filename)

    if best_evidence_frame is not None:
        annotated_evidence = draw_person_annotations(best_evidence_frame, best_evidence_results, best_evidence_all_dets)
        cv2.imwrite(evidence_abs_path, annotated_evidence)
    else:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy, f"Session {session_uid} - 0 Persons", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(evidence_abs_path, dummy)

    session_res = session_manager.complete_session(
        session_uid=session_uid,
        person_results=consolidated_persons,
        evidence_image=evidence_rel_path,
        duration_seconds=duration_sec,
        source_name=file.filename,
        notes=f"Video analysis ({sampled_count} frames sampled out of {total_frames})",
        user_id=session.get('user_id')
    )

    # Fire-and-forget Telegram notifications (text alert for new PPE violations
    # + PDF report via sendDocument). Runs on a background thread and never
    # blocks persistence or the HTTP response.
    _telegram_notify(session_uid, session_res)

    return jsonify({
        'success': True,
        'session_uid': session_uid,
        'evidence_image_url': f"/{evidence_rel_path}",
        'duration_seconds': round(duration_sec, 1),
        'people_count': session_res['people_count'],
        'compliant_count': session_res['compliant_count'],
        'violation_count': session_res['violation_count'],
        'critical_count': session_res['critical_count'],
        'overall_compliance': session_res['overall_compliance'] if session_res['people_count'] > 0 else 'N/A',
        'safety_score': session_res['safety_score'],
        'person_results': consolidated_persons,
        'people_results': consolidated_persons,
        'people': consolidated_persons,
        'summary': {
            'people_detected': session_res['people_count'],
            'compliant': session_res['compliant_count'],
            'violations': session_res['violation_count'],
            'critical': session_res['critical_count'],
            'overall_compliance': session_res['overall_compliance'] if session_res['people_count'] > 0 else 'N/A'
        },
        'report_download_url': f"/reports/session/{session_uid}/download"
    })

@app.route('/api/video/start', methods=['POST'])
@limiter.limit("10 per minute; 60 per hour", methods=["POST"])
@login_required
def api_video_start():
    """
    Initializes a new atomic Live AI Video Detection Session.
    """
    data = request.get_json(silent=True) or request.form or {}
    filename = data.get('filename') or 'uploaded_video.mp4'
    duration = float(data.get('duration_seconds', 0.0))
    session_uid = video_service.start_session(filename, duration, user_id=session.get('user_id'))
    return jsonify({
        'success': True,
        'session_uid': session_uid
    })

@app.route('/api/video/analyze_frame', methods=['POST'])
@limiter.limit("30 per minute", methods=["POST"])
@login_required
def api_video_analyze_frame():
    """
    Performs real-time YOLO inference, multi-person tracking, spatial PPE association,
    and state-change timeline tracking on a playing video frame.
    """
    data = request.get_json(silent=True) or request.form or {}
    session_uid = data.get('session_uid')
    if not session_uid:
        return jsonify({'success': False, 'error': 'Missing session_uid'}), 400

    img_data = data.get('image_data')
    if not img_data:
        return jsonify({'success': False, 'error': 'No image_data received'}), 400

    if ',' in img_data:
        _, img_data = img_data.split(',', 1)

    try:
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'success': False, 'error': 'Failed to decode video frame'}), 400

        video_time = float(data.get('video_time', 0.0))
        result = video_service.analyze_frame(session_uid, frame, video_time, user_id=session.get('user_id'))
        return jsonify(result)
    except Exception as e:
        if isinstance(e, PermissionError):
            return jsonify({'success': False, 'error': 'Video session is not available for this account.'}), 404
        app.logger.exception('Video analysis failed')
        return jsonify({'success': False, 'error': 'Video analysis failed.'}), 500

@app.route('/api/video/complete', methods=['POST'])
@limiter.limit("10 per minute; 60 per hour", methods=["POST"])
@login_required
def api_video_complete():
    """
    Finalizes live video analysis session, consolidates all observed persons,
    computes final safety score, saves representative evidence, and registers atomic report.
    """
    data = request.get_json(silent=True) or request.form or {}
    session_uid = data.get('session_uid')
    if not session_uid:
        return jsonify({'success': False, 'error': 'Missing session_uid'}), 400

    duration_seconds = float(data.get('duration_seconds', 0.0))
    filename = data.get('filename', 'uploaded_video.mp4')
    client_evidence = data.get('evidence_image')

    try:
        result = video_service.complete_session(
            session_uid=session_uid,
            duration_seconds=duration_seconds,
            filename=filename,
            client_evidence_b64=client_evidence,
            user_id=session.get('user_id')
        )
        # Fire-and-forget Telegram notifications (text alert for new PPE
        # violations + PDF report via sendDocument). Non-blocking.
        if result.get('success'):
            _telegram_notify(session_uid, result)
        return jsonify(result)
    except Exception as e:
        if isinstance(e, PermissionError):
            return jsonify({'success': False, 'error': 'Video session is not available for this account.'}), 404
        app.logger.exception('Video session completion failed')
        return jsonify({'success': False, 'error': 'Session completion failed.'}), 500

@app.route('/api/session/live_complete', methods=['POST'])
@login_required
def api_session_live_complete():
    """
    Finalizes an active live monitoring session, saves primary evidence frame,
    and returns report link.
    """
    data = request.get_json(silent=True) or {}
    owner_id = session.get('user_id')
    try:
        duration_seconds = max(0.0, min(float(data.get('duration_seconds', 0.0)), 86400.0))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Invalid session duration.'}), 400

    logger.info("[LIVE] live_complete received | owner_id=%s | duration=%.1fs",
                owner_id, duration_seconds)

    # ---------------------------------------------------------------
    # Finalize the server-created preview session; never trust the client to
    # choose an owner, session ID, observed people, PPE states, or incidents.
    # ---------------------------------------------------------------
    with _active_live_sessions_lock:
        session_uid = _active_live_sessions.pop(owner_id, None)
        frames_by_user = getattr(app, '_live_last_frames', {})
        latest_frame = frames_by_user.pop(owner_id, None)
        app._live_last_frames = frames_by_user
        ids_by_user = getattr(app, '_live_track_ids', {})
        tracker_ids = ids_by_user.pop(owner_id, set())
        app._live_track_ids = ids_by_user
    if session_uid:
        existing_session = session_manager.get_session(session_uid, user_id=owner_id)
        if not existing_session or existing_session.get('status') != 'IN_PROGRESS':
            return jsonify({'success': False, 'error': 'Live session is not available for this account.'}), 404
    else:
        session_uid, _ = session_manager.create_session('LIVE', 'Laptop Integrated Webcam', user_id=owner_id)
    person_results = pipeline.get_live_session_person_results(owner_id, session_uid=session_uid, tracker_ids=tracker_ids)
    frame = latest_frame
    logger.info("[LIVE] session created | %s", session_uid)

    evidence_filename = f"evidence_{session_uid}.jpg"
    evidence_rel_path = f"static/uploads/{evidence_filename}"
    evidence_abs_path = os.path.join(UPLOAD_FOLDER, evidence_filename)

    # Annotated copy kept alongside the original; the report embeds the
    # annotated version so real YOLO boxes/labels are preserved.
    annotated_filename = f"annotated_{session_uid}.jpg"
    annotated_rel_path = f"static/uploads/{annotated_filename}"
    annotated_abs_path = os.path.join(UPLOAD_FOLDER, annotated_filename)

    if frame is not None:
        # Preserve the original captured frame untouched (audit/history).
        cv2.imwrite(evidence_abs_path, frame)
        # Draw ALL real detections on the captured frame. The per-person results
        # from the live loop may omit a valid PPE box that anatomical association
        # dropped (e.g. NO-Mask whose center sits below the strict face zone), so
        # run the existing inference once on this frame and pass the raw
        # detections to the annotator, which draws every PPE/hazard box present.
        # This is a single inference on one frame (same as image/video routes);
        # it does not alter counts, thresholds, weights, or association logic.
        try:
            conf_thresh = float(get_setting('confidence_threshold', 0.35))
            iou_thresh = float(get_setting('iou_threshold', 0.45))
            raw_dets, _ = pipeline.model_mgr.predict(frame, conf=conf_thresh, iou=iou_thresh)
            annotated = draw_person_annotations(frame, person_results, raw_dets)
            cv2.imwrite(annotated_abs_path, annotated)
            report_evidence_rel = annotated_rel_path
            logger.info("[LIVE] annotated evidence saved | %s | persons=%d | raw_dets=%d",
                        annotated_rel_path, len(person_results), len(raw_dets or []))
        except Exception as e:
            # Never block the session on annotation; fall back to original.
            logger.warning("[LIVE] annotation failed for %s: %s", session_uid, e)
            report_evidence_rel = evidence_rel_path
    else:
        # No frame supplied: write a safe placeholder so reports stay valid.
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, f"Session {session_uid} - Live Webcam", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(evidence_abs_path, blank)
        report_evidence_rel = evidence_rel_path
        logger.warning("[LIVE] no image data for %s; placeholder evidence written", session_uid)

    session_res = session_manager.complete_session(
        session_uid=session_uid,
        person_results=person_results,
        evidence_image=report_evidence_rel,
        duration_seconds=duration_seconds,
        source_name="Laptop Webcam",
        notes="Live webcam monitoring session",
        user_id=owner_id
    )
    pipeline.reset_tracker(user_id=owner_id)
    logger.info("[LIVE] session finalized | %s | people=%d | violations=%d | compliant=%d | overall=%s",
                session_uid, session_res.get('people_count'), session_res.get('violation_count'),
                session_res.get('compliant_count'), session_res.get('overall_compliance'))

    # ---------------------------------------------------------------
    # 2. Generate the PDF and verify it exists BEFORE notifying.
    # ---------------------------------------------------------------
    pdf_rel = None
    pdf_abs = None
    try:
        pdf_rel = generate_session_pdf_report(session_uid, user_id=owner_id, is_admin=False)
        if pdf_rel:
            pdf_abs = os.path.join(app.root_path, pdf_rel)
            conn = get_db()
            conn.execute('INSERT OR IGNORE INTO generated_reports (owner_id,file_path,report_type,session_uid) VALUES (?,?,?,?)',
                         (owner_id, pdf_rel, 'SESSION', session_uid))
            conn.commit()
            conn.close()
            pdf_ok = os.path.exists(pdf_abs)
            logger.info("[LIVE] PDF generated | %s | exists=%s | bytes=%s",
                        pdf_rel, pdf_ok,
                        os.path.getsize(pdf_abs) if pdf_ok else 'n/a')
        else:
            logger.warning("[LIVE] PDF generation returned None for %s", session_uid)
    except Exception as e:
        logger.warning("[LIVE] PDF generation error for %s: %s", session_uid, e)

    # ---------------------------------------------------------------
    # 3. Fire-and-forget Telegram delivery (summary text + PDF doc).
    #    Runs on a background thread; can never block the response or
    #    lose the persisted session/evidence/report.
    # ---------------------------------------------------------------
    _telegram_notify(session_uid, session_res)
    logger.info("[LIVE] Telegram delivery queued | %s | people=%d | violations=%d",
                session_uid, session_res.get('people_count'), session_res.get('violation_count'))

    return jsonify({
        'success': True,
        'session_uid': session_uid,
        'evidence_image_url': f"/{evidence_rel_path}",
        'annotated_evidence_url': f"/{annotated_rel_path}" if os.path.exists(annotated_abs_path) else None,
        'report_download_url': f"/reports/session/{session_uid}/download",
        'session_res': session_res
    })

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    """
    Preserved existing snap upload route, now integrated into SafetyVision AI pipeline.
    """
    data = request.get_json()
    if not data or 'image_data' not in data:
        return jsonify({'message': 'No image data received'}), 400

    image_data = data.get('image_data')
    if "," in image_data:
        _, image_data = image_data.split(',', 1)

    try:
        image_bytes = base64.b64decode(image_data)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        filename = f'snap_{timestamp}.png'
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        with open(filepath, 'wb') as f:
            f.write(image_bytes)

        # Run pipeline inference on saved snapshot, attributed to the active account.
        frame = cv2.imread(filepath)
        if frame is not None:
            pipeline.process_frame(frame, auto_persist=True, user_id=session.get('user_id'))

        return jsonify({'message': f'Snap saved as {filename}', 'filename': filename})
    except Exception as e:
        return jsonify({'message': f'Failed to save image: {str(e)}'}), 500

@app.route('/api/zones/create', methods=['POST'])
@login_required
@role_required('ADMIN')
def api_zones_create():
    name = request.form.get('name')
    zone_type = request.form.get('zone_type', 'ENTRY')
    desc = request.form.get('description', '')
    ppe_list = request.form.getlist('ppe')
    coords = request.form.get('polygon_coords', '[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]')
    risk_mult = float(request.form.get('risk_multiplier', 1.0))

    conn = get_db()
    conn.execute('''
        INSERT INTO zones (name, zone_type, description, polygon_coords, required_ppe, risk_multiplier)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (name, zone_type, desc, coords, json.dumps(ppe_list), risk_mult))
    conn.commit()
    conn.close()

    log_audit(session.get('user_id'), session.get('username'), 'CREATE_ZONE', f"Created zone {name}")
    return redirect(url_for('zones'))

@app.route('/api/zones/<int:zone_id>/delete', methods=['POST'])
@login_required
@role_required('ADMIN')
def api_zones_delete(zone_id):
    conn = get_db()
    conn.execute('DELETE FROM zones WHERE id = ?', (zone_id,))
    conn.commit()
    conn.close()

    log_audit(session.get('user_id'), session.get('username'), 'DELETE_ZONE', f"Deleted zone #{zone_id}")
    return jsonify({'success': True})

@app.route('/api/cameras/create', methods=['POST'])
@login_required
@role_required('ADMIN')
def api_cameras_create():
    name = request.form.get('name')
    location = request.form.get('location')
    stream_source = request.form.get('stream_source', 'webcam')
    rtsp_url = request.form.get('rtsp_url', '')
    zone_id = request.form.get('assigned_zone_id')

    conn = get_db()
    conn.execute('''
        INSERT INTO cameras (name, location, stream_source, rtsp_url, assigned_zone_id)
        VALUES (?, ?, ?, ?, ?)
    ''', (name, location, stream_source, rtsp_url, zone_id))
    conn.commit()
    conn.close()

    log_audit(session.get('user_id'), session.get('username'), 'CREATE_CAMERA', f"Registered camera {name}")
    return redirect(url_for('cameras'))

@app.route('/api/cameras/<int:camera_id>/delete', methods=['POST'])
@login_required
@role_required('ADMIN')
def api_cameras_delete(camera_id):
    conn = get_db()
    conn.execute('DELETE FROM cameras WHERE id = ?', (camera_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/incidents/update-status', methods=['POST'])
@login_required
@role_required('ADMIN')
def api_incidents_update_status():
    inc_id = request.form.get('incident_id')
    status = request.form.get('status')
    notes = request.form.get('resolution_notes', '')
    allowed_statuses = {'DETECTED', 'ACKNOWLEDGED', 'IN_PROGRESS', 'RESOLVED', 'FALSE_POSITIVE'}
    try:
        inc_id = int(inc_id)
    except (TypeError, ValueError):
        abort(400)
    if status not in allowed_statuses or len(notes) > 2000:
        abort(400)

    conn = get_db()
    resolved_at = datetime.now() if status == 'RESOLVED' else None
    is_admin = session.get('role') == 'ADMIN'
    conn.execute('''
        UPDATE incidents
        SET status = ?, resolution_notes = ?, operator = ?, resolved_at = ?
        WHERE id = ? AND (? = 1 OR user_id = ?)
    ''', (status, notes, session.get('username'), resolved_at, inc_id, int(is_admin), session.get('user_id')))
    conn.commit()
    conn.close()

    log_audit(session.get('user_id'), session.get('username'), 'UPDATE_INCIDENT', f"Incident #{inc_id} status updated to {status}")
    return redirect(url_for('incidents'))

@app.route('/api/incidents/export-csv')
@login_required
def api_incidents_export_csv():
    # Export honours the same filters as the Incidents page (q/status/severity)
    # so users download exactly the rows they see. Falls back to the full set
    # when no filters are supplied. Parameterized; no schema change.
    q = request.args.get('q', '').strip()
    status = request.args.get('status', '').strip()
    severity = request.args.get('severity', '').strip()
    query = 'SELECT * FROM incidents WHERE 1=1'
    params = []
    if session.get('role') != 'ADMIN':
        query += ' AND user_id = ?'
        params.append(session.get('user_id'))
    if q:
        query += ' AND (incident_uid LIKE ? OR violation_type LIKE ? OR person_tracking_id LIKE ?)'
        params.extend(["%%%s%%" % q, "%%%s%%" % q, "%%%s%%" % q])
    if status:
        query += ' AND status = ?'
        params.append(status)
    if severity:
        query += ' AND severity = ?'
        params.append(severity)
    query += ' ORDER BY id DESC'
    conn = get_db()
    rows = conn.execute(query, params).fetchall()
    conn.close()

    output = io.StringIO()
    output.write("id,incident_uid,timestamp,person_tracking_id,violation_type,confidence,duration_seconds,zone_name,risk_score,severity,status,operator\n")
    for r in rows:
        output.write(f"{r['id']},{r['incident_uid']},{r['timestamp']},{r['person_tracking_id']},{r['violation_type']},{r['confidence']},{r['duration_seconds']},{r['zone_name']},{r['risk_score']},{r['severity']},{r['status']},{r['operator']}\n")

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"SafetyVision_Incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

@app.route('/api/notifications/mark-all-read', methods=['POST'])
@login_required
def api_notifications_mark_all_read():
    conn = get_db()
    if session.get('role') == 'ADMIN':
        conn.execute('UPDATE notifications SET is_read = 1')
    else:
        conn.execute('UPDATE notifications SET is_read=1 WHERE user_id=?', (session.get('user_id'),))
    conn.commit()
    conn.close()
    return redirect(url_for('notifications'))

@app.route('/api/reports/generate', methods=['POST'])
@login_required
def api_reports_generate():
    inspector = session.get('full_name', 'Lead Safety Auditor')
    is_admin = session.get('role') == 'ADMIN'
    report_rel_path = generate_pdf_report(inspector=inspector, user_id=session.get('user_id'), is_admin=is_admin)
    conn = get_db()
    conn.execute('INSERT OR IGNORE INTO generated_reports (owner_id,file_path,report_type) VALUES (?,?,?)',
                 (session.get('user_id'), report_rel_path, 'OVERALL'))
    conn.commit()
    conn.close()
    log_audit(session.get('user_id'), session.get('username'), 'GENERATE_REPORT', f"Generated report {report_rel_path}")
    return redirect(url_for('reports'))

@app.route('/api/settings/update', methods=['POST'])
@login_required
@role_required('ADMIN')
def api_settings_update():
    conn = get_db()
    for key, value in request.form.items():
        conn.execute('UPDATE system_settings SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE key = ?', (value, key))
    conn.commit()
    conn.close()

    log_audit(session.get('user_id'), session.get('username'), 'UPDATE_SETTINGS', 'System threshold settings updated')
    flash('Settings updated successfully.')
    return redirect(url_for('settings'))


# ── Admin: Reset Detection Data ──────────────────────────────────────────────

@app.route('/api/admin/reset-detection-data', methods=['POST'])
@login_required
@role_required('ADMIN')
def api_admin_reset_detection_data():
    """
    Permanently deletes all application-generated detection data.

    DELETED tables (detection data only):
      incidents, evidence, notifications, detection_sessions, inspection_sessions (if present)
    Evidence image FILES on disk are also removed.

    PRESERVED (never touched):
      users, system_settings, cameras, zones, audit_logs, model weights, source code.
    """
    confirm = request.form.get('confirm_reset', '').strip()
    if confirm != 'RESET':
        return jsonify({'success': False, 'error': 'Confirmation phrase mismatch. Type RESET exactly.'}), 400

    # 1. Create a DB backup before any destructive operation
    import shutil
    backup_dir = os.path.join(app.root_path, 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    from database.db import DB_PATH
    backup_path = os.path.join(backup_dir, f'safetyvision_pre_reset_{ts}.db')
    try:
        shutil.copy2(DB_PATH, backup_path)
    except Exception as e:
        return jsonify({'success': False, 'error': f'DB backup failed: {e}. Reset aborted.'}), 500

    # 2. Collect evidence image paths before deletion
    conn = get_db()
    try:
        ev_inc = conn.execute(
            "SELECT evidence_image FROM incidents WHERE evidence_image IS NOT NULL AND evidence_image != ''"
        ).fetchall()
        ev_tbl = conn.execute(
            "SELECT image_path FROM evidence WHERE image_path IS NOT NULL AND image_path != ''"
        ).fetchall()
        ev_paths = [r[0] for r in ev_inc] + [r[0] for r in ev_tbl]

        # 3. Delete rows in FK-safe order inside a transaction
        conn.execute('BEGIN')
        conn.execute('DELETE FROM notifications')
        conn.execute('DELETE FROM evidence')
        conn.execute('DELETE FROM incidents')
        conn.execute('DELETE FROM detection_sessions')
        has_insp = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='inspection_sessions'"
        ).fetchone()
        if has_insp:
            conn.execute('DELETE FROM inspection_sessions')

        # Reset auto-increment counters
        for tbl in ['incidents', 'evidence', 'notifications', 'detection_sessions', 'inspection_sessions']:
            conn.execute('DELETE FROM sqlite_sequence WHERE name=?', (tbl,))

        conn.execute('COMMIT')
    except Exception as e:
        try:
            conn.execute('ROLLBACK')
        except Exception:
            pass
        conn.close()
        return jsonify({'success': False, 'error': f'DB reset failed and rolled back: {e}'}), 500

    conn.close()

    # 4. Remove evidence image files from disk
    deleted_files = 0
    failed_files = []
    for rel_path in set(ev_paths):
        abs_path = os.path.join(app.root_path, rel_path)
        if os.path.isfile(abs_path):
            try:
                os.remove(abs_path)
                deleted_files += 1
            except Exception as ex:
                failed_files.append(str(ex))

    # Sweep evidence folder for any orphaned files not tracked in DB
    try:
        for fname in os.listdir(EVIDENCE_FOLDER):
            fp = os.path.join(EVIDENCE_FOLDER, fname)
            if os.path.isfile(fp) and fname.startswith('evidence_'):
                try:
                    os.remove(fp)
                    deleted_files += 1
                except Exception as ex:
                    failed_files.append(str(ex))
    except Exception:
        pass

    # 5. Reset in-memory tracker state
    try:
        pipeline.reset_tracker()
    except Exception:
        pass

    # 6. Audit log
    details = (
        f"Full detection data reset. Backup: {os.path.basename(backup_path)}. "
        f"Evidence files removed: {deleted_files}."
    )
    if failed_files:
        details += f" File delete warnings: {len(failed_files)}."
    log_audit(
        session.get('user_id'),
        session.get('username'),
        'ADMIN_RESET_DETECTION_DATA',
        details,
        ip_address=request.remote_addr or '127.0.0.1'
    )

    return jsonify({
        'success': True,
        'message': 'All detection data has been permanently deleted.',
        'backup': os.path.basename(backup_path),
        'evidence_files_removed': deleted_files,
        'file_warnings': failed_files[:5]
    })


# ─── JSON READ API ─────────────────────────────────────────────────────────────


@app.route('/api/about')
def api_about():
    try:
        conn = get_db()
        conn.close()
        return jsonify({
            'status': 'ok',
            'backend': 'safetyvision-flask',
            'version': '1.0.0',
            'session_store': 'sqlite_detected',
        })
    except Exception as e:
        app.logger.exception('Public about health check failed')
        return jsonify({'status': 'error', 'error': 'Health check unavailable.'}), 500


@app.route('/api/detection-history', methods=['GET'])
@login_required
def api_detection_history():
    try:
        is_admin = session.get('role') == 'ADMIN'
        sessions = session_manager.get_all_sessions(limit=200,
            user_id=None if is_admin else session.get('user_id'), include_unowned=is_admin)
        rows = []
        for s in sessions:
            rows.append({
                'id': s.get('session_uid') or s.get('id') or f"SESS-{s.get('id')}",
                'time': s.get('start_time', '')[:19].replace('T', ' ') if s.get('start_time') else '—',
                'source': s.get('source_name') or (s.get('session_type', '').upper() if s.get('session_type') else '—'),
                'file': s.get('evidence_image') or s.get('session_uid') or '—',
                'people': s.get('people_count') if s.get('people_count') is not None else '—',
                'violations': s.get('violation_count') if s.get('violation_count') is not None else '—',
                'status': s.get('status') or 'UNKNOWN',
            })
        return jsonify(rows)
    except Exception:
        app.logger.exception('Detection history retrieval failed')
        return jsonify({'error': 'Detection history is temporarily unavailable.'}), 500


@app.route('/api/users/create', methods=['POST'])
@login_required
@role_required('ADMIN')
def api_users_create():
    full_name = request.form.get('full_name', '').strip()
    username = request.form.get('username', '').strip()
    email = request.form.get('email', '').strip().lower()
    role = request.form.get('role', 'OPERATOR').strip().upper()
    password = request.form.get('password', '')
    if (not full_name or len(full_name) > 120 or not re.fullmatch(r'[A-Za-z0-9_.-]{3,40}', username)
            or not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email) or len(email) > 254
            or role not in ('OPERATOR', 'SAFETY_OFFICER') or len(password) < 12 or len(password) > 256):
        flash('Enter a valid name, username, email, normal-user role, and password of at least 12 characters.', 'error')
        return redirect(url_for('user_management'))

    conn = get_db()
    try:
        conn.execute('''
            INSERT INTO users (username, email, password_hash, role, full_name)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, email, generate_password_hash(password), role, full_name))
        conn.commit()
    except Exception:
        flash('Could not create user. Check that username and email are not already in use.', 'error')
    finally:
        conn.close()

    log_audit(session.get('user_id'), session.get('username'), 'CREATE_USER', f"Created user {username} ({role})")
    return redirect(url_for('user_management'))

@app.route('/models/<path:filename>')
@login_required
def serve_models(filename):
    return send_from_directory('models', filename)


# ========================================================
# 6.5. HEALTH CHECK
# ========================================================

@app.route('/health', methods=['GET'])
def health():
    """Lightweight health-check endpoint for load balancers and monitoring."""
    db_ok = False
    db_error = None
    try:
        conn = get_db()
        conn.execute('SELECT 1').fetchone()
        conn.close()
        db_ok = True
    except Exception as exc:  # never leak internals
        db_error = type(exc).__name__
    model_device = 'unknown'
    try:
        model_device = pipeline.model_mgr.device.upper()
    except Exception:
        pass
    status = 'ok' if db_ok else 'degraded'
    return jsonify({
        'status': status,
        'database': 'connected' if db_ok else 'disconnected',
        'device': model_device,
        'app': 'safetyvision-ai',
    }), 200 if db_ok else 503


# ========================================================
# 7. SERVER RUNNER
# ========================================================

if __name__ == '__main__':
    print("==================================================")
    print("🛡️ SAFETYVISION AI - Production Server Starting")
    print(f"Device: {pipeline.model_mgr.device.upper()} ({pipeline.model_mgr.device_name})")
    print(f"Model: {pipeline.model_mgr.model_path}")
    print("Classes: 10 Supported Classes Verified")
    print("==================================================")
    app.run(host='0.0.0.0', port=5000, debug=False)
