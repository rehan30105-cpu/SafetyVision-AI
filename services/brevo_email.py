"""Safe Brevo transactional email client for SafetyVision AI."""

import html
import logging
import re
from urllib.parse import urlsplit

import requests


BREVO_SEND_URL = 'https://api.brevo.com/v3/smtp/email'
BREVO_TIMEOUT_SECONDS = 10
logger = logging.getLogger('safetyvision.email')


def _valid_absolute_url(value):
    try:
        parsed = urlsplit(value or '')
        return parsed.scheme in ('http', 'https') and bool(parsed.netloc) and not parsed.username and not parsed.password
    except (TypeError, ValueError):
        return False


def send_transactional_email(config, *, recipient_email, recipient_name, subject,
                             text_content, html_content):
    if not isinstance(recipient_email, str) or not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', recipient_email):
        return {'ok': False, 'message_id': None, 'error': 'Recipient email address is invalid.'}
    api_key = (config.get('BREVO_API_KEY') or '').strip()
    sender_email = (config.get('MAIL_FROM_EMAIL') or '').strip()
    sender_name = (config.get('MAIL_FROM_NAME') or '').strip()
    admin_email = (config.get('ADMIN_EMAIL') or '').strip()
    if not all((api_key, sender_email, sender_name, admin_email)):
        return {'ok': False, 'message_id': None, 'error': 'Email configuration is incomplete.'}
    payload = {
        'sender': {'email': sender_email, 'name': sender_name},
        'to': [{'email': recipient_email, 'name': (recipient_name or '')[:120]}],
        'replyTo': {'email': admin_email}, 'subject': str(subject)[:200],
        'textContent': str(text_content)[:50000], 'htmlContent': str(html_content)[:100000],
    }
    return _post_brevo(api_key, payload)


def send_access_decision_email(config, *, decision, recipient_email, recipient_name,
                               activation_url=None):
    decision = (decision or '').strip().upper()
    base_url = (config.get('PUBLIC_BASE_URL') or '').strip().rstrip('/')
    if decision not in ('APPROVED', 'REJECTED'):
        return {'ok': False, 'message_id': None, 'error': 'Unsupported access decision.'}
    if not _valid_absolute_url(base_url):
        return {'ok': False, 'message_id': None, 'error': 'Public website URL is invalid.'}
    name = (recipient_name or 'there').strip()[:120]
    if decision == 'APPROVED':
        if activation_url:
            if not _valid_absolute_url(activation_url):
                return {'ok': False, 'message_id': None, 'error': 'The account setup link is invalid.'}
            intro = 'Your request is approved. Set a password using this secure one-time link. It expires in 24 hours and can only be used once.'
            link, label = activation_url, 'Set your password'
        else:
            intro = 'Your request is approved and your account is active. Sign in using your request email and the password you chose during the request.'
            link, label = base_url + '/login', 'Sign in to SafetyVision AI'
        subject = 'Your SafetyVision AI access request was approved'
        text = 'Hello {},\n\n{}\n\n{}: {}'.format(name, intro, label, link)
        action_html = '<p><a href="{}">{}</a></p>'.format(html.escape(link, quote=True), html.escape(label))
    else:
        subject = 'Update on your SafetyVision AI access request'
        intro = 'Your access request was not approved at this time. You may contact the administrator for more information or submit a new request.'
        text = 'Hello {},\n\n{}\n\n{}'.format(name, intro, base_url)
        action_html = '<p><a href="{}">Submit a new request</a></p>'.format(html.escape(base_url, quote=True))
    html_content = '<!doctype html><html><body><p>Hello {},</p><p>{}</p>{}<p>SafetyVision AI</p></body></html>'.format(
        html.escape(name), html.escape(intro), action_html)
    return send_transactional_email(config, recipient_email=recipient_email, recipient_name=name,
                                    subject=subject, text_content=text, html_content=html_content)


def _post_brevo(api_key, payload):
    try:
        response = requests.post(BREVO_SEND_URL,
            headers={'api-key': api_key, 'accept': 'application/json', 'content-type': 'application/json'},
            json=payload, timeout=BREVO_TIMEOUT_SECONDS)
    except requests.Timeout:
        logger.warning('Brevo transactional email timed out.')
        return {'ok': False, 'message_id': None, 'error': 'Email provider timed out.'}
    except requests.RequestException as exc:
        logger.warning('Brevo transactional email request failed (error_type=%s).', type(exc).__name__)
        return {'ok': False, 'message_id': None, 'error': 'Email provider request failed.'}
    except Exception as exc:
        logger.warning('Brevo email integration failed safely (error_type=%s).', type(exc).__name__)
        return {'ok': False, 'message_id': None, 'error': 'Email provider request failed.'}
    if response.status_code != 201:
        if response.status_code in (401, 403):
            # Brevo authenticates this endpoint with the account API key. Keep
            # provider bodies private, but distinguish credential rejection
            # from delivery/application failures for the admin queue.
            logger.warning('Brevo rejected email credentials (http_status=%s); verify the configured API key and account access.', response.status_code)
            return {'ok': False, 'message_id': None,
                    'error': 'Brevo rejected the configured API credential (HTTP {}). Verify the API key is current and has transactional-email access.'.format(response.status_code)}
        reason = 'rejected' if 400 <= response.status_code < 500 else 'unavailable'
        logger.warning('Brevo did not accept transactional email (status=%s).', response.status_code)
        return {'ok': False, 'message_id': None,
                'error': 'Email provider {} the message (HTTP {}).'.format(reason, response.status_code)}
    try:
        message_id = response.json().get('messageId')
    except (ValueError, AttributeError):
        message_id = None
    if not isinstance(message_id, str) or not message_id.strip():
        logger.warning('Brevo returned HTTP 201 without a message identifier.')
        return {'ok': False, 'message_id': None, 'error': 'Provider response did not confirm acceptance.'}
    return {'ok': True, 'message_id': message_id.strip(), 'error': None}