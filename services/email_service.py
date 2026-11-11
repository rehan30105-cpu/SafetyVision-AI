"""Small SMTP email helper configured entirely through environment variables."""
import os
import smtplib
import ssl
import logging
from email.message import EmailMessage
from email.utils import parseaddr

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
except Exception:
    pass

logger = logging.getLogger("safetyvision.email")


def send_email(recipient, subject, body):
    host = os.environ.get('SMTP_HOST', '').strip()
    sender = os.environ.get('SMTP_FROM', '').strip()
    username = os.environ.get('SMTP_USERNAME', '').strip()
    if not host or not sender:
        logger.warning("Email delivery is not configured; recipient=%s subject=%s", recipient, subject)
        return False, 'Email delivery is not configured (SMTP_HOST and SMTP_FROM are required).'
    if not parseaddr(sender)[1] or '@' not in parseaddr(sender)[1]:
        logger.error("Email delivery configuration has an invalid SMTP_FROM value")
        return False, 'Email delivery is misconfigured (SMTP_FROM must be an email address).'
    if username and (not parseaddr(username)[1] or '@' not in parseaddr(username)[1]):
        logger.error("Email delivery configuration has an invalid SMTP_USERNAME value")
        return False, 'Email delivery is misconfigured (SMTP_USERNAME must be the full mailbox address).'

    stage = 'configuration'
    try:
        port = int(os.environ.get('SMTP_PORT', '587'))
        timeout = int(os.environ.get('SMTP_TIMEOUT', '15'))
        message = EmailMessage()
        message['From'] = sender
        message['To'] = recipient
        message['Subject'] = subject
        message.set_content(body)
        context = ssl.create_default_context()
        stage = 'connection'
        with smtplib.SMTP(host, port, timeout=timeout) as client:
            stage = 'EHLO before STARTTLS'
            client.ehlo()
            if os.environ.get('SMTP_STARTTLS', 'true').lower() in ('1', 'true', 'yes', 'on'):
                stage = 'STARTTLS negotiation'
                client.starttls(context=context)
                stage = 'EHLO after STARTTLS'
                client.ehlo()
            password = os.environ.get('SMTP_PASSWORD', '')
            if username:
                stage = 'authentication'
                client.login(username, password)
            stage = 'message submission'
            client.send_message(message)
        return True, 'Email sent.'
    except (OSError, ValueError, smtplib.SMTPException) as exc:
        logger.error(
            "Email delivery failed at stage=%s; exception=%s; smtp_code=%s",
            stage,
            type(exc).__name__,
            getattr(exc, 'smtp_code', None),
        )
        return False, 'Email delivery failed: {}'.format(type(exc).__name__)