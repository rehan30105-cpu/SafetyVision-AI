"""Telegram transport dedicated to private demo-request approval messages.

This module intentionally does not import or call the PPE report notifier. It
never sends documents or media; the Flask webhook handles callbacks separately.
"""
import logging
import re

import requests


logger = logging.getLogger('safetyvision.telegram_approval')
API_BASE = 'https://api.telegram.org'
TIMEOUT_SECONDS = 10


class TelegramDemoApprovalService:
    def __init__(self, config):
        self.token = (config.get('TELEGRAM_APPROVAL_BOT_TOKEN') or config.get('TELEGRAM_BOT_TOKEN') or '').strip()
        self.chat_id = str(config.get('TELEGRAM_APPROVAL_CHAT_ID') or '').strip()
        self.admin_user_id = str(config.get('TELEGRAM_APPROVAL_ADMIN_USER_ID') or '').strip()
        self.webhook_secret = (config.get('TELEGRAM_APPROVAL_WEBHOOK_SECRET') or '').strip()
        self.enabled = bool(config.get('TELEGRAM_APPROVAL_ENABLED'))

    def is_configured(self):
        return bool(self.token and self.chat_id and self.admin_user_id.isdigit()
                    and re.fullmatch(r'[A-Za-z0-9_-]{1,256}', self.webhook_secret))

    def is_active(self):
        return self.enabled and self.is_configured()

    def _post(self, method, data):
        if not self.is_active():
            return {'ok': False, 'skipped': True, 'reason': 'not_configured_or_disabled'}
        try:
            response = requests.post('{}/bot{}/{}'.format(API_BASE, self.token, method),
                                     json=data, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            logger.exception('Telegram demo approval %s failed (error_type=%s).', method, type(exc).__name__)
            return {'ok': False, 'error': 'Telegram request failed.'}
        try:
            payload = response.json()
        except (ValueError, AttributeError) as exc:
            logger.warning('Telegram demo approval %s failed safely (error_type=%s).', method, type(exc).__name__)
            return {'ok': False, 'http_status': response.status_code,
                    'error': 'Telegram response was invalid.'}
        if response.status_code == 200 and isinstance(payload, dict) and payload.get('ok') is True:
            return {'ok': True, 'http_status': response.status_code, 'result': payload.get('result')}
        # Never log Telegram's response body, which can echo private request data.
        description = payload.get('description', '') if isinstance(payload, dict) else ''
        description = str(description)[:240]
        if self.token:
            description = re.sub(re.escape(self.token), '[REDACTED]', description, flags=re.IGNORECASE)
        if 'message is not modified' in description.lower():
            return {'ok': True, 'http_status': response.status_code, 'result': payload.get('result'), 'not_modified': True}
        logger.warning('Telegram demo approval %s was rejected (http_status=%s, description=%s).',
                       method, response.status_code, description or 'unavailable')
        return {'ok': False, 'http_status': response.status_code,
                'description': description or None, 'error': 'Telegram rejected the request.'}

    def send_new_request(self, request_row, callback_data):
        if not self.is_active():
            if self.enabled:
                logger.warning('Telegram demo approval notification skipped: required settings are incomplete.')
            return {'ok': False, 'skipped': True, 'reason': 'not_configured_or_disabled'}
        chat_result = self._post('getChat', {'chat_id': self.chat_id})
        if not chat_result.get('ok') or not isinstance(chat_result.get('result'), dict) or chat_result['result'].get('type') != 'private':
            logger.warning('Telegram demo approval notification skipped: configured approval chat is not verified private.')
            return {'ok': False, 'skipped': True, 'reason': 'approval_chat_not_private'}
        text = ('NEW DEMO REQUEST\n'
                'Request ID: {id}\nApplicant: {name}\nEmail: {email}\n'
                'Company/organization: {company}\nSubmitted: {created_at}\nStatus: {status}').format(
                    id=request_row['id'], name=request_row['name'], email=request_row['email'],
                    company=request_row.get('company') or 'Not provided',
                    created_at=request_row['created_at'], status=request_row['status'])
        return self._post('sendMessage', {
            'chat_id': self.chat_id,
            'text': text,
            'reply_markup': {'inline_keyboard': [[
                {'text': '✅ Approve', 'callback_data': callback_data('approve', request_row['id'])},
                {'text': '❌ Reject', 'callback_data': callback_data('reject', request_row['id'])},
            ]]},
        })

    def send_diagnostic_test_message(self):
        """Send the explicitly requested plain-text approval-chat diagnostic."""
        if not self.is_active():
            logger.warning('Telegram demo approval diagnostic skipped: required settings are incomplete or disabled.')
            return {'ok': False, 'skipped': True, 'reason': 'not_configured_or_disabled'}
        chat_result = self._post('getChat', {'chat_id': self.chat_id})
        chat = chat_result.get('result') if chat_result.get('ok') else None
        if not isinstance(chat, dict) or chat.get('type') != 'private':
            logger.warning('Telegram demo approval diagnostic skipped: approval chat is not verified private.')
            return {'ok': False, 'skipped': True, 'reason': 'approval_chat_not_private'}
        return self._post('sendMessage', {
            'chat_id': self.chat_id,
            'text': 'SafetyVision AI demo approval test',
        })

    def answer_callback(self, callback_query_id, text=None, show_alert=False):
        data = {'callback_query_id': callback_query_id, 'show_alert': bool(show_alert)}
        if text:
            data['text'] = str(text)[:200]
        return self._post('answerCallbackQuery', data)

    def edit_decision_message(self, chat_id, message_id, request_id, decision):
        if decision not in ('APPROVED', 'REJECTED'):
            return {'ok': False, 'error': 'Unsupported decision.'}
        return self._post('editMessageText', {
            'chat_id': chat_id,
            'message_id': message_id,
            'text': '{} — DEMO REQUEST #{}'.format(decision, request_id),
            'reply_markup': {'inline_keyboard': []},
        })
