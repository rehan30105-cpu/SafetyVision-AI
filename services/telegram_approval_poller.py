"""Single-instance long-poll bridge for local Telegram demo approvals.

This consumes only callback_query updates and forwards them to the existing
secret-protected Flask webhook route. PPE notifications remain outbound-only.
"""
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests


logger = logging.getLogger('safetyvision.telegram_approval_poller')
API_BASE = 'https://api.telegram.org'
POLL_TIMEOUT_SECONDS = 25


def configure_diagnostic_logging(log_path):
    """Write polling diagnostics to a dedicated file without logging credentials."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(getattr(handler, '_safetyvision_telegram_approval_poller', False)
           for handler in logger.handlers):
        return
    handler = logging.FileHandler(path, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    handler._safetyvision_telegram_approval_poller = True
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class TelegramApprovalPoller:
    def __init__(self, config, webhook_url='http://127.0.0.1:5000/telegram/demo-approval/webhook'):
        self.token = str(config.get('TELEGRAM_APPROVAL_BOT_TOKEN') or config.get('TELEGRAM_BOT_TOKEN') or '').strip()
        self.secret = str(config.get('TELEGRAM_APPROVAL_WEBHOOK_SECRET') or '').strip()
        self.enabled = bool(config.get('TELEGRAM_APPROVAL_POLLING_ENABLED'))
        self.webhook_url = webhook_url
        self.offset = None
        self._backoff = 1  # starting backoff seconds for exponential backoff

    def _local_webhook_is_safe(self):
        parsed = urlsplit(self.webhook_url)
        return parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost')

    def _api_get(self, method, params=None):
        try:
            response = requests.get('{}/bot{}/{}'.format(API_BASE, self.token, method),
                                    params=params, timeout=POLL_TIMEOUT_SECONDS + 10)
            payload = response.json()
        except requests.exceptions.ReadTimeout:
            # Long-polling timeouts are normal when no updates arrive within the window.
            return []
        except (requests.RequestException, ValueError) as exc:
            logger.exception('Telegram approval polling %s failed (error_type=%s).', method, type(exc).__name__)
            return None
        if response.status_code == 409 and method == 'getUpdates':
            raise RuntimeError('Telegram reports another active update consumer; stopping instead of competing.')
        if response.status_code != 200 or not isinstance(payload, dict) or payload.get('ok') is not True:
            logger.warning('Telegram approval polling %s rejected (http_status=%s).', method,
                           getattr(response, 'status_code', 'unknown'))
            return None
        return payload.get('result')

    def _forward_update(self, update):
        try:
            response = requests.post(
                self.webhook_url, json=update,
                headers={'X-Telegram-Bot-Api-Secret-Token': self.secret}, timeout=30)
        except requests.RequestException as exc:
            logger.exception('Could not forward Telegram approval update (error_type=%s).', type(exc).__name__)
            return False
        if response.status_code != 200:
            logger.warning('Telegram approval update forwarding failed (http_status=%s).', response.status_code)
            return False
        return True

    def run(self):
        """Run the approval poller loop.

        Features:
        - Single-instance enforcement via file lock (inherited from caller).
        - Refuses to start if a Telegram webhook is already configured.
        - Exponential backoff on transient network failures.
        - Preserves unacknowledged offsets until forwarding succeeds.
        """
        if not self.enabled or not self.token or not self.secret:
            raise RuntimeError('Telegram approval polling is disabled or incompletely configured.')
        if not self._local_webhook_is_safe():
            raise RuntimeError('The polling bridge may only forward updates to this machine over loopback HTTP.')
        webhook = self._api_get('getWebhookInfo')
        if not isinstance(webhook, dict):
            raise RuntimeError('Could not verify Telegram webhook state; polling was not started.')
        if webhook.get('url'):
            raise RuntimeError('The Telegram bot already has a webhook configured; polling was not started.')
        if webhook.get('pending_update_count'):
            logger.warning('Telegram approval bot has %s pending updates before polling starts.',
                           webhook.get('pending_update_count'))
        if webhook.get('last_error_date'):
            logger.warning('Telegram approval bot reports a previous webhook delivery error (last_error_present=true).')

        logger.info('Telegram approval update consumer starting (mode=polling).')
        backoff = 1
        max_backoff = 30
        while True:
            updates = self._api_get('getUpdates', {
                'timeout': POLL_TIMEOUT_SECONDS,
                'allowed_updates': '["message","callback_query"]',
                **({'offset': self.offset} if self.offset is not None else {}),
            })
            if updates is None:
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue
            backoff = 1
            for update in updates:
                update_id = update.get('update_id')
                if isinstance(update_id, int):
                    self.offset = update_id
                if 'callback_query' in update:
                    logger.info('Telegram approval callback update received (update_id=%s, callback_present=true).',
                                update_id if isinstance(update_id, int) else 'unknown')
                    if not self._forward_update(update):
                        logger.warning(
                            'Telegram approval callback forward failed (update_id=%s); '
                            'backing off before retrying.',
                            update_id)
                        time.sleep(backoff)
                        backoff = min(backoff * 2, max_backoff)
                        break
                if isinstance(update_id, int):
                    self.offset = update_id + 1


def acquire_single_instance_lock(lock_path):
    """Hold a Windows byte-range lock for the poller's entire lifetime."""
    if os.name != 'nt':
        raise RuntimeError('The local poller lock currently requires Windows.')
    import msvcrt
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, 'a+b')
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        raise RuntimeError('Another Telegram approval poller already holds the local lock.')
    return handle
