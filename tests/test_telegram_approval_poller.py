"""Offline safety tests for local approval callback polling."""
import pytest

from services.telegram_approval_poller import TelegramApprovalPoller


class Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


def test_poller_refuses_existing_webhook(monkeypatch):
    import services.telegram_approval_poller as module
    monkeypatch.setattr(module.requests, 'get', lambda *args, **kwargs: Response({
        'ok': True, 'result': {'url': 'https://existing.example/webhook'}
    }))
    poller = TelegramApprovalPoller({
        'TELEGRAM_BOT_TOKEN': 'test-token',
        'TELEGRAM_APPROVAL_WEBHOOK_SECRET': 'test-secret',
        'TELEGRAM_APPROVAL_POLLING_ENABLED': True,
    })
    with pytest.raises(RuntimeError, match='webhook configured'):
        poller.run()


def test_poller_forwards_callback_only_to_loopback_with_webhook_secret(monkeypatch):
    import services.telegram_approval_poller as module
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(('get', url.rsplit('/', 1)[-1], params))
        if url.endswith('/getWebhookInfo'):
            return Response({'ok': True, 'result': {'url': ''}})
        raise AssertionError('The poller must use its update processor in this test.')

    monkeypatch.setattr(module.requests, 'get', fake_get)
    poller = TelegramApprovalPoller({
        'TELEGRAM_BOT_TOKEN': 'test-token',
        'TELEGRAM_APPROVAL_WEBHOOK_SECRET': 'test-secret',
        'TELEGRAM_APPROVAL_POLLING_ENABLED': True,
    })
    monkeypatch.setattr(poller, '_forward_update', lambda update: calls.append(('forward', update)) or False)
    updates = [{'update_id': 10, 'callback_query': {'id': 'cb'}}, {'update_id': 11, 'message': {}}]
    def fake_api_get(method, params=None):
        if method == 'getWebhookInfo':
            return {'url': ''}
        assert method == 'getUpdates'
        assert params['allowed_updates'] == '["message","callback_query"]'
        return updates

    monkeypatch.setattr(poller, '_api_get', fake_api_get)
    monkeypatch.setattr(module.time, 'sleep', lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        poller.run()
    assert calls == [('forward', updates[0])]
    assert poller.offset == 10


def test_poller_rejects_non_loopback_forward_target():
    poller = TelegramApprovalPoller({}, webhook_url='https://example.test/telegram/demo-approval/webhook')
    assert poller._local_webhook_is_safe() is False


def test_poller_exponential_backoff_and_reconnection(monkeypatch):
    import services.telegram_approval_poller as module
    sleep_durations = []
    attempts = [0]

    def fake_api_get(method, params=None):
        if method == 'getWebhookInfo':
            return {'url': ''}
        if method == 'getUpdates':
            attempts[0] += 1
            if attempts[0] < 3:
                return None  # Simulate network failure
            if attempts[0] == 3:
                return [{'update_id': 50, 'message': {}}]
            raise KeyboardInterrupt()
        return None

    poller = TelegramApprovalPoller({
        'TELEGRAM_BOT_TOKEN': 'test-token',
        'TELEGRAM_APPROVAL_WEBHOOK_SECRET': 'test-secret',
        'TELEGRAM_APPROVAL_POLLING_ENABLED': True,
    })
    monkeypatch.setattr(poller, '_api_get', fake_api_get)

    def fake_sleep(sec):
        sleep_durations.append(sec)
        if len(sleep_durations) >= 3:
            raise KeyboardInterrupt()

    monkeypatch.setattr(module.time, 'sleep', fake_sleep)
    with pytest.raises(KeyboardInterrupt):
        poller.run()

    assert sleep_durations[:2] == [1, 2]
    assert poller.offset == 51


def test_poller_multiple_worker_startup_lock(tmp_path):
    import os
    if os.name != 'nt':
        pytest.skip('Windows file lock test')
    from services.telegram_approval_poller import acquire_single_instance_lock
    lock_file = tmp_path / 'poller.lock'
    handle1 = acquire_single_instance_lock(lock_file)
    try:
        with pytest.raises(RuntimeError, match='Another Telegram approval poller already holds the local lock'):
            acquire_single_instance_lock(lock_file)
    finally:
        handle1.close()
