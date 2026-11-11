"""Run one local Telegram demo-approval callback poller alongside Flask."""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / '.env')

from services.telegram_approval_poller import (
    TelegramApprovalPoller, acquire_single_instance_lock, configure_diagnostic_logging,
)


def main():
    configure_diagnostic_logging(PROJECT_ROOT / 'instance' / 'telegram-approval-poller.log')
    config = {
        # Match the dedicated approval bot used by Flask; the PPE report bot
        # remains configured independently through TELEGRAM_BOT_TOKEN.
        'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_APPROVAL_BOT_TOKEN') or os.getenv('TELEGRAM_BOT_TOKEN', ''),
        'TELEGRAM_APPROVAL_WEBHOOK_SECRET': os.getenv('TELEGRAM_APPROVAL_WEBHOOK_SECRET', ''),
        'TELEGRAM_APPROVAL_POLLING_ENABLED': os.getenv('TELEGRAM_APPROVAL_POLLING_ENABLED', '').strip().lower()
        in ('1', 'true', 'yes', 'on'),
    }
    lock_handle = acquire_single_instance_lock(PROJECT_ROOT / 'instance' / 'telegram-approval-poller.lock')
    # Keep the open handle alive; Windows releases the byte-range lock on close.
    _ = lock_handle
    TelegramApprovalPoller(config).run()


if __name__ == '__main__':
    main()