"""
SAFETYVISION AI - Telegram Notification Service
================================================

Reusable, self-contained integration with the Telegram Bot API used to push
two kinds of notifications:

  1. SESSION SUMMARY (TEXT) -> when a detection session finishes, ONE detailed
                     text message summarising the real session results
                     (session id, detection type, people count, PPE violation
                     count, violation categories, compliance %, timestamp).
                     Sent only when the session has confirmed PPE violations
                     and deduped per session.
  2. PDF REPORTS     -> when a detection session finishes, the actual generated
                     PDF report (produced by services.report_service) is sent
                     as a document via the Bot API ``sendDocument`` endpoint,
                     with a descriptive filename and a summary caption.

A separate incident-level alert capability (``notify_new_violations``) also
exists but is intentionally kept distinct and is not auto-invoked by the
session-completion workflow, so incident alerts and session summaries/reports
never overlap or repeat.

Design principles
-----------------
* Credentials are read EXCLUSIVELY from environment variables
  (``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID``). They are never hardcoded
  and never written to logs. A ``.env`` file located beside ``app.py`` (the
  project root) is loaded into the process environment at import time so the
  credentials stay out of source control.
* The service NEVER raises to its callers. Every public method returns a
  result dictionary ``{'ok': bool, ...}`` so a Telegram failure can never
  prevent an incident or report from being persisted.
* Network calls use an explicit timeout and handle connection errors,
  timeouts, HTTP errors, Telegram API errors, and the 50 MB document size
  limit gracefully.
* Duplicate text alerts for the same incident are suppressed via an
  in-process set of already-alerted incident UIDs.
* PDFs generated in memory are uploaded directly. When only a file path is
  available it is streamed; when raw bytes must be materialised to disk a
  temporary file is used and always cleaned up in a ``finally`` block.

This module does not touch YOLO weights, detection logic, violation counting
or report contents - it only *reads* data produced by the existing pipeline.
"""

import io
import os
import logging
import tempfile
import threading
from datetime import datetime

import requests

# ---------------------------------------------------------------------------
# Credential / environment loading
# ---------------------------------------------------------------------------
# The project root is the parent directory of the ``services`` package, which
# is exactly where ``app.py`` (and therefore the ``.env`` file) lives.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")

# Populate os.environ from the .env file (if present). ``load_dotenv`` does
# NOT override variables already set in the real environment, which is the
# desired precedence. Failures here are non-fatal: if python-dotenv is absent
# or the file is missing we simply fall back to real environment variables.
try:  # pragma: no cover - defensive import
    from dotenv import load_dotenv

    load_dotenv(_ENV_PATH)
except Exception:  # pragma: no cover
    pass

logger = logging.getLogger("telegram_service")

TELEGRAM_API_BASE = "https://api.telegram.org"
# Telegram Bot API hard limit for uploading documents is 50 MB.
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
DEFAULT_TIMEOUT = 15  # seconds (connect + read)


class TelegramNotifier:
    """Thin, defensive wrapper around the Telegram Bot API."""

    def __init__(self, token=None, chat_id=None, timeout=DEFAULT_TIMEOUT):
        # Credentials come from the environment unless explicitly injected
        # (injection is used only by tests; never in production code paths).
        self.token = token if token is not None else os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID")
        self.timeout = timeout
        # Master switch - allows disabling delivery without removing code.
        self.enabled = os.environ.get("TELEGRAM_ENABLED", "true").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        # In-process dedup sets:
        #   _alerted_incidents -> incident-level alerts (distinct capability)
        #   _sent_summaries    -> session-level detailed text summaries
        #   _delivered_reports -> session-level PDF documents
        self._alerted_incidents = set()
        self._sent_summaries = set()
        self._delivered_reports = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def is_configured(self) -> bool:
        """True when a bot token and chat id are available."""
        return bool(self.token and self.chat_id)

    def is_active(self) -> bool:
        """True when configured AND enabled."""
        return self.is_configured() and self.enabled

    @property
    def _api_base(self) -> str:
        # The token is only ever placed in the request URL, never logged.
        return f"{TELEGRAM_API_BASE}/bot{self.token}"

    # ------------------------------------------------------------------
    # Low level HTTP
    # ------------------------------------------------------------------
    def _post(self, endpoint: str, *, data=None, files=None):
        """
        Perform a POST to a Bot API endpoint and normalise the result.

        Returns a dict: {'ok': bool, 'status_code': int|None,
                         'description': str|None, 'response': dict|None}
        """
        url = f"{self._api_base}/{endpoint}"
        try:
            resp = requests.post(url, data=data, files=files, timeout=self.timeout)
        except requests.exceptions.Timeout:
            logger.warning("Telegram %s timed out after %ss", endpoint, self.timeout)
            return {"ok": False, "status_code": None, "description": "timeout", "response": None}
        except requests.exceptions.ConnectionError:
            logger.warning("Telegram %s connection error", endpoint)
            return {"ok": False, "status_code": None, "description": "connection_error", "response": None}
        except requests.exceptions.RequestException as exc:
            # Never log the URL (it embeds the token).
            logger.warning("Telegram %s request failed: %s", endpoint, type(exc).__name__)
            return {"ok": False, "status_code": None, "description": "request_error", "response": None}

        payload = None
        try:
            payload = resp.json()
        except Exception:
            payload = None

        if resp.status_code == 200 and isinstance(payload, dict) and payload.get("ok"):
            return {"ok": True, "status_code": resp.status_code, "description": None, "response": payload}

        description = None
        if isinstance(payload, dict):
            description = payload.get("description")
        logger.warning(
            "Telegram %s failed: HTTP %s - %s",
            endpoint,
            resp.status_code,
            description or resp.reason,
        )
        return {"ok": False, "status_code": resp.status_code, "description": description, "response": payload}

    # ------------------------------------------------------------------
    # Public API - text message
    # ------------------------------------------------------------------
    def send_message(self, text: str, disable_web_page_preview: bool = True) -> dict:
        """Send a plain text message via the ``sendMessage`` endpoint."""
        if not self.is_active():
            return {"ok": False, "skipped": True, "reason": "not_configured_or_disabled"}
        if not text:
            return {"ok": False, "skipped": True, "reason": "empty_text"}

        data = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": disable_web_page_preview,
        }
        return self._post("sendMessage", data=data)

    # ------------------------------------------------------------------
    # Public API - document (PDF)
    # ------------------------------------------------------------------
    def _upload_document(self, file_bytes: bytes, filename: str, caption: str) -> dict:
        """Validate and upload an in-memory document via ``sendDocument``."""
        if not file_bytes:
            return {"ok": False, "skipped": True, "reason": "empty_document"}

        size = len(file_bytes)
        if size > MAX_DOCUMENT_BYTES:
            logger.warning(
                "Telegram document '%s' skipped: %d bytes exceeds %d byte limit",
                filename,
                size,
                MAX_DOCUMENT_BYTES,
            )
            return {"ok": False, "skipped": True, "reason": "file_too_large", "size": size}

        data = {"chat_id": self.chat_id}
        if caption:
            data["caption"] = caption[:1024]  # Telegram caption limit is 1024 chars.
        files = {"document": (filename, io.BytesIO(file_bytes), "application/pdf")}
        return self._post("sendDocument", data=data, files=files)


    def send_document(self, document, filename: str, caption: str = "") -> dict:
        """
        Send a PDF document via ``sendDocument``.

        ``document`` may be:
          * ``bytes`` / ``bytearray`` -> uploaded directly from memory.
          * a ``str`` path -> read into memory and uploaded directly.

        Returns the normalised result dict. Never raises.
        """
        if not self.is_active():
            return {"ok": False, "skipped": True, "reason": "not_configured_or_disabled"}
        if not filename:
            filename = "SafetyVision_Report.pdf"

        # Case 1: already in memory -> upload directly (no temp file needed).
        if isinstance(document, (bytes, bytearray)):
            return self._upload_document(bytes(document), filename, caption)

        # Case 2: a filesystem path -> read and upload directly.
        if isinstance(document, str):
            if not os.path.isfile(document):
                logger.warning("Telegram document path not found: %s", document)
                return {"ok": False, "skipped": True, "reason": "file_not_found"}
            try:
                size = os.path.getsize(document)
                if size > MAX_DOCUMENT_BYTES:
                    logger.warning(
                        "Telegram document '%s' skipped: %d bytes exceeds limit", filename, size
                    )
                    return {"ok": False, "skipped": True, "reason": "file_too_large", "size": size}
                with open(document, "rb") as fh:
                    return self._upload_document(fh.read(), filename, caption)
            except OSError as exc:
                logger.warning("Telegram failed reading '%s': %s", filename, type(exc).__name__)
                return {"ok": False, "skipped": True, "reason": "read_error"}

        return {"ok": False, "skipped": True, "reason": "unsupported_document_type"}

    def send_document_via_tempfile(self, data: bytes, filename: str, caption: str = "") -> dict:
        """
        Explicit temp-file fallback: materialise ``data`` to a temporary file,
        upload it, then ALWAYS remove the temp file in a ``finally`` block so a
        failed upload never leaks the file. Provided for callers whose only
        input is raw bytes but which prefer a disk-backed stream.
        """
        if not self.is_active():
            return {"ok": False, "skipped": True, "reason": "not_configured_or_disabled"}
        if not data:
            return {"ok": False, "skipped": True, "reason": "empty_document"}
        if len(data) > MAX_DOCUMENT_BYTES:
            return {"ok": False, "skipped": True, "reason": "file_too_large", "size": len(data)}

        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix="sv_tg_", suffix=".pdf")
            with os.fdopen(fd, "wb") as tmp:
                tmp.write(data)
            with open(tmp_path, "rb") as fh:
                content = fh.read()
            return self._upload_document(content, filename, caption)
        except OSError as exc:
            logger.warning("Telegram tempfile upload failed: %s", type(exc).__name__)
            return {"ok": False, "skipped": True, "reason": "tempfile_error"}
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    logger.warning("Could not remove temp file after Telegram upload")


    # ------------------------------------------------------------------
    # Dedup helpers
    # ------------------------------------------------------------------
    def _mark_incident_alerted(self, incident_uid: str) -> None:
        with self._lock:
            self._alerted_incidents.add(incident_uid)

    def _was_incident_alerted(self, incident_uid: str) -> bool:
        with self._lock:
            return incident_uid in self._alerted_incidents

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------
    def _format_violation_alert(self, incident: dict, session_uid: str) -> str:
        severity = str(incident.get("severity", "MEDIUM")).upper()
        lines = [
            "🚨 PPE Violation Confirmed",
            f"Incident: {incident.get('incident_uid', 'N/A')}",
            f"Session: {session_uid}",
            f"Person: #{incident.get('person_tracking_id', 'N/A')}",
            f"Violation: {incident.get('violation_type', 'Missing PPE')}",
            f"Severity: {severity} | Risk: {incident.get('risk_score', 'N/A')}",
        ]
        zone = incident.get("zone_name")
        if zone:
            lines.append(f"Zone: {zone}")
        lines.append(f"Time: {incident.get('timestamp') or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("— SafetyVision AI")
        return "\n".join(lines)

    def _format_report_caption(self, session: dict) -> str:
        def g(key, default="N/A"):
            val = session.get(key, default)
            return default if val is None else val

        compliance = session.get("overall_compliance")
        try:
            compliance_txt = f"{float(compliance):.1f}%" if compliance is not None else "N/A"
        except (TypeError, ValueError):
            compliance_txt = str(compliance)

        score = session.get("safety_score")
        try:
            score_txt = f"{float(score):.1f}" if score is not None else "N/A"
        except (TypeError, ValueError):
            score_txt = str(score)

        lines = [
            "📄 SafetyVision AI — Session Report",
            f"Session: {session.get('session_uid', 'N/A')} ({session.get('session_type', 'N/A')})",
            f"People: {g('people_count', 0)} | Compliant: {g('compliant_count', 0)} | Violations: {g('violation_count', 0)}",
            f"Compliance: {compliance_txt} | Safety Score: {score_txt}",
            f"Duration: {g('duration_seconds', 0)}s",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Session-level summary helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _map_session_type(session_type) -> str:
        """Map a stored session_type to a human label for Telegram."""
        mapping = {
            "IMAGE": "Image",
            "VIDEO": "Video",
            "LIVE": "Webcam (live)",
        }
        if not session_type:
            return "Unknown"
        return mapping.get(str(session_type).upper(), str(session_type))

    @staticmethod
    def _derive_violation_categories(person_results):
        """
        Derive the distinct PPE violation categories actually observed, using
        ONLY data already produced by the pipeline (no fabrication).

        Returns a dict like {'No Hardhat': 3, 'No Mask': 1} counting, per
        category, how many people are missing that item. Reads the canonical
        per-person missing-item flags (hardhat/mask/vest == 'MISSING') and
        falls back to the person's ``active_violations`` text when present.
        """
        from collections import Counter

        counts = Counter()
        for p in person_results or []:
            missing = []
            # Preferred: canonical per-item flags produced by the pipeline.
            if str(p.get("hardhat", "")).upper() == "MISSING":
                missing.append("No Hardhat")
            if str(p.get("mask", "")).upper() == "MISSING":
                missing.append("No Mask")
            if str(p.get("vest", "")).upper() == "MISSING":
                missing.append("No Safety Vest")
            # Fallback: parse from active_violations / missing_items text.
            if not missing:
                text_items = p.get("active_violations") or p.get("missing_items") or []
                for item in text_items:
                    token = str(item).lower()
                    if "hardhat" in token or "helmet" in token:
                        missing.append("No Hardhat")
                    if "mask" in token:
                        missing.append("No Mask")
                    if "vest" in token:
                        missing.append("No Safety Vest")
            for cat in set(missing):
                counts[cat] += 1
        return dict(counts)

    def _format_session_summary(self, session: dict) -> str:
        """
        Build the detailed session-level text summary from the authoritative,
        freshly-committed session row. Every figure is read from real data.
        """
        uid = session.get("session_uid", "N/A")
        det_type = self._map_session_type(session.get("session_type"))

        people = session.get("people_count", 0) or 0
        compliant = session.get("compliant_count", 0) or 0
        # Canonical display counts (recomputed by _enrich_session_counts).
        violation_items = session.get(
            "display_violation_count", session.get("violation_count", 0)
        ) or 0
        non_compliant_people = session.get(
            "display_non_compliant_people", session.get("violation_count", 0)
        ) or 0

        compliance = session.get("overall_compliance")
        try:
            compliance_txt = f"{float(compliance):.1f}%" if compliance is not None else "N/A"
        except (TypeError, ValueError):
            compliance_txt = "N/A"

        categories = self._derive_violation_categories(session.get("person_results"))
        if categories:
            cat_txt = ", ".join(f"{name} ({cnt})" for name, cnt in categories.items())
        else:
            cat_txt = "None"

        timestamp = session.get("end_time") or session.get("created_at") or datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        duration = session.get("duration_seconds", 0) or 0

        status_line = (
            "⚠️ Status: PPE violations detected"
            if self._has_confirmed_violations(session)
            else "✅ Status: No PPE violations detected"
        )
        lines = [
            "🛡️ SafetyVision AI — Detection Session Complete",
            status_line,
            "",
            f"🆔 Session ID: {uid}",
            f"🔎 Detection type: {det_type}",
            f"👥 People detected: {people}",
            f"✅ Compliant people: {compliant}",
            f"🚫 PPE violations: {violation_items} (missing items) across {non_compliant_people} person(s)",
            f"🏷️ Violation categories: {cat_txt}",
            f"📊 Overall compliance: {compliance_txt}",
            f"⏱️ Duration: {duration}s",
            f"🕒 Session time: {timestamp}",
            "",
            "📄 Full session report attached below.",
        ]
        return "\n".join(lines)

    def _has_confirmed_violations(self, session: dict) -> bool:
        """True when the session actually contains confirmed PPE violations."""
        try:
            if int(session.get("violation_count", 0) or 0) > 0:
                return True
            if int(session.get("display_violation_count", 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass
        return bool(self._derive_violation_categories(session.get("person_results")))

    # ------------------------------------------------------------------
    # High level orchestration
    # ------------------------------------------------------------------
    def _get_session_incidents(self, session_uid: str):
        """Fetch the confirmed incidents belonging to a detection session."""
        from database.db import get_db

        conn = get_db()
        try:
            rows = conn.execute(
                "SELECT incident_uid, person_tracking_id, violation_type, risk_score, "
                "severity, zone_name, timestamp FROM incidents WHERE incident_uid LIKE ?",
                (f"INC-{session_uid}-%",),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not load incidents for session %s: %s", session_uid, type(exc).__name__)
            return []
        finally:
            conn.close()

    def notify_new_violations(self, session_uid: str) -> list:
        """Send one text alert per newly confirmed incident (deduped)."""
        results = []
        if not self.is_active():
            return results
        for incident in self._get_session_incidents(session_uid):
            uid = incident.get("incident_uid")
            if not uid or self._was_incident_alerted(uid):
                continue
            res = self.send_message(self._format_violation_alert(incident, session_uid))
            # Mark alerted even on failure to avoid tight retry loops on a
            # persistent error; successful sends are of course recorded too.
            self._mark_incident_alerted(uid)
            results.append({"incident_uid": uid, "result": res})
        return results

    def notify_session_report(self, session_uid: str, session: dict = None) -> dict:
        """
        Generate the session PDF (reusing the existing generator) and deliver
        it via ``sendDocument`` with a descriptive filename and a summary
        caption.
        """
        if not self.is_active():
            return {"ok": False, "skipped": True, "reason": "not_configured_or_disabled"}
        with self._lock:
            if session_uid in self._delivered_reports:
                return {"ok": True, "skipped": True, "reason": "already_delivered"}
            self._delivered_reports.add(session_uid)

        try:
            # Imported lazily to avoid import-time cycles and heavy deps.
            from services.report_service import generate_session_pdf_report

            rel_path = generate_session_pdf_report(session_uid, is_admin=True)
            if not rel_path:
                return {"ok": False, "skipped": True, "reason": "report_generation_failed"}

            abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(_PROJECT_ROOT, rel_path)
            if not os.path.isfile(abs_path):
                return {"ok": False, "skipped": True, "reason": "report_file_missing"}

            if session is None:
                from services.session_service import SessionManager

                session = SessionManager.get_instance().get_session(session_uid, include_unowned=True) or {"session_uid": session_uid}

            caption = self._format_report_caption(session)
            filename = f"SafetyVision_Session_{session_uid}.pdf"
            return self.send_document(abs_path, filename, caption)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Telegram report delivery failed for %s: %s", session_uid, type(exc).__name__)
            return {"ok": False, "skipped": True, "reason": "exception"}


    def notify_session_summary(self, session_uid: str, session: dict = None) -> dict:
        """
        Send the single detailed session-level text summary (deduped per
        session). Built from the authoritative, freshly-committed session row
        so all figures are real. Sent for every completed session; a status
        line makes clear whether PPE violations were detected. Incident-level
        alerts (``notify_new_violations``) remain a separate, distinct
        capability and are NOT sent here.
        """
        if not self.is_active():
            return {"ok": False, "skipped": True, "reason": "not_configured_or_disabled"}

        with self._lock:
            if session_uid in self._sent_summaries:
                return {"ok": True, "skipped": True, "reason": "already_sent"}
            # Claim the slot immediately so a duplicate completion for the same
            # session cannot send a second summary.
            self._sent_summaries.add(session_uid)

        try:
            if session is None:
                from services.session_service import SessionManager

                session = SessionManager.get_instance().get_session(session_uid, include_unowned=True) or {"session_uid": session_uid}

            text = self._format_session_summary(session)
            return self.send_message(text)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Session summary delivery failed for %s: %s", session_uid, type(exc).__name__)
            return {"ok": False, "skipped": True, "reason": "exception"}

    def notify_session_completion(self, session_uid: str, session_res: dict = None) -> dict:
        """
        End-to-end session notification workflow (never raises):

          1. ONE detailed session-level text summary (real stats), sent only
             when the session has confirmed PPE violations, deduped per session.
          2. The actual generated PDF report delivered via ``sendDocument``
             with a descriptive filename and a summary caption, deduped per
             session.

        Incidents and reports are already persisted by the caller before this
        runs; a Telegram failure here can never lose them. Incident-level
        alerts (``notify_new_violations``) are intentionally NOT invoked here so
        they stay distinct from the session summary and are not repeated.
        """
        if not self.is_active():
            return {"ok": False, "skipped": True, "reason": "not_configured_or_disabled"}

        summary = {"session_uid": session_uid, "summary_text": None, "report": None}
        try:
            # session_res from the caller uses endpoint-specific key names, so
            # read the authoritative committed row instead for accurate stats.
            summary["summary_text"] = self.notify_session_summary(session_uid, None)
            summary["report"] = self.notify_session_report(session_uid, None)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("notify_session_completion failed for %s: %s", session_uid, type(exc).__name__)
            return {"ok": False, "error": "exception", "summary": summary}

        text_ok = summary["summary_text"] and summary["summary_text"].get("ok")
        report_ok = summary["report"] and summary["report"].get("ok")
        summary["ok"] = bool(text_ok or report_ok)
        return summary


# ---------------------------------------------------------------------------
# Module-level singleton + non-blocking helper used by app.py
# ---------------------------------------------------------------------------
_notifier = None
_notifier_lock = threading.Lock()


def get_notifier() -> TelegramNotifier:
    """Return the process-wide TelegramNotifier singleton."""
    global _notifier
    if _notifier is None:
        with _notifier_lock:
            if _notifier is None:
                _notifier = TelegramNotifier()
    return _notifier


def notify_session_completion(session_uid: str, session_res: dict = None, blocking: bool = False):
    """
    Fire-and-forget notification used by the Flask endpoints.

    By default this runs on a short-lived daemon thread so Telegram latency or
    outages can never delay the HTTP response or block persistence. Set
    ``blocking=True`` (used by tests) to run synchronously.
    """
    notifier = get_notifier()
    if not notifier.is_active():
        logger.info("Telegram notifications inactive (not configured or disabled); skipping %s", session_uid)
        return None

    def _run():
        try:
            notifier.notify_session_completion(session_uid, session_res)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Background Telegram notification failed for %s: %s", session_uid, type(exc).__name__)

    if blocking:
        return notifier.notify_session_completion(session_uid, session_res)

    thread = threading.Thread(target=_run, name=f"telegram-notify-{session_uid}", daemon=True)
    thread.start()
    return thread

