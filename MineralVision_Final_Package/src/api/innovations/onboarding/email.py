"""Email delivery for onboarding (invitations, password resets).

Honesty contract:
- ``ConsoleEmailBackend`` (default) logs the full link loudly and reports
  delivery mode ``"console"`` — nothing is actually sent.
- ``SMTPEmailBackend`` is selected automatically when ``SMTP_HOST`` is set
  and uses real smtplib delivery.
- Tests can monkeypatch ``get_email_service`` (module-level provider) or use
  FastAPI ``app.dependency_overrides`` to capture messages.
"""

import logging
import os
import smtplib
from email.message import EmailMessage
from typing import List, Optional, Protocol

logger = logging.getLogger("mineralvision.onboarding.email")


class EmailService(Protocol):
    """Delivery backend protocol."""

    delivery_mode: str

    def send(self, to: str, subject: str, body: str) -> str:
        """Send a message; returns the delivery mode used."""
        ...


class ConsoleEmailBackend:
    """Default backend: loudly logs the email instead of sending it."""

    delivery_mode = "console"

    def send(self, to: str, subject: str, body: str) -> str:
        logger.warning(
            "ONBOARDING EMAIL (console backend — NOT actually sent)\n"
            "  To: %s\n  Subject: %s\n  Body:\n%s",
            to, subject, body,
        )
        return self.delivery_mode


class SMTPEmailBackend:
    """Real SMTP delivery via smtplib, configured purely from env vars."""

    delivery_mode = "smtp"

    def __init__(
        self,
        host: str,
        port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_addr: Optional[str] = None,
        use_tls: bool = True,
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.from_addr = from_addr or username or "no-reply@mineralvision.local"
        self.use_tls = use_tls

    def send(self, to: str, subject: str, body: str) -> str:
        msg = EmailMessage()
        msg["From"] = self.from_addr
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP(self.host, self.port, timeout=30) as smtp:
            if self.use_tls:
                smtp.starttls()
            if self.username:
                smtp.login(self.username, self.password or "")
            smtp.send_message(msg)
        logger.info("Onboarding email sent via SMTP to %s", to)
        return self.delivery_mode


# ---------------------------------------------------------------------------
# Provider (monkeypatch / dependency-override friendly)
# ---------------------------------------------------------------------------

_cached_backend: Optional[EmailService] = None


def _build_backend_from_env() -> EmailService:
    host = os.getenv("SMTP_HOST")
    if host:
        return SMTPEmailBackend(
            host=host,
            port=int(os.getenv("SMTP_PORT", "587")),
            username=os.getenv("SMTP_USER"),
            password=os.getenv("SMTP_PASSWORD"),
            from_addr=os.getenv("SMTP_FROM"),
            use_tls=os.getenv("SMTP_TLS", "true").lower() != "false",
        )
    return ConsoleEmailBackend()


def get_email_service() -> EmailService:
    """Provider used as a FastAPI dependency; tests may monkeypatch this."""
    global _cached_backend
    if _cached_backend is None:
        _cached_backend = _build_backend_from_env()
    return _cached_backend
