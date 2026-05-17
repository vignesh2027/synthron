"""Email tool — send and read emails via SMTP/IMAP."""

from __future__ import annotations

import asyncio
import email as email_lib
import imaplib
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from synthron.tools.base_tool import BaseTool
from synthron.utils.exceptions import ToolExecutionError
from synthron.utils.logger import get_logger

logger = get_logger(__name__)


class EmailTool(BaseTool):
    """Send and read emails using SMTP and IMAP.

    Configure via environment variables:
        EMAIL_ADDRESS  — sender email
        EMAIL_PASSWORD — app password
        SMTP_HOST      — default: smtp.gmail.com
        SMTP_PORT      — default: 587
        IMAP_HOST      — default: imap.gmail.com

    Input format:
        send:to@example.com:Subject:Body text
        read:inbox:5  (read last 5 emails from inbox)
    """

    name = "email_tool"
    description = (
        "Send and read emails. "
        "Send: 'send:to@email.com:Subject:Body'. "
        "Read: 'read:inbox:5' (last N emails)."
    )
    category = "communication"
    requires_network = True
    is_destructive = True

    def __init__(self) -> None:
        self.email_address = os.getenv("EMAIL_ADDRESS", "")
        self.email_password = os.getenv("EMAIL_PASSWORD", "")
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.imap_host = os.getenv("IMAP_HOST", "imap.gmail.com")

    async def run(self, input_text: str, context: Any = None) -> str:
        """Execute an email operation.

        Args:
            input_text: Formatted email command string.
            context: Unused.

        Returns:
            Result message as string.
        """
        if not self.email_address or not self.email_password:
            return (
                "Email not configured. Set EMAIL_ADDRESS and EMAIL_PASSWORD env vars.\n"
                "For Gmail: use an App Password (not your account password)."
            )

        text = input_text.strip()
        parts = text.split(":", 3)
        action = parts[0].lower() if parts else ""

        if action == "send":
            to = parts[1].strip() if len(parts) > 1 else ""
            subject = parts[2].strip() if len(parts) > 2 else "No Subject"
            body = parts[3].strip() if len(parts) > 3 else ""
            return await asyncio.to_thread(self._send_email, to, subject, body)

        elif action == "read":
            folder = parts[1].strip() if len(parts) > 1 else "INBOX"
            count = int(parts[2].strip()) if len(parts) > 2 and parts[2].strip().isdigit() else 5
            return await asyncio.to_thread(self._read_emails, folder, count)

        else:
            # Treat as a send with natural language format
            return (
                "Unknown email command. Use:\n"
                "  send:to@email.com:Subject:Body\n"
                "  read:INBOX:5"
            )

    def _send_email(self, to: str, subject: str, body: str) -> str:
        """Send an email via SMTP."""
        if not to or "@" not in to:
            return f"Invalid recipient email: '{to}'"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.email_address
        msg["To"] = to

        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.login(self.email_address, self.email_password)
                server.sendmail(self.email_address, [to], msg.as_string())
            logger.info(f"[email_tool] Sent email to {to}: {subject}")
            return f"✅ Email sent to {to}\nSubject: {subject}"
        except smtplib.SMTPAuthenticationError:
            raise ToolExecutionError(
                "email_tool",
                "Authentication failed. Check EMAIL_ADDRESS and EMAIL_PASSWORD.",
            )
        except Exception as exc:
            raise ToolExecutionError("email_tool", f"SMTP error: {exc}") from exc

    def _read_emails(self, folder: str = "INBOX", count: int = 5) -> str:
        """Read recent emails via IMAP."""
        try:
            with imaplib.IMAP4_SSL(self.imap_host) as imap:
                imap.login(self.email_address, self.email_password)
                imap.select(folder)

                _, msg_ids = imap.search(None, "ALL")
                ids = msg_ids[0].split()
                recent_ids = ids[-count:] if len(ids) >= count else ids

                emails = []
                for msg_id in reversed(recent_ids):
                    _, msg_data = imap.fetch(msg_id, "(RFC822)")
                    raw = msg_data[0][1]
                    msg = email_lib.message_from_bytes(raw)

                    subject = msg.get("Subject", "(no subject)")
                    sender = msg.get("From", "unknown")
                    date = msg.get("Date", "")

                    # Extract body
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                                break
                    else:
                        body = msg.get_payload(decode=True).decode("utf-8", errors="replace")

                    emails.append(
                        f"FROM: {sender}\nDATE: {date}\nSUBJECT: {subject}\n"
                        f"BODY:\n{body[:500]}{'...' if len(body) > 500 else ''}"
                    )

            return f"Last {len(emails)} emails from {folder}:\n\n" + "\n\n---\n\n".join(emails)

        except imaplib.IMAP4.error as exc:
            raise ToolExecutionError("email_tool", f"IMAP error: {exc}") from exc
        except Exception as exc:
            raise ToolExecutionError("email_tool", str(exc)) from exc
