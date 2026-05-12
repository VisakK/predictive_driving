"""Send a notification email via Gmail SMTP.

Reads the App Password from a file (default ``~/.gmail_app_password``) and
sends a plain-text email with optional file attachments. Designed to be
called from shell scripts at the end of long-running training/eval
pipelines so a missing or wrong credential doesn't fail the pipeline:
errors are logged to stderr and the script exits 0.

Usage examples:

  python scripts/send_email.py \
      --subject "v3ts pipeline finished" \
      --body "All 4 trainings + evals complete. See attached summary."

  python scripts/send_email.py \
      --subject "v3ts results" \
      --body-file experiments/v3ts_eval/results/comparison.md \
      --attach experiments/v3ts_eval/results/comparison.md
"""
from __future__ import annotations

import argparse
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from pathlib import Path


DEFAULT_SENDER = "vishak.vk@gmail.com"
DEFAULT_RECIPIENT = "vishak.vk@gmail.com"
DEFAULT_PASSWORD_FILE = "~/.gmail_app_password"


def _read_password(path: str) -> str | None:
    p = Path(path).expanduser()
    if not p.exists():
        print(f"[send_email] credential file not found: {p}", file=sys.stderr)
        return None
    pw = p.read_text().strip()
    if not pw:
        print(f"[send_email] credential file is empty: {p}", file=sys.stderr)
        return None
    return pw


def send(
    subject: str,
    body: str,
    sender: str = DEFAULT_SENDER,
    recipient: str = DEFAULT_RECIPIENT,
    password_file: str = DEFAULT_PASSWORD_FILE,
    attachments: list[str] | None = None,
) -> bool:
    pw = _read_password(password_file)
    if pw is None:
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(body)

    for path in attachments or []:
        p = Path(path)
        if not p.exists():
            print(f"[send_email] attachment not found, skipping: {p}", file=sys.stderr)
            continue
        data = p.read_bytes()
        # text/plain for .md/.txt/.log/.json; octet-stream otherwise.
        suffix = p.suffix.lower()
        if suffix in {".md", ".txt", ".log", ".json", ".yaml", ".yml", ".csv"}:
            msg.add_attachment(data, maintype="text", subtype="plain", filename=p.name)
        else:
            msg.add_attachment(
                data, maintype="application", subtype="octet-stream", filename=p.name
            )

    ctx = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as smtp:
            smtp.login(sender, pw)
            smtp.send_message(msg)
    except Exception as e:
        print(f"[send_email] SMTP error: {e}", file=sys.stderr)
        return False
    print(f"[send_email] sent to {recipient} (subject: {subject!r})")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--subject", required=True)
    body_grp = p.add_mutually_exclusive_group(required=True)
    body_grp.add_argument("--body", help="Inline message body")
    body_grp.add_argument("--body-file", help="Read message body from a file")
    p.add_argument("--sender", default=DEFAULT_SENDER)
    p.add_argument("--recipient", default=DEFAULT_RECIPIENT)
    p.add_argument(
        "--password-file",
        default=DEFAULT_PASSWORD_FILE,
        help="Path to file containing the Gmail App Password (default ~/.gmail_app_password)",
    )
    p.add_argument(
        "--attach",
        action="append",
        default=[],
        help="Path to a file to attach (may be repeated)",
    )
    args = p.parse_args()

    if args.body is not None:
        body = args.body
    else:
        body = Path(args.body_file).expanduser().read_text()

    ok = send(
        subject=args.subject,
        body=body,
        sender=args.sender,
        recipient=args.recipient,
        password_file=args.password_file,
        attachments=args.attach,
    )
    # Always exit 0 so a misconfigured credential never breaks an upstream
    # pipeline; the warning was already printed to stderr.
    sys.exit(0 if ok else 0)


if __name__ == "__main__":
    main()
