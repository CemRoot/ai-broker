#!/usr/bin/env python3
"""Fill TELEGRAM_ALLOWED_USER_IDS from Telegram getUpdates (stdlib only).

Run from repo root after messaging your bot (e.g. /start):

    python scripts/sync_telegram_allowed_from_updates.py

Requires TELEGRAM_BOT_TOKEN in .env. Does not print the token.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
ALLOW_KEY = "TELEGRAM_ALLOWED_USER_IDS"


def _load_token(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("TELEGRAM_BOT_TOKEN=") and not line.startswith(
            "TELEGRAM_BOT_TOKEN=#"
        ):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("TELEGRAM_BOT_TOKEN missing in .env")


def _collect_user_ids(updates: list[dict]) -> list[int]:
    ids: set[int] = set()

    def take_from(obj: dict | None) -> None:
        if not obj:
            return
        user = obj.get("from")
        if user and isinstance(user.get("id"), int):
            ids.add(user["id"])

    for u in updates:
        take_from(u.get("message"))
        take_from(u.get("edited_message"))
        take_from(u.get("channel_post"))
        take_from(u.get("edited_channel_post"))
        cq = u.get("callback_query")
        if isinstance(cq, dict):
            take_from(cq)

    return sorted(ids)


def _fetch_updates(token: str) -> list[dict]:
    url = f"https://api.telegram.org/bot{token}/getUpdates?limit=100"
    try:
        with urlopen(url, timeout=30) as resp:
            body = json.loads(resp.read().decode())
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        sys.exit(f"getUpdates failed: {exc}")

    if not body.get("ok"):
        sys.exit(f"Telegram API error: {body!r}")
    return list(body.get("result") or [])


def _write_allowed_ids(path: Path, user_ids: str) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    if not lines:
        lines = [""]

    key_re = re.compile(rf"^{re.escape(ALLOW_KEY)}=.*$")
    new_line = f"{ALLOW_KEY}={user_ids}\n"
    replaced = False
    out: list[str] = []
    for line in lines:
        if key_re.match(line.rstrip("\n")):
            out.append(new_line)
            replaced = True
        else:
            out.append(line)

    if not replaced:
        out.append(new_line)

    path.write_text("".join(out), encoding="utf-8")


def main() -> None:
    if not ENV_PATH.is_file():
        sys.exit(f".env not found at {ENV_PATH}")

    token = _load_token(ENV_PATH)
    updates = _fetch_updates(token)
    ids = _collect_user_ids(updates)

    if not ids:
        sys.exit(
            "No user ids in update queue. Open your bot in Telegram, send /start once, "
            "then run this script again (getUpdates is empty until then)."
        )

    value = ",".join(str(i) for i in ids)
    _write_allowed_ids(ENV_PATH, value)
    print(f"Updated {ENV_PATH}: {ALLOW_KEY}={value}")


if __name__ == "__main__":
    main()
