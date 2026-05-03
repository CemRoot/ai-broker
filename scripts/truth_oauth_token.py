"""
Truth Social (Mastodon-compatible) OAuth helper.

Goal: reduce the token setup to a single command + one browser approval.

What it does:
- Registers an OAuth app via POST /api/v1/apps
  (or uses provided client_id/client_secret if you already have them)
- Starts a local callback server on http://127.0.0.1:8765/callback
- Opens the browser to /oauth/authorize
- Exchanges the returned code for an access token via POST /oauth/token
- Writes TRUTH_SOCIAL_ACCESS_TOKEN into .env (idempotent)

Notes:
- You will still need to log in and approve in the browser (cannot be automated safely).
- Keep tokens/secrets out of chat logs.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx


def _upsert_env_var(env_path: Path, key: str, value: str) -> None:
    if not env_path.exists():
        env_path.write_text(f"{key}={value}\n", encoding="utf-8")
        return

    lines = env_path.read_text(encoding="utf-8").splitlines(keepends=False)
    out: list[str] = []
    replaced = False

    for line in lines:
        if line.startswith(f"{key}="):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)

    if not replaced:
        if out and out[-1].strip() != "":
            out.append("")
        out.append(f"{key}={value}")

    env_path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


class _CallbackState:
    def __init__(self) -> None:
        self.code: str | None = None
        self.error: str | None = None


def _serve_oauth_callback(host: str, port: int, state: _CallbackState) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/callback":
                self.send_response(404)
                self.end_headers()
                return

            qs = urllib.parse.parse_qs(parsed.query)
            if "error" in qs:
                state.error = qs.get("error", ["unknown"])[0]
            if "code" in qs:
                state.code = qs.get("code", [None])[0]

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            msg = (
                "<h2>AI Broker — OAuth callback received</h2>"
                "<p>You can close this tab and return to the terminal.</p>"
            )
            self.wfile.write(msg.encode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Avoid noisy request logging / leaking query params.
            return

    httpd = HTTPServer((host, port), Handler)
    return httpd


def _register_app(
    client: httpx.Client,
    base_url: str,
    redirect_uri: str,
    scopes: str,
    client_name: str,
    website: str,
) -> tuple[str, str]:
    url = f"{base_url.rstrip('/')}/api/v1/apps"
    # Truth Social frequently sits behind bot / WAF checks; mimic a browser-ish request.
    resp = client.post(
        url,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        },
        json={
            "client_name": client_name,
            "redirect_uris": redirect_uri,
            "scopes": scopes,
            "website": website,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    cid = data.get("client_id")
    csec = data.get("client_secret")
    if not isinstance(cid, str) or not isinstance(csec, str) or not cid or not csec:
        raise RuntimeError(f"Unexpected /api/v1/apps response: {json.dumps(data)[:400]}")
    return cid, csec


def _validate_token(
    client: httpx.Client,
    base_url: str,
    token: str,
    trump_acct: str = "realDonaldTrump",
) -> None:
    api = f"{base_url.rstrip('/')}/api/v1"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
    }

    # Prefer curl_cffi (curl-impersonate) to bypass Cloudflare/WAF.
    try:
        from curl_cffi import requests as crequests  # type: ignore

        me = crequests.get(
            f"{api}/accounts/verify_credentials",
            headers=headers,
            timeout=30,
            impersonate="chrome",
        )
        if int(me.status_code) != 200:
            raise RuntimeError(f"verify_credentials HTTP {me.status_code}: {(me.text or '')[:200]}")
        me_data = me.json()

        lookup = crequests.get(
            f"{api}/accounts/lookup",
            headers=headers,
            params={"acct": trump_acct},
            timeout=30,
            impersonate="chrome",
        )
        if int(lookup.status_code) != 200:
            raise RuntimeError(f"lookup HTTP {lookup.status_code}: {(lookup.text or '')[:200]}")
        lu = lookup.json()
    except Exception:
        me = client.get(f"{api}/accounts/verify_credentials", headers=headers, timeout=30)
        me.raise_for_status()
        me_data = me.json()
        lookup = client.get(
            f"{api}/accounts/lookup",
            headers=headers,
            params={"acct": trump_acct},
            timeout=30,
        )
        lookup.raise_for_status()
        lu = lookup.json()
    me_username = me_data.get("username") or me_data.get("acct") or "<unknown>"
    trump_id = lu.get("id")
    trump_username = lu.get("acct") or lu.get("username") or trump_acct

    print("Token OK.")
    print("Authenticated user:", me_username)
    print("Trump lookup:", trump_username, "(id:", trump_id, ")")
    if not trump_id:
        print("Warning: could not resolve Trump account id; check TRUMP_TRUTH_ACCOUNT_USERNAME.")


def _exchange_token(
    client: httpx.Client,
    base_url: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    code: str,
) -> str:
    url = f"{base_url.rstrip('/')}/oauth/token"
    resp = client.post(
        url,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
            "code": code,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    tok = data.get("access_token")
    if not isinstance(tok, str) or not tok:
        raise RuntimeError(f"Unexpected /oauth/token response: {json.dumps(data)[:400]}")
    return tok


def main() -> int:
    ap = argparse.ArgumentParser(description="Obtain Truth Social OAuth access token and write to .env")
    ap.add_argument("--base-url", default="https://truthsocial.com", help="Truth Social base URL")
    ap.add_argument("--scopes", default="read write follow", help="OAuth scopes")
    ap.add_argument("--client-name", default="ai-broker-local", help="OAuth app name for registration")
    ap.add_argument("--website", default="https://localhost", help="OAuth app website for registration")
    ap.add_argument("--host", default="127.0.0.1", help="Local callback host")
    ap.add_argument("--port", type=int, default=8765, help="Local callback port")
    ap.add_argument("--env-path", default=".env", help="Path to .env to update")
    ap.add_argument("--client-id", default="", help="Reuse existing client_id (skip app registration)")
    ap.add_argument("--client-secret", default="", help="Reuse existing client_secret (skip app registration)")
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="Do not run OAuth; validate TRUTH_SOCIAL_ACCESS_TOKEN from .env",
    )
    args = ap.parse_args()

    env_path = Path(args.env_path)
    if args.validate_only:
        if not env_path.exists():
            raise SystemExit(f"{args.env_path} not found")
        token = ""
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("TRUTH_SOCIAL_ACCESS_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
        if not token:
            raise SystemExit("TRUTH_SOCIAL_ACCESS_TOKEN missing/empty in .env")
        with httpx.Client(follow_redirects=True) as client:
            _validate_token(client=client, base_url=args.base_url, token=token)
        return 0

    redirect_uri = f"http://{args.host}:{args.port}/callback"

    state = _CallbackState()
    httpd = _serve_oauth_callback(args.host, args.port, state)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    with httpx.Client(follow_redirects=True) as client:
        client_id = args.client_id.strip()
        client_secret = args.client_secret.strip()
        if not client_id or not client_secret:
            try:
                client_id, client_secret = _register_app(
                    client=client,
                    base_url=args.base_url,
                    redirect_uri=redirect_uri,
                    scopes=args.scopes,
                    client_name=args.client_name,
                    website=args.website,
                )
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response else None
                if status == 403:
                    httpd.shutdown()
                    raise SystemExit(
                        "403 Forbidden while registering app at /api/v1/apps.\n\n"
                        "This is typically Truth Social's WAF/bot protection blocking non-browser requests.\n"
                        "Workaround:\n"
                        "1) Register the app using your in-browser REST tool (the one that returned 200).\n"
                        "2) Re-run this script with the issued credentials:\n\n"
                        "   uv run python scripts/truth_oauth_token.py \\\n"
                        "     --client-id '<CLIENT_ID>' \\\n"
                        "     --client-secret '<CLIENT_SECRET>'\n\n"
                        "The rest of the flow (browser approve + token exchange + .env write) will be automated."
                    ) from e
                raise

        auth_url = (
            f"{args.base_url.rstrip('/')}/oauth/authorize?"
            + urllib.parse.urlencode(
                {
                    "client_id": client_id,
                    "redirect_uri": redirect_uri,
                    "response_type": "code",
                    "scope": args.scopes,
                }
            )
        )

        print("Open this URL in your browser and approve access:")
        print(auth_url)
        try:
            webbrowser.open(auth_url, new=1, autoraise=True)
        except Exception:
            pass

        deadline = time.time() + 180
        while time.time() < deadline and not state.code and not state.error:
            time.sleep(0.25)

        httpd.shutdown()

        if state.error:
            raise SystemExit(f"OAuth error returned to callback: {state.error}")
        if not state.code:
            raise SystemExit("Timed out waiting for OAuth callback. Try again or check the browser tab.")

        token = _exchange_token(
            client=client,
            base_url=args.base_url,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            code=state.code,
        )

    _upsert_env_var(env_path, "TRUTH_SOCIAL_ACCESS_TOKEN", token)
    print("Wrote TRUTH_SOCIAL_ACCESS_TOKEN to", str(env_path))
    # Immediate sanity check so you know it works.
    with httpx.Client(follow_redirects=True) as client:
        _validate_token(client=client, base_url=args.base_url, token=token)
    print("Next: ensure you follow @realDonaldTrump from the same Truth account.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

