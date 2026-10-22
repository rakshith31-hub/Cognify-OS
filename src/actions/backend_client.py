"""
backend_client.py
-------------------
Talks to the optional teacher/supervision backend (backend/app.py). Every
call here is BEST-EFFORT: no network call in this module may ever crash
the daemon or block it for long — a student without connectivity, or
whose teacher hasn't set anything up, should get exactly the same local
experience as before this feature existed.

Credentials (server URL + bearer token) live in
logs/backend_credentials.json, written by `cognitive-monitor login` /
`cognitive-monitor link` (see cli.py). Nothing here ever stores a
password -- only the token issued at login.

Two calls matter to the daemon lifecycle (see daemon.py):
  - fetch_blocked_domains() at session START -- merged into
    site_blocking.domains for that session only, per the design
    decision that a teacher's domain list takes effect at the
    student's NEXT session, not pushed live into one in progress.
  - upload_session_summary() at session END -- a compact SUMMARY only
    (state percentages, top apps, blocked-attempt counts). Raw
    window-by-window predictions, camera data, and keystroke content
    never leave the machine; only the same kind of aggregate numbers
    the local dashboard already shows the student themselves.

If a session fails to upload (offline, server down), it's queued to
logs/pending_uploads/ and retried at the START of the next session --
so a teacher eventually sees it without the daemon needing to hang
around retrying mid-session.
"""

import json
import logging
import os
import time

import requests

logger = logging.getLogger("cognitive_monitor.backend_client")

DEFAULT_TIMEOUT = 5.0  # seconds -- never let a slow/unreachable server stall startup or shutdown


class BackendClient:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.credentials_path = os.path.join(log_dir, "backend_credentials.json")
        self.pending_dir = os.path.join(log_dir, "pending_uploads")
        self._creds = self._load_credentials()

    # ------------------------------------------------------------ setup --

    def _load_credentials(self):
        if not os.path.exists(self.credentials_path):
            return None
        try:
            with open(self.credentials_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not read backend credentials (%s): %s", type(e).__name__, e)
            return None

    def _save_credentials(self, creds: dict):
        os.makedirs(self.log_dir, exist_ok=True)
        tmp = self.credentials_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(creds, f)
        os.replace(tmp, self.credentials_path)
        self._creds = creds

    def is_configured(self) -> bool:
        return bool(self._creds and self._creds.get("server_url") and self._creds.get("token"))

    def _headers(self):
        return {"Authorization": f"Bearer {self._creds['token']}"}

    # -------------------------------------------- CLI-facing operations --

    def signup(self, server_url: str, email: str, password: str, role: str, display_name: str = None):
        """Called by `cognitive-monitor login --signup`. Returns the parsed
        response dict (includes class_code for teachers) or raises."""
        resp = requests.post(
            f"{server_url.rstrip('/')}/api/signup",
            json={"email": email, "password": password, "role": role, "display_name": display_name},
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        self._save_credentials({"server_url": server_url.rstrip("/"), "token": data["token"],
                                 "role": data["role"], "display_name": data.get("display_name")})
        return data

    def login(self, server_url: str, email: str, password: str):
        """Called by `cognitive-monitor login`. Returns the parsed response
        dict or raises requests.HTTPError / requests.ConnectionError."""
        resp = requests.post(
            f"{server_url.rstrip('/')}/api/login",
            json={"email": email, "password": password},
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        self._save_credentials({"server_url": server_url.rstrip("/"), "token": data["token"],
                                 "role": data["role"], "display_name": data.get("display_name")})
        return data

    def logout(self):
        try:
            os.remove(self.credentials_path)
        except FileNotFoundError:
            pass
        self._creds = None

    def link(self, class_code: str):
        """Called by `cognitive-monitor link <code>`. Raises on failure --
        this one's interactive/explicit, so the CLI should show the real
        error rather than silently swallowing it like the daemon-lifecycle
        calls below do."""
        if not self.is_configured():
            raise RuntimeError("Not logged in yet — run `cognitive-monitor login` first.")
        resp = requests.post(
            f"{self._creds['server_url']}/api/link",
            json={"class_code": class_code},
            headers=self._headers(),
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def whoami(self):
        return dict(self._creds) if self._creds else None

    # --------------------------------------- daemon-lifecycle operations --
    # Everything below is best-effort: logs a warning and returns quietly
    # on any failure. The daemon must work identically with no backend
    # configured, an unreachable backend, or a fully-linked one.

    def fetch_blocked_domains(self):
        """Returns a list of domains the linked teacher has blocked, or []
        if not configured/not linked/unreachable. Called once at daemon
        start, before site blocking is applied for the session."""
        if not self.is_configured() or self._creds.get("role") != "student":
            return []
        try:
            resp = requests.get(
                f"{self._creds['server_url']}/api/blocked-domains",
                headers=self._headers(), timeout=DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            domains = resp.json().get("domains", [])
            if domains:
                logger.info("Fetched %d teacher-blocked domain(s) for this session: %s",
                            len(domains), domains)
            return domains
        except Exception as e:
            logger.info("Could not fetch teacher's blocked-domain list (%s: %s) — "
                        "continuing with local config only.", type(e).__name__, e)
            return []

    def upload_session_summary(self, payload: dict):
        """Uploads one session's summary. On failure, queues it to
        logs/pending_uploads/ for retry_pending() to pick up next time.
        Never raises -- called from _shutdown(), which must always
        complete cleanly regardless of network state."""
        if not self.is_configured() or self._creds.get("role") != "student":
            return
        if not self._post_session(payload):
            self._queue_pending(payload)

    def retry_pending(self):
        """Called once at daemon start, before fetch_blocked_domains(), so
        a teacher eventually sees sessions uploaded from a machine that
        was offline when they actually ended."""
        if not self.is_configured() or self._creds.get("role") != "student":
            return
        if not os.path.isdir(self.pending_dir):
            return
        for fname in sorted(os.listdir(self.pending_dir)):
            path = os.path.join(self.pending_dir, fname)
            try:
                with open(path) as f:
                    payload = json.load(f)
            except Exception:
                continue
            if self._post_session(payload):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _post_session(self, payload: dict) -> bool:
        try:
            resp = requests.post(
                f"{self._creds['server_url']}/api/sessions",
                json=payload, headers=self._headers(), timeout=DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            logger.info("Session summary uploaded to supervision backend.")
            return True
        except Exception as e:
            logger.info("Could not upload session summary (%s: %s) — queued for retry.",
                        type(e).__name__, e)
            return False

    def _queue_pending(self, payload: dict):
        os.makedirs(self.pending_dir, exist_ok=True)
        fname = f"session_{int(payload.get('ended_at', time.time()))}.json"
        try:
            with open(os.path.join(self.pending_dir, fname), "w") as f:
                json.dump(payload, f)
        except OSError as e:
            logger.warning("Could not queue session summary for retry: %s", e)
