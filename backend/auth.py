"""
auth.py
-------
Two SEPARATE auth mechanisms, deliberately not shared:

  1. API bearer tokens (Authorization: Bearer <token>) -- used by the
     STUDENT'S DAEMON (backend_client.py) to call /api/* endpoints
     unattended, from a machine with no browser session.
  2. Browser cookie session (Flask's signed `session`) -- used by the
     WEB UI (teacher and student dashboards) when a person is logged in
     in an actual browser.

Password hashing uses werkzeug's generate_password_hash/check_password_hash
(already a transitive Flask dependency, so no new package needed) --
PBKDF2-SHA256 by default, salted per-password.
"""

import functools
import secrets

from flask import g, jsonify, redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

CLASS_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I confusion


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return check_password_hash(password_hash, password)


def generate_class_code() -> str:
    return "".join(secrets.choice(CLASS_CODE_ALPHABET) for _ in range(6))


# --------------------------------------------------------------------------
# API (bearer token) auth -- for backend_client.py on the student's machine
# --------------------------------------------------------------------------

def require_api_auth(db, role: str = None):
    """
    Decorator for JSON API routes. Reads `Authorization: Bearer <token>`,
    looks up the user, and stores it on flask.g.user for the route to use.
    role='student' / 'teacher' additionally rejects the wrong account type
    with 403 rather than letting e.g. a student token hit a teacher-only
    endpoint.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return jsonify(error="missing bearer token"), 401
            token = auth_header[len("Bearer "):].strip()
            user = db.get_user_by_token(token)
            if not user:
                return jsonify(error="invalid or expired token"), 401
            if role is not None and user["role"] != role:
                return jsonify(error=f"requires role={role}"), 403
            g.user = user
            return fn(*args, **kwargs)
        return wrapped
    return decorator


# --------------------------------------------------------------------------
# Web (cookie) auth -- for the browser dashboard
# --------------------------------------------------------------------------

def require_web_auth(db, role: str = None):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            user_id = session.get("user_id")
            if not user_id:
                return redirect(url_for("login", next=request.path))
            user = db.get_user(user_id)
            if not user:
                session.clear()
                return redirect(url_for("login"))
            if role is not None and user["role"] != role:
                return redirect(url_for("index"))
            g.user = user
            return fn(*args, **kwargs)
        return wrapped
    return decorator
