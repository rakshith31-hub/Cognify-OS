"""
db.py
-----
Plain sqlite3 (same style as src/actions/bandit.py -- no ORM) backing the
teacher/student supervision backend.

Tables:
  users            one row per account (role: 'teacher' or 'student')
  tokens           opaque bearer tokens issued on login, used by the
                    STUDENT'S MACHINE (backend_client.py) to authenticate
                    API calls -- separate from the browser's cookie session
                    used by the web UI (see app.py's `session`).
  links            student -> teacher, one teacher per student (a student
                    joins a class with the teacher's class_code)
  blocked_domains  a teacher's own domain list; students linked to that
                    teacher fetch it at their next session start
  sessions         one row per uploaded (ended) daemon session, holding a
                    compact JSON summary -- never raw event-by-event logs,
                    see backend_client.py's docstring for why
"""

import json
import os
import secrets
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK(role IN ('teacher', 'student')),
    display_name  TEXT NOT NULL,
    class_code    TEXT UNIQUE,          -- teachers only
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tokens (
    token      TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS links (
    student_id INTEGER PRIMARY KEY REFERENCES users(id),
    teacher_id INTEGER NOT NULL REFERENCES users(id),
    linked_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS blocked_domains (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_id INTEGER NOT NULL REFERENCES users(id),
    domain     TEXT NOT NULL,
    added_at   REAL NOT NULL,
    UNIQUE(teacher_id, domain)
);

CREATE TABLE IF NOT EXISTS sessions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id          INTEGER NOT NULL REFERENCES users(id),
    started_at          REAL NOT NULL,
    ended_at            REAL NOT NULL,
    total_windows       INTEGER NOT NULL DEFAULT 0,
    idle_windows        INTEGER NOT NULL DEFAULT 0,
    state_summary_json  TEXT NOT NULL DEFAULT '[]',
    top_apps_json       TEXT NOT NULL DEFAULT '[]',
    blocked_summary_json TEXT NOT NULL DEFAULT '[]',
    rating              INTEGER,
    uploaded_at         REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_student ON sessions(student_id);
CREATE INDEX IF NOT EXISTS idx_tokens_user ON tokens(user_id);
"""


class Database:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ---------------------------------------------------------- users ----

    def create_user(self, email: str, password_hash: str, role: str, display_name: str,
                     class_code: str = None):
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO users (email, password_hash, role, display_name, class_code, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (email.lower().strip(), password_hash, role, display_name, class_code, time.time()),
            )
            self._conn.commit()
            return cur.lastrowid

    def get_user_by_email(self, email: str):
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
            ).fetchone()
            return dict(row) if row else None

    def get_user(self, user_id: int):
        with self._lock:
            row = self._conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return dict(row) if row else None

    def get_teacher_by_class_code(self, class_code: str):
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE class_code = ? AND role = 'teacher'", (class_code.strip(),)
            ).fetchone()
            return dict(row) if row else None

    # --------------------------------------------------------- tokens ----

    def create_token(self, user_id: int) -> str:
        token = secrets.token_hex(32)
        with self._lock:
            self._conn.execute(
                "INSERT INTO tokens (token, user_id, created_at) VALUES (?, ?, ?)",
                (token, user_id, time.time()),
            )
            self._conn.commit()
        return token

    def get_user_by_token(self, token: str):
        with self._lock:
            row = self._conn.execute(
                "SELECT users.* FROM tokens JOIN users ON users.id = tokens.user_id "
                "WHERE tokens.token = ?",
                (token,),
            ).fetchone()
            return dict(row) if row else None

    def delete_token(self, token: str):
        with self._lock:
            self._conn.execute("DELETE FROM tokens WHERE token = ?", (token,))
            self._conn.commit()

    # ---------------------------------------------------------- links ----

    def link_student(self, student_id: int, teacher_id: int):
        with self._lock:
            self._conn.execute(
                "INSERT INTO links (student_id, teacher_id, linked_at) VALUES (?, ?, ?) "
                "ON CONFLICT(student_id) DO UPDATE SET teacher_id = excluded.teacher_id, "
                "linked_at = excluded.linked_at",
                (student_id, teacher_id, time.time()),
            )
            self._conn.commit()

    def get_teacher_for_student(self, student_id: int):
        with self._lock:
            row = self._conn.execute(
                "SELECT teacher_id FROM links WHERE student_id = ?", (student_id,)
            ).fetchone()
            return row["teacher_id"] if row else None

    def get_students_for_teacher(self, teacher_id: int):
        with self._lock:
            rows = self._conn.execute(
                "SELECT users.* FROM links JOIN users ON users.id = links.student_id "
                "WHERE links.teacher_id = ? ORDER BY users.display_name",
                (teacher_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------- domains ----

    def add_blocked_domain(self, teacher_id: int, domain: str):
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO blocked_domains (teacher_id, domain, added_at) VALUES (?, ?, ?)",
                (teacher_id, domain.lower().strip(), time.time()),
            )
            self._conn.commit()

    def remove_blocked_domain(self, teacher_id: int, domain_id: int):
        with self._lock:
            self._conn.execute(
                "DELETE FROM blocked_domains WHERE id = ? AND teacher_id = ?",
                (domain_id, teacher_id),
            )
            self._conn.commit()

    def get_blocked_domains(self, teacher_id: int):
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM blocked_domains WHERE teacher_id = ? ORDER BY domain",
                (teacher_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------- sessions ----

    def insert_session(self, student_id: int, payload: dict) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions (student_id, started_at, ended_at, total_windows, "
                "idle_windows, state_summary_json, top_apps_json, blocked_summary_json, "
                "rating, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    student_id,
                    payload["started_at"],
                    payload["ended_at"],
                    payload.get("total_windows", 0),
                    payload.get("idle_windows", 0),
                    json.dumps(payload.get("state_summary", [])),
                    json.dumps(payload.get("top_apps", [])),
                    json.dumps(payload.get("blocked_summary", [])),
                    payload.get("rating"),
                    time.time(),
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    def get_sessions_for_student(self, student_id: int, limit: int = 100):
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE student_id = ? ORDER BY started_at DESC LIMIT ?",
                (student_id, limit),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["state_summary"] = json.loads(d.pop("state_summary_json"))
                d["top_apps"] = json.loads(d.pop("top_apps_json"))
                d["blocked_summary"] = json.loads(d.pop("blocked_summary_json"))
                out.append(d)
            return out

    def close(self):
        self._conn.close()
