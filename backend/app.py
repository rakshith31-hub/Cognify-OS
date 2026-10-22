"""
app.py (backend)
------------------
The teacher/student supervision server.

  - Students install the daemon as before (see project README) and run
    `cognitive-monitor login`, then `cognitive-monitor link <class code>`
    once, pointing at wherever this app is hosted.
  - At every session START, the daemon fetches the teacher's current
    blocked-domain list from here (GET /api/blocked-domains) and applies
    it for that session -- see the project's README section on this: a
    domain added by the teacher mid-session applies from the STUDENT'S
    NEXT session onward, not live into an already-running one.
  - At every session END, the daemon uploads a compact SUMMARY (state
    percentages, top apps, blocked-attempt counts -- never raw
    window-by-window logs or camera data) via POST /api/sessions.
  - Teachers log in through the browser, see their linked students, and
    can drill into any student's session history. Students can also log
    in through the browser and see only their own.

Run standalone for development:
    python backend/app.py --host 0.0.0.0 --port 5800

For anything beyond local testing, run behind a real WSGI server
(gunicorn/waitress) and HTTPS -- this dev server is not hardened for
public exposure, and everything here (passwords aside) is otherwise
plain data over whatever transport you put it behind.
"""

import argparse
import os
import secrets
import sys
from datetime import datetime

from flask import Flask, g, jsonify, redirect, render_template, request, session, url_for

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auth import (generate_class_code, hash_password, require_api_auth,
                   require_web_auth, verify_password)
from db import Database

BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))


def create_app(db_path: str = None, secret_key: str = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = secret_key or os.environ.get("COGMON_SECRET_KEY") or secrets.token_hex(32)

    db = Database(db_path or os.path.join(BACKEND_ROOT, "cogmon_backend.db"))

    @app.template_filter("timestamp")
    def _fmt_timestamp(epoch_seconds):
        try:
            return datetime.fromtimestamp(epoch_seconds).strftime("%b %d, %Y %H:%M")
        except (TypeError, ValueError, OSError):
            return "—"

    # ---------------------------------------------------------------- #
    # API — used by the student's daemon (bearer token auth)           #
    # ---------------------------------------------------------------- #

    @app.route("/api/signup", methods=["POST"])
    def api_signup():
        data = request.get_json(force=True, silent=True) or {}
        email = (data.get("email") or "").strip()
        password = data.get("password") or ""
        role = data.get("role")
        display_name = (data.get("display_name") or email.split("@")[0]).strip()

        if not email or "@" not in email:
            return jsonify(error="a valid email is required"), 400
        if len(password) < 8:
            return jsonify(error="password must be at least 8 characters"), 400
        if role not in ("teacher", "student"):
            return jsonify(error="role must be 'teacher' or 'student'"), 400
        if db.get_user_by_email(email):
            return jsonify(error="an account with that email already exists"), 409

        class_code = generate_class_code() if role == "teacher" else None
        user_id = db.create_user(email, hash_password(password), role, display_name, class_code)
        token = db.create_token(user_id)
        resp = {"token": token, "role": role, "display_name": display_name}
        if class_code:
            resp["class_code"] = class_code
        return jsonify(resp), 201

    @app.route("/api/login", methods=["POST"])
    def api_login():
        data = request.get_json(force=True, silent=True) or {}
        email = (data.get("email") or "").strip()
        password = data.get("password") or ""
        user = db.get_user_by_email(email)
        if not user or not verify_password(password, user["password_hash"]):
            return jsonify(error="invalid email or password"), 401
        token = db.create_token(user["id"])
        resp = {"token": token, "role": user["role"], "display_name": user["display_name"]}
        if user["role"] == "teacher":
            resp["class_code"] = user["class_code"]
        return jsonify(resp)

    @app.route("/api/link", methods=["POST"])
    @require_api_auth(db, role="student")
    def api_link():
        data = request.get_json(force=True, silent=True) or {}
        class_code = (data.get("class_code") or "").strip().upper()
        teacher = db.get_teacher_by_class_code(class_code)
        if not teacher:
            return jsonify(error="no teacher found for that class code"), 404
        db.link_student(g.user["id"], teacher["id"])
        return jsonify(linked_to=teacher["display_name"])

    @app.route("/api/blocked-domains", methods=["GET"])
    @require_api_auth(db, role="student")
    def api_get_blocked_domains():
        teacher_id = db.get_teacher_for_student(g.user["id"])
        if not teacher_id:
            return jsonify(domains=[])
        domains = [d["domain"] for d in db.get_blocked_domains(teacher_id)]
        return jsonify(domains=domains)

    @app.route("/api/sessions", methods=["POST"])
    @require_api_auth(db, role="student")
    def api_upload_session():
        payload = request.get_json(force=True, silent=True) or {}
        for required in ("started_at", "ended_at"):
            if required not in payload:
                return jsonify(error=f"missing '{required}'"), 400
        session_id = db.insert_session(g.user["id"], payload)
        return jsonify(id=session_id), 201

    @app.route("/api/student/sessions", methods=["GET"])
    @require_api_auth(db, role="student")
    def api_own_sessions():
        return jsonify(sessions=db.get_sessions_for_student(g.user["id"]))

    @app.route("/api/teacher/students", methods=["GET"])
    @require_api_auth(db, role="teacher")
    def api_teacher_students():
        return jsonify(students=[
            {"id": s["id"], "email": s["email"], "display_name": s["display_name"]}
            for s in db.get_students_for_teacher(g.user["id"])
        ])

    @app.route("/api/teacher/students/<int:student_id>/sessions", methods=["GET"])
    @require_api_auth(db, role="teacher")
    def api_teacher_student_sessions(student_id):
        if db.get_teacher_for_student(student_id) != g.user["id"]:
            return jsonify(error="that student is not linked to you"), 403
        return jsonify(sessions=db.get_sessions_for_student(student_id))

    @app.route("/api/teacher/domains", methods=["GET", "POST"])
    @require_api_auth(db, role="teacher")
    def api_teacher_domains():
        if request.method == "POST":
            data = request.get_json(force=True, silent=True) or {}
            domain = (data.get("domain") or "").strip()
            if not domain:
                return jsonify(error="domain is required"), 400
            db.add_blocked_domain(g.user["id"], domain)
        return jsonify(domains=db.get_blocked_domains(g.user["id"]))

    @app.route("/api/teacher/domains/<int:domain_id>", methods=["DELETE"])
    @require_api_auth(db, role="teacher")
    def api_teacher_domain_delete(domain_id):
        db.remove_blocked_domain(g.user["id"], domain_id)
        return jsonify(ok=True)

    # ---------------------------------------------------------------- #
    # Web UI — browser, cookie session auth                            #
    # ---------------------------------------------------------------- #

    @app.route("/")
    def index():
        if not session.get("user_id"):
            return redirect(url_for("login"))
        user = db.get_user(session["user_id"])
        return redirect(url_for("teacher_home") if user["role"] == "teacher" else url_for("student_home"))

    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        if request.method == "GET":
            return render_template("signup.html", error=None)
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        role = request.form.get("role")
        display_name = request.form.get("display_name", "").strip() or email.split("@")[0]

        if not email or "@" not in email or len(password) < 8 or role not in ("teacher", "student"):
            return render_template("signup.html", error="Please fill in a valid email, a password "
                                    "of at least 8 characters, and a role.")
        if db.get_user_by_email(email):
            return render_template("signup.html", error="An account with that email already exists.")

        class_code = generate_class_code() if role == "teacher" else None
        user_id = db.create_user(email, hash_password(password), role, display_name, class_code)
        session["user_id"] = user_id
        return redirect(url_for("index"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET":
            return render_template("login.html", error=None)
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        user = db.get_user_by_email(email)
        if not user or not verify_password(password, user["password_hash"]):
            return render_template("login.html", error="Invalid email or password.")
        session["user_id"] = user["id"]
        return redirect(url_for("index"))

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/teacher")
    @require_web_auth(db, role="teacher")
    def teacher_home():
        students = db.get_students_for_teacher(g.user["id"])
        for s in students:
            sessions = db.get_sessions_for_student(s["id"], limit=1)
            s["last_session_at"] = sessions[0]["started_at"] if sessions else None
            s["session_count"] = len(db.get_sessions_for_student(s["id"], limit=10000))
        return render_template("teacher_home.html", user=g.user, students=students)

    @app.route("/teacher/domains", methods=["GET", "POST"])
    @require_web_auth(db, role="teacher")
    def teacher_domains():
        if request.method == "POST":
            domain = request.form.get("domain", "").strip()
            if domain:
                db.add_blocked_domain(g.user["id"], domain)
            return redirect(url_for("teacher_domains"))
        return render_template("teacher_domains.html", user=g.user,
                                domains=db.get_blocked_domains(g.user["id"]))

    @app.route("/teacher/domains/<int:domain_id>/delete", methods=["POST"])
    @require_web_auth(db, role="teacher")
    def teacher_domain_delete(domain_id):
        db.remove_blocked_domain(g.user["id"], domain_id)
        return redirect(url_for("teacher_domains"))

    @app.route("/teacher/students/<int:student_id>")
    @require_web_auth(db, role="teacher")
    def teacher_student_sessions(student_id):
        if db.get_teacher_for_student(student_id) != g.user["id"]:
            return redirect(url_for("teacher_home"))
        student = db.get_user(student_id)
        sessions = db.get_sessions_for_student(student_id)
        return render_template("teacher_student_sessions.html", user=g.user,
                                student=student, sessions=sessions)

    @app.route("/student")
    @require_web_auth(db, role="student")
    def student_home():
        sessions = db.get_sessions_for_student(g.user["id"])
        teacher_id = db.get_teacher_for_student(g.user["id"])
        teacher = db.get_user(teacher_id) if teacher_id else None
        return render_template("student_home.html", user=g.user, sessions=sessions, teacher=teacher)

    return app


def main():
    parser = argparse.ArgumentParser(description="Cognitive Monitor supervision backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5800)
    parser.add_argument("--db", default=None, help="Path to the SQLite database file")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(db_path=args.db)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
