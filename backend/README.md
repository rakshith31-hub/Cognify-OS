# Cognitive Monitor — supervision backend

Optional teacher/student layer on top of the daemon. Nothing here is
required for the daemon to work standalone — a student who never logs in
gets exactly the same local-only behavior as before.

## What it does

- **Accounts**: teachers and students sign up with email/password.
- **Linking**: each teacher gets a 6-character class code on signup.
  Students link to a teacher once, on their own machine, with
  `cognitive-monitor link <code>`.
- **Blocked domains**: a teacher manages one domain list (web UI:
  `/teacher/domains`). Every linked student fetches that list at the
  **start** of their next session and it's merged into that session's
  site blocking — it does not get pushed live into a session already in
  progress.
- **Session history**: at the end of every session, the student's daemon
  uploads a compact summary (state percentages, top apps, blocked-site
  attempt counts — never raw window-by-window predictions, camera data,
  or keystroke content). Teachers see it under a student's name; students
  see their own under `/student`.

## Running it

```bash
pip install -r ../requirements.txt   # flask + requests, already listed there
python app.py --host 0.0.0.0 --port 5800
```

For anything beyond local testing/a single classroom LAN, put this behind
a real WSGI server (gunicorn/waitress) and HTTPS, and set a stable
`COGMON_SECRET_KEY` environment variable (otherwise a new random one is
generated on every restart, which invalidates all browser sessions — API
tokens are unaffected since they're stored in the database, not signed
with the secret key).

The SQLite file defaults to `backend/cogmon_backend.db`; override with
`--db /path/to/file.db`.

## Student setup (once, per machine)

```bash
cognitive-monitor login --server https://your-backend-url --email you@school.edu --signup --role student
cognitive-monitor link ABC123          # the class code your teacher gives you
```

After that, `cognitive-monitor start` / `stop` automatically fetch the
teacher's domain list at each session start and upload a summary at each
session end — no extra steps.

## Teacher setup (once)

```bash
cognitive-monitor login --server https://your-backend-url --email you@school.edu --signup --role teacher
```

This prints your class code — share it with students. Manage the
blocked-domain list and view all linked students' session history by
logging into the same URL in a browser.

## What is NOT sent

Raw per-window predictions, camera frames, gaze/eye data, keystrokes, and
app-usage detail beyond top-8-by-time never leave the student's machine.
Only the same aggregate numbers already shown to the student themselves in
the local dashboard (`cognitive-monitor dashboard`) are uploaded.
