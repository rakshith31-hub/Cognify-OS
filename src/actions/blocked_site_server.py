"""
blocked_site_server.py
------------------------
Works together with ui_controller.block_domains(), which routes blocked
domains to 127.0.0.1 via the hosts file. On its own, that just gives the
browser a dead connection — this module makes that landing spot actually
useful:

  1. HTTP (port 80): serves an honest "you blocked this yourself" page
     instead of a generic connection-failed error, and logs the request
     (domain + path) to logs/blocked_attempts.csv.
  2. HTTPS (port 443): we don't have a valid TLS certificate for someone
     else's domain (nor should we try to forge one), so we can't serve a
     clean page here — the browser will show its own security warning.
     What we CAN do is read the domain name out of the TLS ClientHello's
     SNI (Server Name Indication) field BEFORE the handshake completes,
     log the attempt, then close the connection. This is standard,
     well-documented TLS record parsing (the same technique SNI-routing
     proxies use) — it reads only the domain name being requested, never
     any content, and only ever sees connections that were already routed
     to this machine's own loopback address by its own hosts file.

Because the hosts file ONLY routes domains the user explicitly listed to
127.0.0.1, any connection this server receives is, by definition, an
attempt to reach a self-blocked domain — no blocklist lookup needed here.

Requires binding to ports 80/443, which needs admin/root — exactly the
same privilege level already required to edit the hosts file. If binding
fails (not elevated, or ports already in use by something else), this
logs a clear warning and the daemon continues normally without it; the
hosts-file block itself still works, you just get a browser error page
instead of our friendlier one.
"""

import logging
import os
import socket
import threading
import time

logger = logging.getLogger("cognitive_monitor.blocked_site_server")

HTTP_PORT = 80
HTTPS_PORT = 443
SOCKET_TIMEOUT = 3.0

_BLOCK_PAGE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Blocked — Cognitive Monitor</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; background: #1E2749;
         color: #E7ECF5; display: flex; align-items: center; justify-content: center;
         height: 100vh; margin: 0; }}
  .card {{ max-width: 480px; text-align: center; padding: 2rem; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.5rem; }}
  p {{ color: #AEB6D0; line-height: 1.5; }}
  .domain {{ font-family: Consolas, monospace; color: #5FE0D0; }}
</style>
</head>
<body>
  <div class="card">
    <h1>You blocked this yourself</h1>
    <p><span class="domain">{domain}</span> is on your own distraction list right now,
    for the next block window you set. This isn't someone else's rule — you added it.</p>
    <p>This attempt has been logged, same as every other one, so you can see the pattern later.</p>
  </div>
</body>
</html>
"""


class BlockedSiteServer:
    def __init__(self, attempt_logger):
        self.attempt_logger = attempt_logger
        self._http_thread = None
        self._https_thread = None
        self._running = False

    def start(self):
        self._running = True
        self._http_thread = threading.Thread(target=self._run_http_server, daemon=True)
        self._https_thread = threading.Thread(target=self._run_https_sniffer, daemon=True)
        self._http_thread.start()
        self._https_thread.start()

    def stop(self):
        self._running = False
        # Both loops use short accept() timeouts, so they'll notice
        # `_running` flipping within SOCKET_TIMEOUT seconds and exit
        # cleanly on their own; nothing further to signal here.

    # ------------------------------------------------------------------
    # HTTP: real block page + logging
    # ------------------------------------------------------------------

    def _run_http_server(self):
        sock = self._bind(HTTP_PORT)
        if sock is None:
            return
        logger.info("Blocked-site HTTP page listening on 127.0.0.1:%d", HTTP_PORT)
        while self._running:
            try:
                sock.settimeout(SOCKET_TIMEOUT)
                conn, _addr = sock.accept()
            except socket.timeout:
                continue
            except Exception as e:
                logger.debug("HTTP accept loop error (non-fatal): %s", e)
                continue
            threading.Thread(target=self._handle_http_conn, args=(conn,), daemon=True).start()
        sock.close()

    def _handle_http_conn(self, conn: socket.socket):
        try:
            conn.settimeout(SOCKET_TIMEOUT)
            request = conn.recv(4096).decode("utf-8", errors="replace")
            domain, path = self._parse_http_request(request)
            self.attempt_logger.log(domain or "unknown", "http", path)

            body = _BLOCK_PAGE_TEMPLATE.format(domain=domain or "this site").encode("utf-8")
            response = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/html; charset=utf-8\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                b"Connection: close\r\n\r\n" + body
            )
            conn.sendall(response)
        except Exception as e:
            logger.debug("HTTP block-page handling failed (non-fatal): %s", e)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    @staticmethod
    def _parse_http_request(request: str):
        """Extract Host header and request path from a raw HTTP request."""
        domain, path = None, ""
        lines = request.split("\r\n")
        if lines:
            parts = lines[0].split(" ")
            if len(parts) >= 2:
                path = parts[1]
        for line in lines[1:]:
            if line.lower().startswith("host:"):
                domain = line.split(":", 1)[1].strip()
                break
        return domain, path

    # ------------------------------------------------------------------
    # HTTPS: SNI-only sniff + log, then close (no fake certificate)
    # ------------------------------------------------------------------

    def _run_https_sniffer(self):
        sock = self._bind(HTTPS_PORT)
        if sock is None:
            return
        logger.info("Blocked-site HTTPS attempt logger listening on 127.0.0.1:%d "
                    "(browser will show its own security warning — that's expected, "
                    "we don't forge certificates; the attempt is still logged).", HTTPS_PORT)
        while self._running:
            try:
                sock.settimeout(SOCKET_TIMEOUT)
                conn, _addr = sock.accept()
            except socket.timeout:
                continue
            except Exception as e:
                logger.debug("HTTPS accept loop error (non-fatal): %s", e)
                continue
            threading.Thread(target=self._handle_https_conn, args=(conn,), daemon=True).start()
        sock.close()

    def _handle_https_conn(self, conn: socket.socket):
        try:
            conn.settimeout(SOCKET_TIMEOUT)
            client_hello = conn.recv(4096)
            domain = self._parse_sni(client_hello)
            self.attempt_logger.log(domain or "unknown", "https", "")
        except Exception as e:
            logger.debug("HTTPS SNI parsing failed (non-fatal): %s", e)
        finally:
            try:
                conn.close()  # no valid cert to offer; close rather than hang
            except Exception:
                pass

    @staticmethod
    def _parse_sni(data: bytes):
        """
        Best-effort parse of the SNI (Server Name Indication) extension out
        of a raw TLS ClientHello. Standard TLS record structure — reads
        only the domain name field, nothing else. Returns None on any
        malformed/unexpected input rather than raising.
        """
        try:
            if len(data) < 5 or data[0] != 0x16:  # not a TLS handshake record
                return None
            pos = 5  # skip record header (type, version x2, length x2)
            if data[pos] != 0x01:  # not a ClientHello
                return None
            pos += 4  # skip handshake header (type, length x3)
            pos += 2 + 32  # skip client version + random
            session_id_len = data[pos]
            pos += 1 + session_id_len
            cipher_suites_len = int.from_bytes(data[pos:pos + 2], "big")
            pos += 2 + cipher_suites_len
            compression_len = data[pos]
            pos += 1 + compression_len
            if pos >= len(data):
                return None
            extensions_len = int.from_bytes(data[pos:pos + 2], "big")
            pos += 2
            end = pos + extensions_len
            while pos < end:
                ext_type = int.from_bytes(data[pos:pos + 2], "big")
                ext_len = int.from_bytes(data[pos + 2:pos + 4], "big")
                ext_data = data[pos + 4:pos + 4 + ext_len]
                if ext_type == 0x0000:  # server_name extension
                    # server_name_list: 2-byte list len, then 1-byte type + 2-byte len + name
                    name_len = int.from_bytes(ext_data[3:5], "big")
                    return ext_data[5:5 + name_len].decode("utf-8", errors="replace")
                pos += 4 + ext_len
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------

    @staticmethod
    def _bind(port: int):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            sock.listen(20)
            return sock
        except PermissionError:
            logger.warning(
                "Cannot bind port %d (needs admin/root — same privilege level "
                "already required for hosts-file editing). Blocked-domain "
                "logging/page disabled; the hosts-file block itself still works, "
                "browsers will just show their own connection error instead.",
                port,
            )
            return None
        except OSError as e:
            logger.warning("Cannot bind port %d (%s) — likely already in use by "
                            "another local service. Blocked-domain page/logging disabled.",
                            port, e)
            return None
