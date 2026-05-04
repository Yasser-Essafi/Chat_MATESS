"""
Assistant Tourisme Maroc Flask Web Server
========================
REST API + Static Frontend for the Assistant Tourisme Maroc chatbot.

Install:
    pip install flask

Run:
    python server.py

Open: http://localhost:5000
"""

import os
import re
import sys
import time
import uuid
import threading
import traceback
from collections import deque
from datetime import datetime
from typing import Optional

# ── Path setup ──
# server.py lives one level above chatbotfinal/ — resolve the actual code root
_SERVER_DIR  = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT   = os.path.join(_SERVER_DIR, "chatbotfinal")
PROJECT_ROOT = _CODE_ROOT if os.path.isdir(_CODE_ROOT) else _SERVER_DIR
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, request, jsonify, send_file, send_from_directory, abort, g

# ── Security & resource limits (override via env vars in production) ──
MAX_MESSAGE_CHARS = int(os.environ.get("MAX_MESSAGE_CHARS", "4000"))
MAX_REQUEST_BYTES = int(os.environ.get("MAX_REQUEST_BYTES", str(64 * 1024)))  # 64 KB
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("RATE_LIMIT_WINDOW_SEC", "60"))
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "20"))  # requests / window / IP
# CORS allowlist — comma-separated origins. Defaults to common dev hosts.
# Set to "*" only when you intentionally want a public API.
_default_origins = "http://localhost:5000,http://127.0.0.1:5000"
ALLOWED_ORIGINS = set(
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",")
    if o.strip()
)
ALLOW_ANY_ORIGIN = "*" in ALLOWED_ORIGINS

app = Flask(__name__)
# Reject oversized bodies at the WSGI layer (cheap DoS guard).
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES

# ── In-memory per-IP rate limiter (dev-grade — for multi-process use Redis) ──
_rate_lock = threading.Lock()
_rate_history: dict = {}  # ip → deque[timestamps]


def _rate_limited(ip: str) -> bool:
    """Return True when the caller has exceeded the per-IP request budget.

    Why in-memory: this server runs as a single process. For gunicorn/uWSGI
    with multiple workers, swap this for a Redis-backed limiter (e.g.
    flask-limiter with a Redis storage URI).
    """
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SEC
    with _rate_lock:
        hist = _rate_history.get(ip)
        if hist is None:
            hist = deque()
            _rate_history[ip] = hist
        # Drop timestamps outside the window
        while hist and hist[0] < cutoff:
            hist.popleft()
        if len(hist) >= RATE_LIMIT_MAX:
            return True
        hist.append(now)
    return False


# ── Inline chart-path extractor (avoids importing Streamlit components) ──
_CHARTS_DIR = os.path.join(PROJECT_ROOT, "charts")

def extract_chart_path(text: str) -> Optional[str]:
    """Extract chart path from LLM response, restricted to the charts/ directory."""
    if not text:
        return None
    patterns = [
        r"📊\s*Chart:\s*(.+\.html)",
        r"Chart:\s*(.+\.html)",
        r"((?:[^\s]*/charts/|charts/)[^\s]+\.html)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            path = match.group(1).strip()
            # Resolve to absolute path
            if not os.path.isabs(path):
                path = os.path.join(PROJECT_ROOT, path)
            # Security: must be inside the charts/ directory
            real_path = os.path.realpath(path)
            real_charts = os.path.realpath(_CHARTS_DIR)
            if real_path.startswith(real_charts) and os.path.exists(real_path):
                return real_path
    return None

# ── Request ID + CORS allowlist + security headers ──
@app.before_request
def _attach_request_id():
    """Stamp every request with a short ID for log correlation."""
    g.request_id = uuid.uuid4().hex[:8]
    g.request_started_at = time.time()


@app.after_request
def _security_headers(response):
    """Add CORS (origin allowlist) and standard security headers."""
    origin = request.headers.get("Origin", "")
    if ALLOW_ANY_ORIGIN:
        response.headers["Access-Control-Allow-Origin"] = "*"
    elif origin and origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    # Methods/headers always advertised so preflight succeeds for allowed origins
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"

    # OWASP-recommended baseline security headers
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    # Echo request ID back to clients so they can attach it to bug reports
    if hasattr(g, "request_id"):
        response.headers["X-Request-ID"] = g.request_id
    return response


@app.route("/api/<path:path>", methods=["OPTIONS"])
def _options(path):
    return "", 204


# ── Generic error handlers (don't leak exception detail to clients) ──
@app.errorhandler(413)
def _too_large(_e):
    return jsonify({"error": "request_too_large",
                    "max_bytes": MAX_REQUEST_BYTES}), 413


@app.errorhandler(429)
def _rate_limited_handler(_e):
    return jsonify({"error": "rate_limited",
                    "retry_after_seconds": RATE_LIMIT_WINDOW_SEC}), 429


@app.errorhandler(500)
def _internal_error(_e):
    rid = getattr(g, "request_id", "?")
    return jsonify({"error": "internal_error", "request_id": rid}), 500


# ══════════════════════════════════════════════════════════════════════════════
# Lazy singletons
# ══════════════════════════════════════════════════════════════════════════════

_orchestrator = None
_session_manager = None
_init_lock = threading.Lock()


def _get_orch():
    global _orchestrator
    if _orchestrator is None:
        with _init_lock:
            if _orchestrator is None:
                from agents.orchestrator import Orchestrator
                _orchestrator = Orchestrator()
    return _orchestrator


def _get_mgr():
    global _session_manager
    if _session_manager is None:
        with _init_lock:
            if _session_manager is None:
                from ui.state.session import SessionManager
                _session_manager = SessionManager()
    return _session_manager


# ══════════════════════════════════════════════════════════════════════════════
# Static files
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
    if os.path.exists(path):
        return send_file(path)
    return "<h2>Assistant Tourisme Maroc</h2><p>Place <code>frontend/index.html</code> in the project root.</p>", 200


@app.route("/charts/<path:filename>")
def serve_chart(filename):
    charts_dir = os.path.join(PROJECT_ROOT, "charts")
    try:
        return send_from_directory(charts_dir, filename)
    except Exception:
        abort(404)


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(PROJECT_ROOT, "ui", "assets"), filename)


# ══════════════════════════════════════════════════════════════════════════════
# API: Health & status
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "platform": "Assistant Tourisme Maroc", "ts": datetime.now().isoformat()})


@app.route("/api/status")
def status():
    try:
        orch = _get_orch()
        return jsonify({
            "status": "ready",
            "datasets": len(orch.analytics_agent.datasets),
            "year_range": {"min": orch.min_year, "max": orch.max_year},
            "last_agent": orch.last_agent,
            "message_count": orch.message_count,
        })
    except Exception:
        rid = getattr(g, "request_id", "?")
        traceback.print_exc()
        return jsonify({"status": "error", "request_id": rid}), 500


# ══════════════════════════════════════════════════════════════════════════════
# API: Database / Fabric status
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/config/database/status")
def database_status():
    """Return the active database layer source and connection status."""
    try:
        from utils.db_layer import DBLayer
        db = DBLayer()
        return jsonify({
            "source": db.source,
            "status": db.status,
            "connected": db.source == "fabric",
            "ping": db.ping() if db.source == "fabric" else None,
        })
    except Exception:
        rid = getattr(g, "request_id", "?")
        traceback.print_exc()
        return jsonify({"source": "unknown", "status": "error",
                        "connected": False, "request_id": rid}), 500


@app.route("/api/config/database/test")
def database_test():
    """Run a deeper Fabric connectivity check: env vars present, AAD token
    obtainable, SQL endpoint reachable, default tables visible. Useful right
    after configuring credentials."""
    rid = getattr(g, "request_id", "?")
    try:
        from utils.db_layer import DBLayer
        from config.settings import (
            FABRIC_SQL_ENDPOINT, FABRIC_DATABASE, FABRIC_TABLES,
            AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET,
            FABRIC_ENABLED,
        )
        report = {
            "request_id": rid,
            "fabric_enabled_in_env": FABRIC_ENABLED,
            "env_vars": {
                "FABRIC_SQL_ENDPOINT": bool(FABRIC_SQL_ENDPOINT),
                "FABRIC_DATABASE": bool(FABRIC_DATABASE),
                "AZURE_TENANT_ID": bool(AZURE_TENANT_ID),
                "AZURE_CLIENT_ID": bool(AZURE_CLIENT_ID),
                "AZURE_CLIENT_SECRET": bool(AZURE_CLIENT_SECRET),
            },
            "expected_tables": FABRIC_TABLES,
        }
        if not FABRIC_ENABLED:
            report["result"] = "fabric_not_configured"
            report["next_steps"] = [
                "Add FABRIC_SQL_ENDPOINT, FABRIC_DATABASE, AZURE_TENANT_ID, "
                "AZURE_CLIENT_ID and AZURE_CLIENT_SECRET to chatbotfinal/.env"
            ]
            return jsonify(report), 200

        db = DBLayer()
        report["dblayer_source"] = db.source
        report["dblayer_status"] = db.status

        if db.source != "fabric":
            report["result"] = "connection_failed"
            return jsonify(report), 200

        # Probe each expected table (use schema-aware qualifier)
        table_status = {}
        for tbl in FABRIC_TABLES:
            qualified = db._qualify(tbl)
            try:
                row_count = db.query_df(f"SELECT COUNT(*) AS n FROM {qualified}")
                n = int(row_count.iloc[0, 0]) if not row_count.empty else 0
                table_status[tbl] = {"reachable": True,
                                     "row_count": n,
                                     "qualified_name": qualified}
            except Exception as e:
                table_status[tbl] = {"reachable": False,
                                     "qualified_name": qualified,
                                     "error": str(e)[:200]}

        report["tables"] = table_status
        all_ok = all(t.get("reachable") for t in table_status.values())
        report["result"] = "ok" if all_ok else "partial"
        return jsonify(report), 200

    except Exception:
        traceback.print_exc()
        return jsonify({"result": "error", "request_id": rid}), 500


# ══════════════════════════════════════════════════════════════════════════════
# API: Conversations
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    return jsonify(_get_mgr().get_conversation_list())


@app.route("/api/conversations", methods=["POST"])
def new_conversation():
    mgr = _get_mgr()
    try:
        orch = _get_orch()
        orch.last_agent = None
        orch.conversation_log.clear()
        orch.message_count = 0
        orch.routing_history.clear()
        orch._session_summary = ""
        orch._last_summarized_at = 0
        orch._classify_cache.clear()
    except Exception:
        pass
    conv = mgr.new_conversation()
    return jsonify({"id": conv.id, "title": conv.title, "created_at": conv.created_at})


@app.route("/api/conversations/<cid>", methods=["GET"])
def get_conversation(cid):
    conv = _get_mgr().conversations.get(cid)
    if not conv:
        return jsonify({"error": "Not found"}), 404
    return jsonify(conv.to_dict())


@app.route("/api/conversations/<cid>/activate", methods=["POST"])
def activate_conversation(cid):
    conv = _get_mgr().switch_conversation(cid)
    if not conv:
        return jsonify({"error": "Not found"}), 404
    try:
        orch = _get_orch()
        orch.last_agent = None
        orch.conversation_log.clear()
        orch.message_count = 0
        orch.routing_history.clear()
        orch._session_summary = ""
        orch._last_summarized_at = 0
        orch._classify_cache.clear()
    except Exception:
        pass
    return jsonify({"success": True})


@app.route("/api/conversations/<cid>", methods=["DELETE"])
def delete_conversation(cid):
    if _get_mgr().delete_conversation(cid):
        return jsonify({"success": True})
    return jsonify({"error": "Not found"}), 404


@app.route("/api/conversations/clear", methods=["POST"])
def clear_all():
    mgr = _get_mgr()
    count = mgr.clear_all_history()
    try:
        orch = _get_orch()
        orch.last_agent = None
        orch.conversation_log.clear()
        orch.routing_history.clear()
        orch.message_count = 0
    except Exception:
        pass
    return jsonify({"cleared": count})


# ══════════════════════════════════════════════════════════════════════════════
# API: Chat
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
def chat():
    # ── Rate limit (per-IP) ──
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0].strip()
    if _rate_limited(client_ip):
        return jsonify({"error": "rate_limited",
                        "retry_after_seconds": RATE_LIMIT_WINDOW_SEC}), 429

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    cid = data.get("conversation_id")

    if not message:
        return jsonify({"error": "message_required"}), 400

    # ── Length validation (cost-DoS guard, OWASP LLM01) ──
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify({"error": "message_too_long",
                        "max_chars": MAX_MESSAGE_CHARS,
                        "received_chars": len(message)}), 400

    try:
        from ui.state.session import Message

        mgr = _get_mgr()
        orch = _get_orch()

        # Activate conversation
        if cid:
            mgr.switch_conversation(cid)
        if mgr.get_active_conversation() is None:
            conv = mgr.new_conversation()
            cid = conv.id
        else:
            cid = mgr.active_conversation_id

        # Save user message
        mgr.add_message_to_active(Message(role="user", content=message))

        # ── Orchestrator commands: /help, /agents, /history, /cache, /resetall ──
        cmd_result = orch.handle_orchestrator_commands(message)
        if cmd_result:
            am = Message(role="assistant", content=cmd_result,
                         agent="normal", agent_name="Orchestrateur", agent_icon="🎯")
            mgr.add_message_to_active(am)
            return jsonify(_resp("normal", "🎯", "Orchestrateur",
                                 cmd_result, False, 0.0, 0.0, None, cid))

        # ── Analytics quick commands ──
        lower = message.lower().strip()
        analytics_cmds = {
            "/datasets": lambda: orch.analytics_agent.list_datasets(),
            "/stats":    lambda: orch.analytics_agent.quick_stats(),
            "/schema":   lambda: orch.analytics_agent.get_schema(),
            "/columns":  lambda: orch.analytics_agent.get_columns(),
            "/sample":   lambda: orch.analytics_agent.get_sample(),
        }
        if lower in analytics_cmds:
            text = analytics_cmds[lower]()
            am = Message(role="assistant", content=text, agent="analytics",
                         agent_name="Analyste de Données", agent_icon="📊")
            mgr.add_message_to_active(am)
            return jsonify(_resp("analytics", "📊", "Analyste de Données",
                                 text, False, 0.0, 0.0, None, cid))

        if lower.startswith("/switch "):
            ds_name = message.split(None, 1)[1]
            text = orch.analytics_agent.switch_dataset(ds_name)
            am = Message(role="assistant", content=text, agent="analytics",
                         agent_name="Analyste de Données", agent_icon="📊")
            mgr.add_message_to_active(am)
            return jsonify(_resp("analytics", "📊", "Analyste de Données",
                                 text, False, 0.0, 0.0, None, cid))

        # ── Normal routing ──
        result = orch.route(message)

        chart_path = extract_chart_path(result["response"])
        chart_url = None
        if chart_path and os.path.exists(chart_path):
            chart_url = "/charts/" + os.path.basename(chart_path)

        am = Message(
            role="assistant",
            content=result["response"],
            agent=result["agent"],
            agent_icon=result["agent_icon"],
            agent_name=result["agent_name"],
            chart_path=chart_path,
            rerouted=result["rerouted"],
            classification_time_ms=result["classification_time_ms"],
            total_time_ms=result["total_time_ms"],
        )
        mgr.add_message_to_active(am)

        return jsonify(_resp(
            result["agent"], result["agent_icon"], result["agent_name"],
            result["response"], result["rerouted"],
            result["classification_time_ms"], result["total_time_ms"],
            chart_url, cid,
        ))

    except Exception:
        # Log the full trace server-side, return only an opaque error to client
        rid = getattr(g, "request_id", "?")
        print(f"[{rid}] /api/chat failed:")
        traceback.print_exc()
        return jsonify({"error": "chat_failed", "request_id": rid}), 500


def _resp(agent, icon, name, response, rerouted, cls_ms, tot_ms, chart_url, cid):
    return {
        "agent": agent,
        "agent_icon": icon,
        "agent_name": name,
        "response": response,
        "rerouted": rerouted,
        "classification_time_ms": cls_ms,
        "total_time_ms": tot_ms,
        "chart_url": chart_url,
        "conversation_id": cid,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    print("\n" + "=" * 58)
    print("  Assistant Tourisme Maroc -- Ministère du Tourisme")
    print("  Ministere du Tourisme, Artisanat & Economie Sociale")
    print("=" * 58 + "\n")

    print("Initialisation du système...")
    try:
        _get_orch()
        _get_mgr()
        print("✅ Système prêt!\n")
    except Exception as exc:
        print(f"⚠️  Avertissement lors de l'init: {exc}")
        print("   Le système réessaiera à la première requête.\n")

    print("🌐 Serveur disponible sur: http://localhost:5000")
    print("   Appuyez sur Ctrl+C pour arrêter.\n")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
