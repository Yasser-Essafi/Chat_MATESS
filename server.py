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
import json
import html
import ipaddress
import base64
import mimetypes
import threading
import traceback
from queue import Queue, Empty
from collections import deque
from datetime import datetime
from typing import Optional

# ── Path setup ──
# server.py lives one level above chatbotfinal/ — resolve the actual code root
_SERVER_DIR  = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT   = os.path.join(_SERVER_DIR, "chatbotfinal")
PROJECT_ROOT = _CODE_ROOT if os.path.isdir(_CODE_ROOT) else _SERVER_DIR
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, Response, request, jsonify, send_file, send_from_directory, abort, g
from utils.logger import get_logger

logger = get_logger("server")

# ── Security & resource limits (override via env vars in production) ──
MAX_MESSAGE_CHARS = int(os.environ.get("MAX_MESSAGE_CHARS", "4000"))
MAX_REQUEST_BYTES = int(os.environ.get("MAX_REQUEST_BYTES", str(64 * 1024)))  # 64 KB
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("RATE_LIMIT_WINDOW_SEC", "60"))
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "20"))  # requests / window / IP
TRUSTED_PROXY_IPS = {
    ip.strip()
    for ip in os.environ.get("TRUSTED_PROXY_IPS", "127.0.0.1,::1").split(",")
    if ip.strip()
}
ADMIN_TOKEN = os.environ.get("STATOUR_ADMIN_TOKEN", "").strip()
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


def create_app() -> Flask:
    """Return the configured Flask app for WSGI servers and tests."""
    _start_background_warmup()
    return app

# ── In-memory per-IP rate limiter (dev-grade — for multi-process use Redis) ──
_rate_lock = threading.Lock()
 
_rate_history: dict = {}  # ip → deque[timestamps]


def _is_loopback_ip(ip: Optional[str]) -> bool:
    try:
        return ipaddress.ip_address((ip or "").strip()).is_loopback
    except ValueError:
        return False


def _client_ip() -> str:
    """Return the rate-limit identity, trusting XFF only from configured proxies."""
    remote = (request.remote_addr or "?").strip()
    xff = request.headers.get("X-Forwarded-For", "")
    if remote in TRUSTED_PROXY_IPS and xff:
        first = xff.split(",")[0].strip()
        return first or remote
    return remote


def _admin_allowed() -> bool:
    if _is_loopback_ip(request.remote_addr):
        return True
    token = request.headers.get("X-Admin-Token", "").strip()
    auth = request.headers.get("Authorization", "").strip()
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    return bool(ADMIN_TOKEN and token and token == ADMIN_TOKEN)


def _admin_required_response():
    return jsonify({"error": "admin_required", "request_id": getattr(g, "request_id", "?")}), 403


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
_CHARTS_DIR = os.environ.get("STATOUR_CHARTS_DIR", os.path.join(PROJECT_ROOT, "charts"))

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
            real_path = _resolve_chart_path(match.group(1))
            if real_path:
                return real_path
    return None


def _resolve_chart_path(path: str) -> Optional[str]:
    if not path:
        return None
    raw = path.strip().strip("`),.;").replace("\\", "/")
    candidate = raw if os.path.isabs(raw) else os.path.join(_CHARTS_DIR, os.path.basename(raw))
    real_path = os.path.realpath(candidate)
    real_charts = os.path.realpath(_CHARTS_DIR)
    if real_path.startswith(real_charts + os.sep) and os.path.exists(real_path):
        return real_path
    return None


def extract_chart_paths(text: str, limit: int = 4) -> list[str]:
    """Extract all chart paths from text, restricted to the charts/ directory."""
    if not text:
        return []
    patterns = [
        r"ðŸ“Š\s*Chart:\s*(.+?\.html)",
        r"📊\s*Chart:\s*(.+?\.html)",
        r"Chart:\s*(.+?\.html)",
        r"((?:[^\s]*/charts/|charts/)[^\s]+\.html)",
    ]
    paths: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            real_path = _resolve_chart_path(match.group(1))
            if real_path and real_path not in paths:
                paths.append(real_path)
                if len(paths) >= limit:
                    return paths
    return paths

def _sanitize_response(text: str) -> str:
    """Defense-in-depth: strip internal references before sending to client."""
    if not text:
        return text
    text = re.sub(r"\[dbo_GOLD\]\.\[[^\]]+\]", "", text)
    text = re.sub(r"(?i)\b(table\s+)?fact_statistiques\w*", "", text)
    text = re.sub(r"(?i)\bgld_dim_\w+", "", text)
    text = re.sub(r"(?m)^.*(?:Chart:|charts[/\\])\S*\.html.*$\n?", "", text)
    text = re.sub(r"[A-Za-z]:[/\\](?:Users|home)[/\\]\S+", "", text)
    text = re.sub(
        r"(?im)^(?:NOTE|NB)\s*:.*(?:fact_|dbo_GOLD|table |eht_id|nationalite_name).*$\n?",
        "",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
    # Charts and generated reports are embedded in the main UI via same-origin iframes.
    # Firefox blocks those when X-Frame-Options is DENY, so we allow same-origin framing.
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Content-Security-Policy", "frame-ancestors 'self'")
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
_warmup_lock = threading.Lock()
_warmup_started = False
_warmup_done = False


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


def _start_background_warmup() -> None:
    """Initialize the heavy singletons without blocking health/readiness."""
    global _warmup_started, _warmup_done
    if _warmup_started or _warmup_done:
        return
    with _warmup_lock:
        if _warmup_started or _warmup_done:
            return
        _warmup_started = True

    def _warmup():
        global _warmup_done
        try:
            orch = _get_orch()
            _get_mgr()
            try:
                from utils.mvp_services import get_readiness

                get_readiness(orch)
            except Exception:
                logger.debug("Readiness warmup skipped", exc_info=True)
            _warmup_done = True
            logger.info("Background warmup complete")
        except Exception as exc:
            with _warmup_lock:
                globals()["_warmup_started"] = False
            logger.warning("Background warmup failed: %s", exc)

    threading.Thread(target=_warmup, daemon=True, name="statour-warmup").start()


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
    chart_path = _resolve_chart_path(filename)
    if not chart_path:
        abort(404)
    return send_from_directory(_CHARTS_DIR, os.path.basename(chart_path))


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(PROJECT_ROOT, "ui", "assets"), filename)


# ══════════════════════════════════════════════════════════════════════════════
# API: Health & status
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health")
def health():
    _start_background_warmup()
    return jsonify({"status": "ok", "platform": "Assistant Tourisme Maroc", "ts": datetime.now().isoformat()})


@app.route("/api/status")
def status():
    if not _admin_allowed():
        return _admin_required_response()
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
        logger.debug("Exception detail:", exc_info=True)
        return jsonify({"status": "error", "request_id": rid}), 500


@app.route("/api/readiness")
def readiness():
    """Executive MVP readiness: dependencies, Fabric, RAG and search."""
    if not _admin_allowed():
        return _admin_required_response()
    try:
        from utils.mvp_services import get_readiness
        return jsonify(get_readiness(_get_orch()))
    except Exception:
        rid = getattr(g, "request_id", "?")
        logger.debug("Exception detail:", exc_info=True)
        return jsonify({"ready": False, "error": "readiness_failed", "request_id": rid}), 500


def _public_readiness_payload(readiness: dict) -> dict:
    fabric = readiness.get("fabric") or {}
    latest = readiness.get("latest_data") or {}
    return {
        "ready": bool(readiness.get("ready")),
        "checked_at": readiness.get("checked_at"),
        "fabric": {
            "connected": bool(fabric.get("connected")),
            "table_count": len(fabric.get("tables") or {}),
        },
        "latest_data": latest,
        "rag": readiness.get("rag") or {"available": False, "chunks": 0},
        "search": readiness.get("search") or {"available": False, "exa_available": False},
        "blockers": readiness.get("blockers") or [],
    }


def _public_readiness_snapshot() -> dict:
    from config.settings import (
        AZURE_OPENAI_API_KEY,
        BRAVE_API_KEY,
        DOCUMENTS_DIR,
        EXA_API_KEY,
        FABRIC_ENABLED,
        FABRIC_TABLES,
        TAVILY_API_KEY,
        VECTORSTORE_DIR,
    )

    fabric_ok = bool(FABRIC_ENABLED and FABRIC_TABLES)
    search_ok = bool(TAVILY_API_KEY or EXA_API_KEY or BRAVE_API_KEY)
    ai_ok = bool(AZURE_OPENAI_API_KEY)
    rag_ok = False
    try:
        rag_ok = (
            os.path.isdir(VECTORSTORE_DIR) and any(os.scandir(VECTORSTORE_DIR))
        ) or (
            os.path.isdir(DOCUMENTS_DIR)
            and any(name.lower().endswith(".md") for name in os.listdir(DOCUMENTS_DIR))
        )
    except Exception:
        rag_ok = False

    blockers = []
    if not fabric_ok:
        blockers.append("Fabric non configuré")
    if not ai_ok:
        blockers.append("Azure OpenAI non configuré")
    if not search_ok:
        blockers.append("Recherche non configurée")
    if not rag_ok:
        blockers.append("Base documentaire non détectée")

    return {
        "ready": fabric_ok and ai_ok and search_ok and rag_ok,
        "checked_at": datetime.now().isoformat(),
        "fabric": {"connected": fabric_ok, "table_count": len(FABRIC_TABLES)},
        "latest_data": {},
        "rag": {"available": rag_ok, "chunks": None},
        "search": {"available": search_ok, "exa_available": bool(EXA_API_KEY)},
        "blockers": blockers,
    }


@app.route("/api/public/readiness")
def public_readiness():
    """Public readiness summary for the frontend status bar.

    The detailed diagnostics remain behind /api/readiness with the admin token.
    """
    _start_background_warmup()
    try:
        return jsonify(_public_readiness_snapshot())
    except Exception:
        rid = getattr(g, "request_id", "?")
        logger.debug("Exception detail:", exc_info=True)
        return jsonify({"ready": False, "error": "readiness_failed", "request_id": rid}), 500


@app.route("/api/dashboard/summary")
def dashboard_summary():
    """Return executive dashboard KPI cards, charts, signals and data freshness."""
    try:
        from utils.mvp_services import get_dashboard_summary
        return jsonify(get_dashboard_summary(_get_orch()))
    except Exception:
        rid = getattr(g, "request_id", "?")
        logger.debug("Exception detail:", exc_info=True)
        return jsonify({"status": "error", "request_id": rid}), 500


@app.route("/api/insights", methods=["POST"])
def executive_insights():
    """Dedicated executive insight endpoint."""
    client_ip = _client_ip()
    if _rate_limited(client_ip):
        return jsonify({"error": "rate_limited",
                        "retry_after_seconds": RATE_LIMIT_WINDOW_SEC}), 429

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    cid = data.get("conversation_id")
    if not message:
        return jsonify({"error": "message_required"}), 400
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify({"error": "message_too_long",
                        "max_chars": MAX_MESSAGE_CHARS,
                        "received_chars": len(message)}), 400

    try:
        return jsonify(_process_chat_turn(message, cid, save_user=True))
    except Exception:
        rid = getattr(g, "request_id", "?")
        logger.exception("[%s] /api/insights failed", rid)
        return jsonify({"error": "insight_failed", "request_id": rid}), 500


# ══════════════════════════════════════════════════════════════════════════════
# API: Database / Fabric status
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/config/database/status")
def database_status():
    """Return the active database layer source and connection status."""
    if not _admin_allowed():
        return _admin_required_response()
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
        logger.debug("Exception detail:", exc_info=True)
        return jsonify({"source": "unknown", "status": "error",
                        "connected": False, "request_id": rid}), 500


@app.route("/api/config/database/test")
def database_test():
    """Run a deeper Fabric connectivity check: env vars present, AAD token
    obtainable, SQL endpoint reachable, default tables visible. Useful right
    after configuring credentials."""
    if not _admin_allowed():
        return _admin_required_response()
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
        logger.debug("Exception detail:", exc_info=True)
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
    conv = mgr.new_conversation()
    try:
        orch = _get_orch()
        orch.reset_conversation_state(conv.id)
    except Exception:
        pass
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
        orch.load_conversation_state(cid, conv.messages)
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
        orch.clear_runtime_states()
    except Exception:
        pass
    return jsonify({"cleared": count})


def _reload_orchestrator_state(cid: str) -> None:
    try:
        mgr = _get_mgr()
        conv = mgr.get_conversation(cid)
        orch = _get_orch()
        if conv:
            orch.load_conversation_state(cid, conv.messages)
        else:
            orch.reset_conversation_state(cid)
    except Exception:
        logger.debug("Exception detail:", exc_info=True)


def _chart_urls_from_paths(chart_paths: list[str]) -> list[str]:
    urls = []
    for path in chart_paths or []:
        url = _chart_url_from_path(path)
        if url and url not in urls:
            urls.append(url)
    return urls


def _result_chart_paths(result: dict, message: str, conv=None) -> list[str]:
    chart_paths: list[str] = []

    def add(path):
        if path and os.path.exists(path) and path not in chart_paths:
            chart_paths.append(path)

    for path in result.get("chart_paths") or []:
        add(path)
    add(result.get("chart_path"))
    for path in extract_chart_paths(result.get("response", "")):
        add(path)

    if not chart_paths and _is_chart_request(message):
        fallback = _chart_from_text_candidates(
            [result.get("response", ""), *_recent_assistant_texts(conv)],
            message,
        )
        add(fallback)

    return chart_paths[:4]


def _assistant_message_from_result(result: dict, chart_paths: list[str], run_id: str):
    from ui.state.session import Message

    return Message(
        role="assistant",
        content=_sanitize_response(result.get("response", "")),
        agent=result.get("agent"),
        agent_icon=result.get("agent_icon"),
        agent_name=result.get("agent_name"),
        chart_paths=chart_paths,
        rerouted=result.get("rerouted", False),
        classification_time_ms=result.get("classification_time_ms", 0.0),
        total_time_ms=result.get("total_time_ms", 0.0),
        sources=result.get("sources", []),
        confidence=result.get("confidence"),
        data_freshness=result.get("data_freshness", {}),
        run_id=run_id,
        status="error" if result.get("errors") else "done",
        trace=result.get("trace", []),
        fallbacks=result.get("fallbacks", []),
        errors=result.get("errors", []),
        error=(result.get("errors") or [None])[-1],
    )


def _process_chat_turn(message: str, cid: Optional[str], save_user: bool = True) -> dict:
    """Run one chat turn and persist the assistant result.

    Used by normal send, edit, and retry so all paths share the same response
    contract and fork-from-turn semantics.
    """
    from ui.state.session import Message

    run_id = uuid.uuid4().hex[:10]
    request_id = getattr(g, "request_id", "?")
    t_start = time.time()
    logger.info("[%s] START message=%r", request_id, message[:80])
    mgr = _get_mgr()
    orch = _get_orch()

    conv = mgr.ensure_conversation(cid, activate=False)
    cid = conv.id

    if save_user:
        conv = mgr.add_message_to_conversation(
            cid,
            Message(role="user", content=message, run_id=run_id, status="done")
        )

    trace = [{
        "stage": "start",
        "label": "Initialisation",
        "status": "done",
        "agent": "orchestrator",
        "detail": "Conversation active et contexte charge.",
    }]

    history_chart = _build_structured_external_chart(message, orch) if _supports_structured_external_chart(message) else None
    if history_chart:
        chart_paths = [history_chart["chart_path"]]
        result = {
            "agent": "analytics",
            "agent_icon": "",
            "agent_name": "Analyste de Donnees",
            "response": history_chart["response"],
            "chart_paths": chart_paths,
            "sources": history_chart.get("sources", []),
            "confidence": history_chart.get("confidence"),
            "rerouted": False,
            "classification_time_ms": 0.0,
            "total_time_ms": 0.0,
            "trace": trace + [{
                "stage": "chart",
                "label": "Graphique structure",
                "status": "done",
                "agent": "analytics",
                "detail": "Source structuree externe et complement Fabric quand disponible.",
                "artifact_count": len(chart_paths),
            }],
            "fallbacks": [],
            "errors": [],
        }
    else:
        lower = message.lower().strip()
        city_chart_request = (
            "casablanca" in lower
            and any(k in lower for k in ["graph", "graphe", "graphique", "chart", "courbe"])
            and any(k in lower for k in ["compar", "autre ville", "marrakech", "tanger"])
        )
        if city_chart_request:
            text = orch.analytics_agent.try_official_kpi_answer(
                message,
                domain_context=None,
            )
            chart_paths = list(getattr(orch.analytics_agent, "last_chart_paths", []) or [])
            if text and chart_paths:
                result = {
                    "agent": "analytics",
                    "agent_icon": "",
                    "agent_name": "Analyste de Donnees",
                    "response": text,
                    "chart_paths": chart_paths,
                    "rerouted": False,
                    "classification_time_ms": 0.0,
                    "total_time_ms": 0.0,
                    "trace": [{
                        "stage": "deterministic",
                        "label": "Analyse ville + graphiques",
                        "status": "done",
                        "agent": "analytics",
                        "detail": "Fast path Fabric pour evolution Casablanca et comparaison villes.",
                        "artifact_count": len(chart_paths),
                    }],
                    "fallbacks": [],
                    "errors": [],
                }
            else:
                result = orch.route(message, conversation_id=cid, run_id=run_id)
        elif lower.startswith("/"):
            result = orch.run_node("command", message, conversation_id=cid, run_id=run_id)
            result.setdefault("classification_time_ms", 0.0)
            result.setdefault("total_time_ms", 0.0)
        else:
            result = orch.route(message, conversation_id=cid, run_id=run_id)

    result["run_id"] = result.get("run_id") or run_id
    result["request_id"] = request_id
    result["trace"] = trace + [s for s in result.get("trace", []) if s not in trace]
    elapsed = (time.time() - t_start) * 1000
    processing_ms = float(result.get("total_time_ms", 0.0) or 0.0)
    queue_ms = float(result.get("queue_time_ms", max(0.0, elapsed - processing_ms)) or 0.0)
    result["processing_time_ms"] = round(processing_ms, 1)
    result["wall_time_ms"] = round(elapsed, 1)
    result["queue_time_ms"] = round(queue_ms, 1)
    result["total_time_ms"] = round(elapsed, 1)
    logger.info(
        "[%s] DONE agent=%s classify=%.0fms processing=%.0fms wall=%.0fms",
        request_id,
        result.get("agent", "?"),
        result.get("classification_time_ms", 0),
        processing_ms,
        elapsed,
    )
    result.setdefault("fallbacks", [])
    result.setdefault("errors", [])
    result.setdefault("data_freshness", {})
    result.setdefault("sources", [])
    result.setdefault("confidence", None)

    chart_paths = _result_chart_paths(result, message, conv)
    chart_urls = _chart_urls_from_paths(chart_paths)
    assistant = _assistant_message_from_result(result, chart_paths, result["run_id"])
    mgr.add_message_to_conversation(cid, assistant)

    return _resp(
        result.get("agent"),
        result.get("agent_icon"),
        result.get("agent_name"),
        result.get("response", ""),
        result.get("rerouted", False),
        result.get("classification_time_ms", 0.0),
        result.get("total_time_ms", 0.0),
        chart_urls[0] if chart_urls else None,
        cid,
        sources=result.get("sources", []),
        confidence=result.get("confidence"),
        data_freshness=result.get("data_freshness", {}),
        metric_context=result.get("metric_context"),
        period=result.get("period", {}),
        data_scope_note=result.get("data_scope_note", ""),
        chart_urls=chart_urls,
        run_id=result["run_id"],
        trace=result.get("trace", []),
        fallbacks=result.get("fallbacks", []),
        errors=result.get("errors", []),
        processing_time_ms=result.get("processing_time_ms"),
        wall_time_ms=result.get("wall_time_ms"),
        queue_time_ms=result.get("queue_time_ms"),
        request_id=request_id,
        message_id=assistant.message_id,
    )


@app.route("/api/conversations/<cid>/messages/<mid>", methods=["PATCH"])
def edit_message(cid, mid):
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or data.get("content") or "").strip()
    if not message:
        return jsonify({"error": "message_required"}), 400
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify({
            "error": "message_too_long",
            "max_chars": MAX_MESSAGE_CHARS,
            "received_chars": len(message),
        }), 400

    try:
        mgr = _get_mgr()
        edited = mgr.edit_user_message(cid, mid, message)
        if not edited:
            return jsonify({"error": "message_not_found"}), 404
        _reload_orchestrator_state(cid)
        return jsonify(_process_chat_turn(edited.content, cid, save_user=False))
    except Exception:
        rid = getattr(g, "request_id", "?")
        logger.debug("Exception detail:", exc_info=True)
        return jsonify({"error": "edit_failed", "request_id": rid}), 500


@app.route("/api/conversations/<cid>/messages/<mid>/retry", methods=["POST"])
def retry_message(cid, mid):
    try:
        mgr = _get_mgr()
        user_msg = mgr.fork_from_message(cid, mid)
        if not user_msg:
            return jsonify({"error": "message_not_found"}), 404
        _reload_orchestrator_state(cid)
        return jsonify(_process_chat_turn(user_msg.content, cid, save_user=False))
    except Exception:
        rid = getattr(g, "request_id", "?")
        logger.debug("Exception detail:", exc_info=True)
        return jsonify({"error": "retry_failed", "request_id": rid}), 500


# ══════════════════════════════════════════════════════════════════════════════
# API: Chat
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
def chat():
    # ── Rate limit (per-IP) ──
    client_ip = _client_ip()
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
        return jsonify(_process_chat_turn(message, cid, save_user=True))
    except Exception:
        rid = getattr(g, "request_id", "?")
        logger.exception("[%s] /api/chat failed", rid)
        return jsonify({"error": "chat_failed", "request_id": rid}), 500


# Streaming was intentionally removed from the public API; the frontend uses /api/chat.
def chat_stream():
    """POST chat endpoint that streams an auditable work log via SSE."""
    client_ip = _client_ip()
    if _rate_limited(client_ip):
        return jsonify({"error": "rate_limited",
                        "retry_after_seconds": RATE_LIMIT_WINDOW_SEC}), 429

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    cid = data.get("conversation_id")

    if not message:
        return jsonify({"error": "message_required"}), 400
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify({"error": "message_too_long",
                        "max_chars": MAX_MESSAGE_CHARS,
                        "received_chars": len(message)}), 400

    request_id = getattr(g, "request_id", "?")
    q: Queue = Queue()
    done = object()

    def put(event: str, payload: dict):
        q.put((event, payload))

    def worker():
        nonlocal cid
        try:
            from ui.state.session import Message
            from utils.mvp_services import get_readiness
            from agents.executive_insight_agent import is_executive_insight_request

            put("step", {"stage": "start", "label": "Initialisation", "detail": "Preparation de la conversation et verification du contexte."})
            mgr = _get_mgr()
            orch = _get_orch()

            if cid:
                mgr.switch_conversation(cid)
            if mgr.get_active_conversation() is None:
                conv = mgr.new_conversation()
                cid = conv.id
            else:
                cid = mgr.active_conversation_id

            mgr.add_message_to_active(Message(role="user", content=message))

            history_chart = _build_structured_external_chart(message, orch) if _supports_structured_external_chart(message) else None
            if history_chart:
                put("step", {"stage": "search", "label": "Chargement source structuree", "detail": "Recherche d'une source publique structuree puis complement Fabric si disponible."})
                chart_path = history_chart["chart_path"]
                chart_url = _chart_url_from_path(chart_path)
                put("step", {"stage": "chart", "label": "Graphique interactif", "detail": "Creation d'une courbe Plotly avec sources explicites et sans interpolation des annees manquantes."})
                am = Message(
                    role="assistant",
                    content=history_chart["response"],
                    agent="analytics",
                    agent_icon="",
                    agent_name="Analyste de Données",
                    chart_path=chart_path,
                    sources=history_chart.get("sources", []),
                    confidence=history_chart.get("confidence"),
                )
                mgr.add_message_to_active(am)
                put("final", _resp(
                    "analytics", "", "Analyste de Données",
                    history_chart["response"], False, 0.0, 0.0, chart_url, cid,
                    sources=history_chart.get("sources", []),
                    confidence=history_chart.get("confidence"),
                ))
                return

            cmd_result = orch.handle_orchestrator_commands(message)
            if cmd_result:
                am = Message(role="assistant", content=cmd_result,
                             agent="normal", agent_name="Orchestrateur", agent_icon="🎯")
                mgr.add_message_to_active(am)
                put("final", _resp("normal", "🎯", "Orchestrateur",
                                   cmd_result, False, 0.0, 0.0, None, cid))
                return

            put("step", {"stage": "routing", "label": "Choix du meilleur agent", "detail": "Detection du besoin: KPI, web, prevision, graphique ou analyse executive."})
            agent_key = orch.classify(message)
            explicit_search = any(p in message.lower() for p in ("cherche", "recherche", "web", "internet", "source", "actualite", "actualité"))
            executive_mode = (
                is_executive_insight_request(message)
                and agent_key not in ("normal", "prediction")
                and not explicit_search
            )

            if executive_mode:
                freshness = get_readiness(orch).get("latest_data", {})
                result = orch.executive_agent._decline_precheck(message) or orch.executive_agent.run(
                    message,
                    domain_context=getattr(orch, "_active_domain", None),
                    data_freshness=freshness,
                    step_callback=lambda step: put("step", step),
                )
                chart_path = result.get("chart_path") or extract_chart_path(result["response"])
                if not chart_path and _is_chart_request(message):
                    chart_path = _chart_from_text_candidates(
                        [result["response"], *_recent_assistant_texts(mgr.get_active_conversation())],
                        message,
                    )
                chart_url = _chart_url_from_path(chart_path)
                am = Message(
                    role="assistant",
                    content=result["response"],
                    agent="executive_insight",
                    agent_icon=result.get("agent_icon", ""),
                    agent_name=result.get("agent_name", "Analyste Exécutif"),
                    chart_path=chart_path,
                    sources=result.get("sources", []),
                    confidence=result.get("confidence"),
                    data_freshness=result.get("data_freshness", {}),
                )
                mgr.add_message_to_active(am)
                put("final", _resp(
                    "executive_insight", result.get("agent_icon", ""),
                    result.get("agent_name", "Analyste Exécutif"), result["response"],
                    False, 0.0, 0.0, chart_url, cid,
                    sources=result.get("sources", []),
                    confidence=result.get("confidence"),
                    data_freshness=result.get("data_freshness", {}),
                ))
                return

            put("step", {"stage": "agent", "label": "Execution de l'agent", "detail": f"Agent selectionne: {agent_key}."})
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
                             agent_name="Analyste de Données", agent_icon="")
                mgr.add_message_to_active(am)
                put("final", _resp("analytics", "", "Analyste de Données",
                                   text, False, 0.0, 0.0, None, cid))
                return

            result = orch.route(message, conversation_id=cid)
            chart_path = result.get("chart_path") or extract_chart_path(result["response"])
            if not chart_path and _is_chart_request(message):
                chart_path = _chart_from_text_candidates(
                    [result["response"], *_recent_assistant_texts(mgr.get_active_conversation())],
                    message,
                )
            chart_url = _chart_url_from_path(chart_path)
            if chart_url:
                put("step", {"stage": "chart", "label": "Graphique interactif", "detail": "Generation d'un graphique Plotly consultable dans la reponse."})

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
                sources=result.get("sources", []),
                confidence=result.get("confidence"),
                data_freshness=result.get("data_freshness", {}),
            )
            mgr.add_message_to_active(am)
            put("final", _resp(
                result["agent"], result["agent_icon"], result["agent_name"],
                result["response"], result["rerouted"],
                result["classification_time_ms"], result["total_time_ms"],
                chart_url, cid,
                sources=result.get("sources", []),
                confidence=result.get("confidence"),
                data_freshness=result.get("data_freshness", {}),
                metric_context=result.get("metric_context"),
                period=result.get("period", {}),
                data_scope_note=result.get("data_scope_note", ""),
            ))
        except Exception:
            logger.debug("Exception detail:", exc_info=True)
            put("error", {"error": "chat_failed", "request_id": request_id})
        finally:
            q.put(done)

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        yield _sse("step", {"stage": "connect", "label": "Connexion etablie", "detail": "Le journal d'analyse va se mettre a jour en direct."})
        while True:
            try:
                item = q.get(timeout=0.5)
            except Empty:
                yield _sse("heartbeat", {"ts": time.time()})
                continue
            if item is done:
                break
            event, payload = item
            yield _sse(event, payload)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _resp(agent, icon, name, response, rerouted, cls_ms, tot_ms, chart_url, cid,
          sources=None, confidence=None, data_freshness=None,
          metric_context=None, period=None, data_scope_note=None,
          chart_urls=None, run_id=None, trace=None, fallbacks=None, errors=None,
          processing_time_ms=None, wall_time_ms=None, queue_time_ms=None,
          request_id=None, message_id=None):
    urls = list(chart_urls or ([] if not chart_url else [chart_url]))
    if chart_url and chart_url not in urls:
        urls.insert(0, chart_url)
    response = _sanitize_response(response)
    return {
        "agent": agent,
        "agent_icon": icon,
        "agent_name": name,
        "response": response,
        "rerouted": rerouted,
        "classification_time_ms": cls_ms,
        "total_time_ms": tot_ms,
        "processing_time_ms": processing_time_ms if processing_time_ms is not None else tot_ms,
        "wall_time_ms": wall_time_ms if wall_time_ms is not None else tot_ms,
        "queue_time_ms": queue_time_ms if queue_time_ms is not None else 0.0,
        "chart_url": urls[0] if urls else None,
        "chart_urls": urls,
        "conversation_id": cid,
        "sources": sources or [],
        "confidence": confidence,
        "data_freshness": data_freshness or {},
        "metric_context": metric_context,
        "period": period or {},
        "data_scope_note": data_scope_note or "",
        "run_id": run_id,
        "trace": trace or [],
        "fallbacks": fallbacks or [],
        "errors": errors or [],
        "request_id": request_id,
        "message_id": message_id,
    }


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chart_url_from_path(chart_path: Optional[str]) -> Optional[str]:
    if chart_path and os.path.exists(chart_path):
        return "/charts/" + os.path.basename(chart_path)
    return None


def _is_chart_request(message: str) -> bool:
    msg = (message or "").lower()
    return any(
        kw in msg
        for kw in [
            "chart", "graphe", "graphique", "visualis", "courbe",
            "diagramme", "histogram", "plot", "trace",
        ]
    )


def _chart_from_markdown_table(text: str, message: str) -> Optional[str]:
    """Best-effort Plotly fallback when an agent returns a table for a chart request."""
    if not text or "|" not in text:
        return None
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if cells and all(set(c) <= set(":- ") for c in cells):
            continue
        rows.append(cells)
    if len(rows) < 2:
        return None
    headers = [h.strip().lower().replace(" ", "_") for h in rows[0]]
    width = len(headers)
    data_rows = [r for r in rows[1:] if len(r) == width]
    if not data_rows:
        return None
    try:
        import pandas as pd
        import plotly.express as px

        df = pd.DataFrame(data_rows, columns=headers)
        for col in df.columns:
            numeric = pd.to_numeric(
                df[col].astype(str).str.replace(" ", "", regex=False).str.replace(",", "", regex=False),
                errors="coerce",
            )
            if numeric.notna().sum() >= max(1, len(df) // 2):
                df[col] = numeric

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            return None
        preferred_x = next((c for c in ["mois_fr", "mois", "annee", "année", "pays_residence", "nationalite", "region"] if c in df.columns), None)
        x_col = preferred_x or next((c for c in df.columns if c not in numeric_cols), df.columns[0])
        y_col = next((c for c in numeric_cols if c != x_col), numeric_cols[0])
        title = "Graphique généré par l'assistant d'analyse du tourisme"
        if "apf" in message.lower():
            title = "Arrivées APF"
        elif "hébergement" in message.lower() or "hebergement" in message.lower():
            title = "Indicateur hébergement"

        if any(k in message.lower() for k in ["mois", "tendance", "évolution", "evolution", "courbe"]):
            fig = px.line(df, x=x_col, y=y_col, markers=True, title=title)
        else:
            fig = px.bar(df, x=x_col, y=y_col, title=title)
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=40, r=24, t=60, b=44),
            font=dict(family="Inter, Arial", size=13),
            hovermode="x unified",
        )
        fig.update_traces(hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>")
        os.makedirs(_CHARTS_DIR, exist_ok=True)
        filename = f"chat_chart_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.html"
        path = os.path.join(_CHARTS_DIR, filename)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True, config={"responsive": True, "displaylogo": False})
        return path
    except Exception:
        logger.debug("Exception detail:", exc_info=True)
        return None


def _parse_number_token(raw: str) -> Optional[float]:
    token = (raw or "").strip()
    if not token:
        return None
    token = token.replace("\u202f", " ").replace("\xa0", " ")
    token = token.replace("%", "").replace("MAD", "").replace("USD", "")
    token = token.replace(" ", "")
    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "," in token:
        token = token.replace(",", ".")
    try:
        return float(token)
    except Exception:
        return None


def _chart_from_ranked_text(text: str, message: str) -> Optional[str]:
    """Build a chart from ranked/narrative text like '1. France: 32,1 %'."""
    if not text:
        return None

    patterns = [
        re.compile(
            r"^\s*(?:[-*]\s+|\d+[\).\-\s]+)(?P<label>[A-Za-zÀ-ÿ0-9 /&'()\-]+?)\s*(?:[:\-–—]\s*|\s+\()\s*(?P<value>-?\d[\d\s.,]*)\s*(?P<unit>%?)\)?\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?P<label>[A-Za-zÀ-ÿ0-9 /&'()\-]+?)\s*[:\-–—]\s*(?P<value>-?\d[\d\s.,]*)\s*(?P<unit>%?)\s*$",
            re.IGNORECASE,
        ),
    ]

    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if len(line) < 4:
            continue
        for pattern in patterns:
            match = pattern.match(line)
            if not match:
                continue
            label = re.sub(r"\s+", " ", match.group("label")).strip(" -:;,.")
            if not label or len(label) > 80:
                continue
            value = _parse_number_token(match.group("value"))
            if value is None:
                continue
            unit = "%" if "%" in (match.group("unit") or "") or "part" in message.lower() else ""
            rows.append({"label": label, "value": value, "unit": unit})
            break

    if len(rows) < 2:
        inline = re.findall(
            r"([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9 /&'()\-]{1,50}?)\s*(?:[:\-–—]|\()\s*(\d[\d\s.,]*)\s*(%?)\)?",
            text,
            flags=re.IGNORECASE,
        )
        for label, value_raw, unit_raw in inline:
            value = _parse_number_token(value_raw)
            if value is None:
                continue
            clean_label = re.sub(r"\s+", " ", label).strip(" -:;,.")
            if clean_label and all(r["label"].lower() != clean_label.lower() for r in rows):
                rows.append({"label": clean_label, "value": value, "unit": "%" if unit_raw else ""})

    deduped = []
    seen = set()
    for row in rows:
        key = row["label"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    rows = deduped[:20]
    if len(rows) < 2:
        return None

    try:
        import pandas as pd
        import plotly.express as px

        df = pd.DataFrame(rows)
        share_mode = any(r["unit"] == "%" for r in rows) or any(
            term in (message or "").lower() for term in ["part", "parts", "share", "repartition", "répartition"]
        )
        df = df.sort_values("value", ascending=True)
        title = "Graphique genere par l'assistant d'analyse du tourisme"
        if share_mode:
            title = "Classement et parts"

        fig = px.bar(
            df,
            x="value",
            y="label",
            orientation="h",
            text="value",
            title=title,
            labels={"value": "Part (%)" if share_mode else "Valeur", "label": "Categorie"},
        )
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=60, r=24, t=60, b=40),
            font=dict(family="Inter, Arial", size=13),
        )
        if share_mode:
            fig.update_traces(texttemplate="%{x:.1f}%", hovertemplate="%{y}<br>%{x:.1f}%<extra></extra>")
        else:
            fig.update_traces(texttemplate="%{x:,.0f}", hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>")

        os.makedirs(_CHARTS_DIR, exist_ok=True)
        filename = f"chat_ranked_chart_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.html"
        path = os.path.join(_CHARTS_DIR, filename)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True, config={"responsive": True, "displaylogo": False})
        return path
    except Exception:
        logger.debug("Exception detail:", exc_info=True)
        return None


def _chart_from_text_candidates(candidates: list[str], message: str) -> Optional[str]:
    for candidate in candidates:
        chart_path = _chart_from_markdown_table(candidate, message)
        if chart_path:
            return chart_path
    for candidate in candidates:
        chart_path = _chart_from_ranked_text(candidate, message)
        if chart_path:
            return chart_path
    return None


def _recent_assistant_texts(conv, limit: int = 3) -> list[str]:
    if not conv:
        return []
    texts = []
    for msg in reversed(conv.messages):
        if getattr(msg, "role", None) == "assistant" and getattr(msg, "content", None):
            texts.append(msg.content)
            if len(texts) >= limit:
                break
    return texts


COUNTRY_ALIASES = {
    "maroc": ("MAR", "Maroc"),
    "morocco": ("MAR", "Maroc"),
    "espagne": ("ESP", "Espagne"),
    "spain": ("ESP", "Espagne"),
    "france": ("FRA", "France"),
    "turquie": ("TUR", "Turquie"),
    "turkey": ("TUR", "Turquie"),
    "egypte": ("EGY", "Egypte"),
    "egypt": ("EGY", "Egypte"),
    "tunisie": ("TUN", "Tunisie"),
    "tunisia": ("TUN", "Tunisie"),
    "portugal": ("PRT", "Portugal"),
}

WORLD_BANK_TOURISM_INDICATORS = {
    "tourism_arrivals": {
        "id": "ST.INT.ARVL",
        "label": "Arrivees touristiques internationales",
        "scale": 1_000_000,
        "scaled_label": "Arrivees touristiques (millions)",
        "terms": ["touriste", "touristes", "arrivee", "arrivees", "arrivée", "arrivées", "flux"],
    },
    "tourism_receipts": {
        "id": "ST.INT.RCPT.CD",
        "label": "Recettes touristiques internationales",
        "scale": 1_000_000_000,
        "scaled_label": "Recettes touristiques (Md USD)",
        "terms": ["recette", "recettes", "devises", "revenus", "revenue", "receipts"],
    },
}


def _parse_year_range(message: str) -> Optional[tuple[int, int]]:
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", message or "")]
    if len(years) >= 2:
        start, end = min(years), max(years)
    elif len(years) == 1:
        start, end = years[0], years[0]
    else:
        return None
    current_year = datetime.now().year
    start = max(1960, min(start, current_year))
    end = max(start, min(end, current_year))
    return start, end


def _country_from_message(message: str) -> Optional[tuple[str, str]]:
    msg = (message or "").lower()
    for alias, country in COUNTRY_ALIASES.items():
        if alias in msg:
            return country
    return None


def _tourism_indicator_from_message(message: str) -> Optional[dict]:
    msg = (message or "").lower()
    for indicator in WORLD_BANK_TOURISM_INDICATORS.values():
        if any(term in msg for term in indicator["terms"]):
            return indicator
    return None


def _supports_structured_external_chart(message: str) -> bool:
    return bool(
        _is_chart_request(message)
        and _parse_year_range(message)
        and _country_from_message(message)
        and _tourism_indicator_from_message(message)
    )


def _build_structured_external_chart(message: str, orch=None) -> Optional[dict]:
    """Create a sourced chart from structured external indicators when possible.

    This is a generic fallback for chart requests that need public historical
    data outside Fabric coverage. If no structured source covers the request,
    the normal research agent keeps control and must state data limitations.
    """
    try:
        import pandas as pd
        import plotly.express as px
        import requests

        year_range = _parse_year_range(message)
        country = _country_from_message(message)
        indicator = _tourism_indicator_from_message(message)
        if not year_range or not country or not indicator:
            return None

        start_year, end_year = year_range
        iso3, country_label = country
        indicator_id = indicator["id"]
        rows_by_year = {}
        wb_url = (
            f"https://api.worldbank.org/v2/country/{iso3}/indicator/{indicator_id}"
            f"?format=json&per_page=200&date={start_year}:{end_year}"
        )
        wb_resp = requests.get(wb_url, timeout=20)
        wb_resp.raise_for_status()
        wb_payload = wb_resp.json()
        if len(wb_payload) > 1:
            for item in wb_payload[1]:
                year = int(item["date"])
                value = item.get("value")
                if start_year <= year <= end_year and value is not None:
                    rows_by_year[year] = {
                        "annee": year,
                        "valeur": float(value),
                        "valeur_affichee": round(float(value) / indicator["scale"], 2),
                        "source": f"World Bank WDI {indicator_id}",
                    }

        # For Morocco arrivals, Fabric APF is the system of record for recent
        # years when public sources are not yet complete.
        if iso3 == "MAR" and indicator_id == "ST.INT.ARVL":
            kpi_cache = getattr(getattr(orch, "analytics_agent", None), "kpi_cache", None)
            if kpi_cache:
                for year in range(start_year, end_year + 1):
                    total = kpi_cache.total_for_year(year)
                    if total:
                        rows_by_year[year] = {
                            "annee": year,
                            "valeur": float(total),
                            "valeur_affichee": round(float(total) / indicator["scale"], 2),
                            "source": "Fabric APF STATOUR",
                        }

        rows = []
        missing_years = []
        for year in range(start_year, end_year + 1):
            row = rows_by_year.get(year)
            if row:
                rows.append(row)
            else:
                missing_years.append(year)
                rows.append({
                    "annee": year,
                    "valeur": None,
                    "valeur_affichee": None,
                    "source": "Non disponible dans les sources connectees",
                })

        df = pd.DataFrame(rows)
        if df["valeur_affichee"].notna().sum() < 2:
            return None

        fig = px.line(
            df,
            x="annee",
            y="valeur_affichee",
            color="source",
            markers=True,
            title=f"{indicator['label']} - {country_label} ({start_year}-{end_year})",
            labels={
                "annee": "Annee",
                "valeur_affichee": indicator["scaled_label"],
                "source": "Source",
            },
        )
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=44, r=24, t=70, b=54),
            font=dict(family="Inter, Arial", size=13),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left", x=0),
        )
        fig.update_xaxes(dtick=max(1, min(5, (end_year - start_year) // 8 or 1)))
        fig.update_traces(connectgaps=False, hovertemplate="%{x}<br>%{y:.2f}<extra></extra>")

        os.makedirs(_CHARTS_DIR, exist_ok=True)
        filename = (
            f"structured_{iso3.lower()}_{indicator_id.lower().replace('.', '_')}_"
            f"{start_year}_{end_year}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.html"
        )
        chart_path = os.path.join(_CHARTS_DIR, filename)
        fig.write_html(
            chart_path,
            include_plotlyjs="cdn",
            full_html=True,
            config={"responsive": True, "displaylogo": False},
        )

        table_rows = [
            f"| {int(r.annee)} | {r.valeur_affichee:.2f} | {r.source} |"
            for r in df.itertuples()
            if pd.notna(r.valeur_affichee)
        ]
        missing_text = (
            ", ".join(str(y) for y in missing_years)
            if missing_years
            else "aucune annee manquante dans les sources structurees utilisees"
        )
        response = (
            "## Graphique cree\n"
            f"J'ai genere un graphique interactif pour {country_label}, sur {start_year}-{end_year}, avec les donnees structurees disponibles.\n\n"
            "## Methode\n"
            f"- Source externe structuree : World Bank WDI `{indicator_id}`.\n"
            "- Complement Fabric/APF uniquement lorsque disponible et pertinent pour les donnees recentes du Maroc.\n"
            "- Les annees absentes ne sont pas interpolees afin de ne pas inventer de chiffres.\n\n"
            "## Donnees utilisees\n"
            f"| Annee | {indicator['scaled_label']} | Source |\n"
            "| :--- | :--- | :--- |\n"
            + "\n".join(table_rows)
            + f"\n\nAnnees sans valeur dans les sources connectees : {missing_text}.\n\n"
            "Confiance : moyenne a elevee, selon la couverture temporelle effective."
        )
        return {
            "response": response,
            "chart_path": chart_path,
            "sources": [
                {
                    "source": f"World Bank Data - {indicator_id}",
                    "title": f"{indicator['label']} - {country_label}",
                    "url": f"https://data.worldbank.org/indicator/{indicator_id}?locations={iso3}",
                },
                {
                    "source": "STATOUR Fabric APF",
                    "title": "Arrivees aux postes frontieres",
                    "url": "",
                },
            ],
            "confidence": "moyenne à élevée",
        }
    except Exception:
        logger.debug("Exception detail:", exc_info=True)
        return None


def _asset_data_uri(filename: str) -> str:
    path = os.path.join(PROJECT_ROOT, "ui", "assets", filename)
    if not os.path.exists(path):
        return ""
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _markdownish_to_html(text: str) -> str:
    lines = (text or "").splitlines()
    out = []
    in_list = False
    for raw in lines:
        line = raw.strip()
        if not line:
            if in_list:
                out.append("</ul>")
                in_list = False
            continue
        if line.startswith("## "):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{html.escape(line[2:])}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<p>{html.escape(line)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _public_display_text(value: str) -> str:
    return re.sub(r"\bSTATOUR\b", "Assistant d'analyse du tourisme", str(value or ""), flags=re.IGNORECASE)


def _build_report_html(payload: dict) -> str:
    ministry_logo = _asset_data_uri("logo.png") or _asset_data_uri("logo2.png")
    title = html.escape(_public_display_text(payload.get("title") or "Rapport d'analyse du tourisme"))
    question = html.escape(_public_display_text(payload.get("question") or ""))
    confidence = html.escape(payload.get("confidence") or "non précisée")
    generated = datetime.now().strftime("%d/%m/%Y %H:%M")
    body = _markdownish_to_html(_public_display_text(payload.get("response") or ""))
    chart_urls = payload.get("chart_urls") or []
    chart_url = payload.get("chart_url") or (chart_urls[0] if chart_urls else "")
    chart_urls = chart_urls or ([chart_url] if chart_url else [])
    chart_block = ""
    if chart_urls:
        chart_sections = []
        for idx, chart_url in enumerate(chart_urls[:4], start=1):
            if not chart_url:
                continue
            chart_srcdoc = ""
            chart_file = os.path.realpath(os.path.join(_CHARTS_DIR, os.path.basename(chart_url)))
            if chart_file.startswith(os.path.realpath(_CHARTS_DIR)) and os.path.exists(chart_file):
                try:
                    with open(chart_file, "r", encoding="utf-8") as f:
                        chart_srcdoc = html.escape(f.read(), quote=True)
                except Exception:
                    chart_srcdoc = ""
            iframe = (
                f"<iframe srcdoc=\"{chart_srcdoc}\"></iframe>"
                if chart_srcdoc
                else f"<iframe src='{html.escape(chart_url)}'></iframe>"
            )
            chart_sections.append(
                f"<h2>Graphique interactif {idx}</h2>"
                f"<p>Graphique associe : <a href='{html.escape(chart_url)}'>{html.escape(chart_url)}</a></p>"
                f"{iframe}"
            )
        chart_block = "<section class='card'>" + "\n".join(chart_sections) + "</section>"
    sources = payload.get("sources") or []
    source_items = "\n".join(
        f"<li><strong>{html.escape(_public_display_text(s.get('source') or s.get('title') or 'Source'))}</strong>"
        + (f" · <a href='{html.escape(s.get('url'))}'>{html.escape(s.get('url'))}</a>" if s.get("url") else "")
        + "</li>"
        for s in sources[:12]
    ) or "<li>Aucune source externe transmise.</li>"
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    @page {{ margin: 20mm; }}
    body {{ margin: 0; background: #f4f6f8; color: #142033; font-family: Inter, Arial, sans-serif; }}
    .page {{ max-width: 980px; margin: 32px auto; background: #fff; border: 1px solid #d9e0ea; box-shadow: 0 18px 48px rgba(15,23,42,.10); }}
    header {{ padding: 28px 34px 22px; border-bottom: 4px solid #0b4f71; display: flex; align-items: center; justify-content: space-between; gap: 24px; }}
    .logos {{ display: flex; align-items: center; gap: 20px; }}
    .logos img.ministry {{ height: 62px; max-width: 460px; object-fit: contain; }}
    .meta {{ text-align: right; color: #64748b; font-size: 12px; line-height: 1.5; }}
    main {{ padding: 30px 34px 38px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; color: #0f2f46; }}
    .question {{ margin: 18px 0 22px; padding: 16px 18px; background: #eef6f8; border-left: 4px solid #0f766e; font-weight: 650; }}
    .card {{ border: 1px solid #dfe6ee; border-radius: 8px; padding: 20px 22px; margin: 18px 0; }}
    h2 {{ margin: 18px 0 8px; font-size: 18px; color: #0b4f71; }}
    p, li {{ font-size: 14px; line-height: 1.65; }}
    ul {{ padding-left: 22px; }}
    .badge {{ display: inline-block; padding: 5px 10px; border-radius: 999px; background: #ecfdf5; color: #047857; font-size: 12px; font-weight: 800; }}
    iframe {{ width: 100%; height: 430px; border: 1px solid #dfe6ee; border-radius: 8px; }}
    a {{ color: #0b4f71; }}
    footer {{ padding: 18px 34px; border-top: 1px solid #dfe6ee; color: #64748b; font-size: 12px; }}
    @media print {{ body {{ background:#fff; }} .page {{ margin:0; box-shadow:none; border:0; }} }}
  </style>
</head>
<body>
  <div class="page">
    <header>
      <div class="logos">
        {f'<img class="ministry" src="{ministry_logo}" alt="Ministere du Tourisme">' if ministry_logo else ''}
      </div>
      <div class="meta">Assistant d'analyse du tourisme<br>{generated}<br><span class="badge">Confiance : {confidence}</span></div>
    </header>
    <main>
      <h1>{title}</h1>
      <div class="question">Question : {question}</div>
      <section class="card">{body}</section>
      {chart_block}
      <section class="card"><h2>Sources utilisées</h2><ul>{source_items}</ul></section>
    </main>
    <footer>Rapport généré automatiquement. Les résultats analytiques doivent être validés selon les processus officiels du Ministère.</footer>
  </div>
</body>
</html>"""


REPORTS_DIR = os.environ.get("STATOUR_REPORTS_DIR", os.path.join(PROJECT_ROOT, "reports"))
os.makedirs(REPORTS_DIR, exist_ok=True)


@app.route("/api/reports", methods=["POST"])
def create_report():
    data = request.get_json(silent=True) or {}
    response = (data.get("response") or "").strip()
    if not response:
        return jsonify({"error": "response_required"}), 400
    report_id = f"tourisme_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.html"
    path = os.path.join(REPORTS_DIR, report_id)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_build_report_html(data))
    return jsonify({"report_id": report_id, "download_url": f"/reports/{report_id}"})


@app.route("/reports/<path:filename>")
def download_report(filename):
    safe = os.path.basename(filename)
    path = os.path.realpath(os.path.join(REPORTS_DIR, safe))
    if not path.startswith(os.path.realpath(REPORTS_DIR)) or not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True, download_name=safe, mimetype="text/html")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    logger.info("=" * 58)
    logger.info("  Assistant Tourisme Maroc -- Ministere du Tourisme")
    logger.info("  Ministere du Tourisme, Artisanat & Economie Sociale")
    logger.info("=" * 58)

    logger.info("Initialisation du systeme en arriere-plan...")
    _start_background_warmup()

    logger.info("Serveur disponible sur: http://localhost:5000")
    logger.info("Appuyez sur Ctrl+C pour arreter.")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
