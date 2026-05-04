"""
STATOUR Chat Application — Main Entry Point
=============================================
Run:
    streamlit run ui/app.py
"""

import os
import sys

# ── Path setup ──
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

# ── Page config (MUST be first Streamlit call) ──
st.set_page_config(
    page_title="STATOUR — Intelligence Touristique",
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Now import everything else ──
from config.settings import validate_config
# FIX: Use correct module name (match your actual filename)
try:
    from agents.orchestrator_agent import Orchestrator
except ImportError:
    from agents.orchestrator import Orchestrator

from ui.state.session import SessionManager, Message
from ui.components.sidebar import render_sidebar
from ui.components.chat_area import render_chat_area, render_welcome
from ui.components.header import render_header
from ui.components.chart_viewer import extract_chart_path
from utils.logger import get_logger

logger = get_logger("statour.ui")


# ══════════════════════════════════════════════════════════════════════════════
# Load Custom CSS
# ══════════════════════════════════════════════════════════════════════════════

def load_css():
    """Load custom CSS file."""
    css_path = os.path.join(APP_DIR, "styles", "main.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                .stDeployButton {display: none;}
            </style>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Initialization
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_orchestrator():
    """Initialize the orchestrator (cached across reruns)."""
    try:
        validate_config(require_tavily=True)
        return Orchestrator()
    except Exception as e:
        logger.error("Failed to initialize orchestrator: %s", e)
        st.error(f"Orchestrator init failed: {e}")
        return None


def init_session_state():
    """Initialize all session state variables."""
    if "session_manager" not in st.session_state:
        st.session_state.session_manager = SessionManager()

    if "orchestrator" not in st.session_state:
        orch = init_orchestrator()
        st.session_state.orchestrator = orch

    if "processing" not in st.session_state:
        st.session_state.processing = False

    if "pending_message" not in st.session_state:
        st.session_state.pending_message = None


# ══════════════════════════════════════════════════════════════════════════════
# Message Processing
# ══════════════════════════════════════════════════════════════════════════════

def process_message(user_input: str):
    """Process a user message through the orchestrator."""
    session_mgr = st.session_state.session_manager
    orchestrator = st.session_state.get("orchestrator")

    if not orchestrator:
        st.error("❌ L'orchestrateur n'est pas initialisé. Vérifiez votre configuration.")
        return

    # Ensure we have an active conversation
    if session_mgr.get_active_conversation() is None:
        session_mgr.new_conversation()

    # Add user message
    user_msg = Message(role="user", content=user_input)
    session_mgr.add_message_to_active(user_msg)

    # Process through orchestrator
    try:
        result = orchestrator.route(user_input)

        # Extract chart path from response
        chart_path = extract_chart_path(result["response"])

        # Create assistant message
        assistant_msg = Message(
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
        session_mgr.add_message_to_active(assistant_msg)

    except Exception as e:
        logger.error("Message processing failed: %s", e, exc_info=True)

        error_msg = Message(
            role="assistant",
            content=f"❌ Erreur de traitement: {str(e)}",
            agent="normal",
            agent_name="Système",
        )
        session_mgr.add_message_to_active(error_msg)


# ══════════════════════════════════════════════════════════════════════════════
# Handle Commands
# ══════════════════════════════════════════════════════════════════════════════

def handle_command(user_input: str) -> bool:
    """Check if input is a slash command and handle it."""
    orchestrator = st.session_state.get("orchestrator")
    if not orchestrator:
        return False

    # Orchestrator-level commands
    cmd_result = orchestrator.handle_orchestrator_commands(user_input)
    if cmd_result:
        session_mgr = st.session_state.session_manager
        if session_mgr.get_active_conversation() is None:
            session_mgr.new_conversation()

        user_msg = Message(role="user", content=user_input)
        session_mgr.add_message_to_active(user_msg)

        assistant_msg = Message(
            role="assistant",
            content=cmd_result,
            agent="normal",
            agent_name="Orchestrateur",
        )
        session_mgr.add_message_to_active(assistant_msg)
        return True

    # Analytics quick commands
    cmd = user_input.lower().strip()

    # Safe access to analytics agent
    if not hasattr(orchestrator, 'analytics_agent'):
        return False

    analytics = orchestrator.analytics_agent

    command_map = {
        "/datasets": lambda: analytics.list_datasets(),
        "/stats": lambda: analytics.quick_stats(),
        "/schema": lambda: analytics.get_schema(),
        "/columns": lambda: analytics.get_columns(),
        "/sample": lambda: analytics.get_sample(),
    }

    if cmd in command_map:
        session_mgr = st.session_state.session_manager
        if session_mgr.get_active_conversation() is None:
            session_mgr.new_conversation()

        user_msg = Message(role="user", content=user_input)
        session_mgr.add_message_to_active(user_msg)

        result_text = command_map[cmd]()
        assistant_msg = Message(
            role="assistant",
            content=result_text,
            agent="analytics",
            agent_name="Analyste de Données",
        )
        session_mgr.add_message_to_active(assistant_msg)
        return True

    # /switch command
    if cmd.startswith("/switch "):
        name = user_input.split(maxsplit=1)[1].strip()
        result_text = analytics.switch_dataset(name)

        session_mgr = st.session_state.session_manager
        if session_mgr.get_active_conversation() is None:
            session_mgr.new_conversation()

        user_msg = Message(role="user", content=user_input)
        session_mgr.add_message_to_active(user_msg)

        assistant_msg = Message(
            role="assistant",
            content=result_text,
            agent="analytics",
            agent_name="Analyste de Données",
        )
        session_mgr.add_message_to_active(assistant_msg)
        return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# Main Application
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point."""

    # ── Initialize ──
    load_css()
    init_session_state()

    # ── Check orchestrator ──
    if st.session_state.orchestrator is None:
        st.error(
            "❌ **Erreur d'initialisation**\n\n"
            "L'orchestrateur STATOUR n'a pas pu démarrer.\n"
            "Vérifiez votre fichier `.env` :\n"
            "- `AZURE_OPENAI_API_KEY`\n"
            "- `AZURE_OPENAI_ENDPOINT`\n"
            "- `AZURE_OPENAI_DEPLOYMENT`\n"
            "- `TAVILY_API_KEY`"
        )
        return

    # ── Render sidebar ──
    render_sidebar()

    # ── Render header ──
    render_header()

    # ── Render chat area ──
    render_chat_area()

    # ── Handle pending messages (from suggestion cards) ──
    if st.session_state.get("pending_message"):
        pending = st.session_state.pending_message
        st.session_state.pending_message = None
        with st.spinner("⏳ Traitement en cours..."):
            process_message(pending)
        st.rerun()

    # ── Chat input ──
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

    user_input = st.chat_input(
        placeholder="Posez votre question sur le tourisme marocain...",
        key="chat_input",
    )

    if user_input:
        user_input = user_input.strip()
        if not user_input:
            return

        # Check for commands first
        if user_input.startswith("/"):
            if handle_command(user_input):
                st.rerun()
                return

        # Process regular message
        with st.spinner("⏳ Traitement en cours..."):
            process_message(user_input)

        st.rerun()


if __name__ == "__main__":
    main()