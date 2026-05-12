"""
STATOUR Chat Area Component
"""

import streamlit as st
from typing import Optional

from ui.state.session import Message
from ui.components.chart_viewer import (
    render_chart,
    render_charts_from_message,
    extract_chart_paths
)


def _get_agent_badge(agent: Optional[str], agent_name: Optional[str]) -> str:
    """Generate HTML for agent badge."""
    if not agent:
        return ""

    colors = {
        "normal": ("#3498DB", "rgba(52,152,219,0.15)"),
        "researcher": ("#2ECC71", "rgba(46,204,113,0.15)"),
        "analytics": ("#F39C12", "rgba(243,156,18,0.15)"),
    }
    color, bg = colors.get(agent, ("#A0A4B0", "rgba(160,164,176,0.15)"))
    icon = ""
    name = agent_name or agent.title()

    return (
        f'<span style="display:inline-flex; align-items:center; gap:4px; '
        f'padding:2px 8px; border-radius:8px; font-size:0.72rem; '
        f'font-weight:600; background:{bg}; color:{color};">'
        f'{icon} {name}</span>'
    )


def _format_timing(msg: Message) -> str:
    """Format timing info for a message."""
    parts = []
    if msg.classification_time_ms > 0:
        parts.append(f"routing: {msg.classification_time_ms:.0f}ms")
    if msg.total_time_ms > 0:
        parts.append(f"total: {msg.total_time_ms:.0f}ms")
    if msg.rerouted:
        parts.append("🔄 rerouted")
    return " · ".join(parts) if parts else ""


def _clean_content_for_display(content: str) -> str:
    """Remove chart path lines from content for cleaner display."""
    lines = content.split("\n")
    clean_lines = []
    for line in lines:
        if "Chart:" in line and ".html" in line:
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines).strip()


def render_welcome():
    """Render the welcome screen when no messages exist."""
    st.markdown(
        """
        <div style="text-align:center; padding: 60px 20px; max-width:650px; margin:0 auto;">
            <div style="font-size: 3rem; margin-bottom: 16px;">🇲🇦</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #FAFAFA; margin-bottom: 8px;">
                Bienvenue sur STATOUR
            </div>
            <div style="font-size: 1rem; color: #A0A4B0; margin-bottom: 40px; line-height: 1.6;">
                Plateforme d'Intelligence Touristique du Maroc<br>
                Posez vos questions en français, anglais ou arabe
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Suggestion cards
    suggestions = [
        {
            "icon": "📊",
            "title": "Analyser les arrivées",
            "desc": "Top 10 pays de résidence en 2024",
            "query": "Quelles sont les top 10 pays de résidence des touristes en 2024?",
        },
        {
            "icon": "🔍",
            "title": "Rechercher des actualités",
            "desc": "Dernières nouvelles du tourisme",
            "query": "Quelles sont les dernières actualités du tourisme au Maroc?",
        },
        {
            "icon": "📈",
            "title": "Évolution mensuelle",
            "desc": "Tendances des arrivées par mois",
            "query": "Montre l'évolution mensuelle des arrivées touristiques",
        },
        {
            "icon": "🌍",
            "title": "Comparaison régionale",
            "desc": "Performance par région",
            "query": "Compare les arrivées touristiques par région du Maroc",
        },
    ]

    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(
                f"{suggestion['icon']}  {suggestion['title']}\n{suggestion['desc']}",
                key=f"suggestion_{i}",
                use_container_width=True,
            ):
                st.session_state["pending_message"] = suggestion["query"]
                st.rerun()


def render_message(msg: Message):
    """Render a single chat message."""
    if msg.role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg.content)

    elif msg.role == "assistant":
        avatar = "🤖"
        if msg.agent:
            avatars = {"normal": "🏛️", "researcher": "🔍", "analytics": "📊"}
            avatar = avatars.get(msg.agent, "🤖")

        with st.chat_message("assistant", avatar=avatar):
            # Agent badge
            badge = _get_agent_badge(msg.agent, msg.agent_name)
            if badge:
                st.markdown(badge, unsafe_allow_html=True)

            # Clean content (remove chart paths)
            clean_content = _clean_content_for_display(msg.content)
            if clean_content:
                st.markdown(clean_content)

            # Render charts if present (supports multiple charts)
            charts_rendered = 0
            
            # First try to render from explicit chart_path
            if msg.chart_path:
                if render_chart(msg.chart_path):
                    charts_rendered += 1
            
            # Then extract and render all charts from content (handles multiple charts)
            # This will find additional charts not in msg.chart_path
            charts_rendered += render_charts_from_message(msg.content, max_charts=4)

            # Timing info
            timing = _format_timing(msg)
            if timing:
                st.markdown(
                    f'<div style="font-size:0.7rem; color:#A0A4B0; '
                    f'margin-top:4px; text-align:right;">{timing}</div>',
                    unsafe_allow_html=True,
                )


def render_chat_area():
    """Render the complete chat area."""
    session_mgr = st.session_state.get("session_manager")
    if not session_mgr:
        render_welcome()
        return

    conv = session_mgr.get_active_conversation()

    if conv is None or len(conv.messages) == 0:
        render_welcome()
        return

    # Render all messages
    for msg in conv.messages:
        render_message(msg)