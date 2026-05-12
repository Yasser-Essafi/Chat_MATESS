"""
STATOUR Sidebar Component
"""

import streamlit as st
from datetime import datetime
from typing import Optional


def _format_date(iso_str: str) -> str:
    """Format ISO date string for display."""
    try:
        dt = datetime.fromisoformat(iso_str)
        now = datetime.now()
        diff = now - dt

        if diff.days == 0:
            return dt.strftime("%H:%M")
        elif diff.days == 1:
            return "Hier"
        elif diff.days < 7:
            days_fr = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
            return days_fr[dt.weekday()]
        else:
            return dt.strftime("%d/%m")
    except Exception:
        return ""


def _get_agent_emoji(agent: Optional[str]) -> str:
    """Get label for agent type."""
    return {
        "normal": "G",
        "researcher": "R",
        "analytics": "A",
    }.get(agent, "?")


def _safe_reset_agents():
    """Safely reset all agent conversations."""
    orch = st.session_state.get("orchestrator")
    if not orch:
        return

    for attr in ["normal_agent", "researcher_agent", "analytics_agent"]:
        agent = getattr(orch, attr, None)
        if agent and hasattr(agent, "reset_conversation"):
            try:
                agent.reset_conversation()
            except Exception:
                pass

    orch.last_agent = None
    if hasattr(orch, "conversation_log"):
        orch.conversation_log.clear()


def render_sidebar():
    """Render the sidebar with conversation history."""
    session_mgr = st.session_state.get("session_manager")
    if not session_mgr:
        return

    with st.sidebar:
        # ── Logo & Title ──
        st.markdown(
            """
            <div style="text-align: center; padding: 16px 0 8px 0;">
                <div style="font-size: 1.6rem; font-weight: 700; color: #FAFAFA;">
                    🇲🇦 STATOUR
                </div>
                <div style="font-size: 0.7rem; color: #A0A4B0; margin-top: 2px;">
                    Ministère du Tourisme du Maroc
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<hr style='border-color:#2D3139; margin: 8px 0 16px 0;'>",
            unsafe_allow_html=True,
        )

        # ── New Chat Button ──
        if st.button(
            "➕  Nouvelle conversation",
            key="new_chat",
            use_container_width=True,
            type="primary",
        ):
            session_mgr.new_conversation()
            _safe_reset_agents()
            st.rerun()

        st.markdown("")

        # ── Search ──
        search_query = st.text_input(
            "🔎 Rechercher",
            key="sidebar_search",
            placeholder="Rechercher dans l'historique...",
            label_visibility="collapsed",
        )

        # ── Conversation List ──
        conversations = session_mgr.get_conversation_list()

        if search_query:
            conversations = [
                c for c in conversations
                if search_query.lower() in c["title"].lower()
            ]

        if not conversations:
            st.markdown(
                """
                <div style="text-align:center; padding: 40px 0; color: #A0A4B0;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">💬</div>
                    <div style="font-size: 0.85rem;">Aucune conversation</div>
                    <div style="font-size: 0.75rem;">Commencez par poser une question!</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        # Group by date
        today_convs = []
        yesterday_convs = []
        older_convs = []
        now = datetime.now()

        for conv in conversations:
            try:
                dt = datetime.fromisoformat(conv["updated_at"])
                diff = (now - dt).days
                if diff == 0:
                    today_convs.append(conv)
                elif diff == 1:
                    yesterday_convs.append(conv)
                else:
                    older_convs.append(conv)
            except Exception:
                older_convs.append(conv)

        def render_group(label: str, convs: list):
            if not convs:
                return

            st.markdown(
                f"<div style='font-size:0.7rem; color:#A0A4B0; padding: 8px 0 4px 4px; "
                f"font-weight:600; text-transform:uppercase; letter-spacing:0.5px;'>"
                f"{label}</div>",
                unsafe_allow_html=True,
            )

            for conv in convs:
                conv_id = conv["id"]
                is_active = conv_id == session_mgr.active_conversation_id
                agent_emoji = _get_agent_emoji(conv.get("last_agent"))

                col1, col2 = st.columns([5, 1])

                with col1:
                    btn_type = "primary" if is_active else "secondary"
                    # Truncate title safely
                    title = conv["title"]
                    if len(title) > 45:
                        title = title[:42] + "..."
                    label_text = f"{agent_emoji} {title}"

                    if st.button(
                        label_text,
                        key=f"conv_{conv_id}",
                        use_container_width=True,
                        type=btn_type,
                    ):
                        session_mgr.switch_conversation(conv_id)
                        st.rerun()

                with col2:
                    if st.button("🗑️", key=f"del_{conv_id}", help="Supprimer"):
                        session_mgr.delete_conversation(conv_id)
                        st.rerun()

        render_group("Aujourd'hui", today_convs)
        render_group("Hier", yesterday_convs)
        render_group("Plus ancien", older_convs)

        # ── Footer ──
        st.markdown(
            "<hr style='border-color:#2D3139; margin: 16px 0;'>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Tout effacer", key="clear_all", use_container_width=True):
                session_mgr.clear_all_history()
                st.rerun()
        with col2:
            count = len(conversations)
            st.markdown(
                f"<div style='text-align:center; font-size:0.7rem; color:#A0A4B0; "
                f"padding-top:8px;'>{count} conversation{'s' if count > 1 else ''}</div>",
                unsafe_allow_html=True,
            )