"""
STATOUR Header Component
"""

import streamlit as st


def render_header():
    """Render the top header bar."""
    col1, col2, col3 = st.columns([1, 4, 1])

    with col1:
        st.markdown("### 🇲🇦")

    with col2:
        st.markdown(
            """
            <div style="text-align: center;">
                <span style="font-size: 1.2rem; font-weight: 700; color: #FAFAFA;">
                    STATOUR
                </span>
                <span style="font-size: 0.85rem; color: #A0A4B0; margin-left: 8px;">
                    Plateforme Intelligente du Tourisme Marocain
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        session_mgr = st.session_state.get("session_manager")
        if session_mgr:
            conv = session_mgr.get_active_conversation()
            if conv and conv.messages:
                agent = conv.get_last_agent()
                icons = {"normal": "🏛️", "researcher": "🔍", "analytics": "📊"}
                st.markdown(
                    f"<div style='text-align:right; font-size:0.8rem; color:#A0A4B0;'>"
                    f"{icons.get(agent, '🤖')} {len(conv.messages)} msgs</div>",
                    unsafe_allow_html=True,
                )

    st.markdown(
        "<hr style='margin:0; border-color:#2D3139;'>",
        unsafe_allow_html=True,
    )