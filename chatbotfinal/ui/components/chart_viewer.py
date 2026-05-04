"""
STATOUR Chart Viewer Component
"""

import os
import re
import streamlit as st
import streamlit.components.v1 as components
from typing import Optional

try:
    from utils.logger import get_logger
    logger = get_logger("statour.ui.chart")
except Exception:
    import logging
    logger = logging.getLogger("statour.ui.chart")


def extract_chart_path(text: str) -> Optional[str]:
    """Extract chart file path from assistant response text."""
    if not text:
        return None

    patterns = [
        r"📊\s*Chart:\s*(.+\.html)",
        r"Chart:\s*(.+\.html)",
        r"chart[_\s]*path[:\s]*(.+\.html)",
        r"((?:charts|/)[^\s]+\.html)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            path = match.group(1).strip()
            # Try both absolute and relative to project root
            if os.path.exists(path):
                return path
            # Try relative to project root
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            abs_path = os.path.join(project_root, path)
            if os.path.exists(abs_path):
                return abs_path

    return None


def render_chart(chart_path: str, height: int = 500) -> bool:
    """Render a Plotly HTML chart inline."""
    if not chart_path or not os.path.exists(chart_path):
        logger.warning("Chart file not found: %s", chart_path)
        return False

    try:
        with open(chart_path, "r", encoding="utf-8") as f:
            chart_html = f.read()

        wrapped_html = f"""
        <div style="
            background: #1A1D23;
            border-radius: 12px;
            border: 1px solid #2D3139;
            overflow: hidden;
            margin: 8px 0;
        ">
            {chart_html}
        </div>
        """

        components.html(wrapped_html, height=height, scrolling=False)
        logger.debug("Chart rendered: %s", chart_path)
        return True

    except Exception as e:
        logger.error("Failed to render chart %s: %s", chart_path, e)
        st.error(f"Erreur d'affichage du graphique: {e}")
        return False


def render_chart_from_message(content: str) -> bool:
    """Try to extract and render a chart from message content."""
    chart_path = extract_chart_path(content)
    if chart_path:
        return render_chart(chart_path)
    return False