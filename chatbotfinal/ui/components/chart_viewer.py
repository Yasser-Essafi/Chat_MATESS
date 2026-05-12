"""
STATOUR Premium Chart Viewer Component
=======================================
Enhanced chart display component for premium ChartEngine charts.
Supports multiple charts, responsive sizing, and dark theme styling.
"""

import os
import re
import streamlit as st
import streamlit.components.v1 as components
from typing import Optional, List, Tuple

try:
    from utils.logger import get_logger
    logger = get_logger("statour.ui.chart")
except Exception:
    import logging
    logger = logging.getLogger("statour.ui.chart")


# Premium chart styling constants (matching ChartEngine theme)
CHART_CONTAINER_STYLE = """
    background: linear-gradient(135deg, #161A23 0%, #0F1117 100%);
    border-radius: 16px;
    border: 1px solid rgba(212, 175, 55, 0.15);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 1px rgba(212, 175, 55, 0.1);
    overflow: hidden;
    margin: 12px 0;
    padding: 4px;
"""

CHART_HEADER_STYLE = """
    background: linear-gradient(90deg, rgba(196, 30, 58, 0.1) 0%, transparent 100%);
    padding: 8px 16px;
    border-bottom: 1px solid rgba(212, 175, 55, 0.1);
    display: flex;
    justify-content: space-between;
    align-items: center;
"""


def extract_chart_paths(text: str) -> List[str]:
    """Extract all chart file paths from assistant response text.
    
    Returns a list of valid chart paths found in the text.
    Supports multiple charts in a single response.
    """
    if not text:
        return []

    patterns = [
        r"📊\s*Chart:\s*(.+\.html)",
        r"Chart:\s*(.+\.html)",
        r"chart[_\s]*path[:\s]*(.+\.html)",
        r"(charts[/\\][^\s\"'<>]+\.html)",
    ]

    paths = []
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            path = match.strip()
            # Try absolute path first
            if os.path.exists(path):
                if path not in paths:
                    paths.append(path)
                continue
            # Try relative to project root
            abs_path = os.path.join(project_root, path)
            if os.path.exists(abs_path):
                if abs_path not in paths:
                    paths.append(abs_path)

    return paths


def extract_chart_path(text: str) -> Optional[str]:
    """Extract single chart file path from assistant response text.
    
    For backward compatibility - returns first chart path found.
    """
    paths = extract_chart_paths(text)
    return paths[0] if paths else None


def get_chart_title(chart_path: str) -> str:
    """Extract chart title from filename or HTML content."""
    # Try to get a readable name from the filename
    filename = os.path.basename(chart_path)
    # Remove extension and timestamp patterns
    name = re.sub(r'\.html$', '', filename)
    name = re.sub(r'_\d{8}_\d{6}(_\d+)?', '', name)
    name = re.sub(r'^chart_', '', name)
    name = name.replace('_', ' ').title()
    return name if name else "Graphique"


def render_chart(
    chart_path: str,
    height: int = 550,
    show_header: bool = True,
    title: Optional[str] = None
) -> bool:
    """Render a premium Plotly HTML chart inline.
    
    Args:
        chart_path: Path to the chart HTML file
        height: Height of the chart container in pixels
        show_header: Whether to show the chart header with title
        title: Optional custom title (extracted from filename if not provided)
    
    Returns:
        True if chart rendered successfully, False otherwise
    """
    if not chart_path or not os.path.exists(chart_path):
        logger.warning("Chart file not found: %s", chart_path)
        return False

    try:
        with open(chart_path, "r", encoding="utf-8") as f:
            chart_html = f.read()

        chart_title = title or get_chart_title(chart_path)
        
        # Build header HTML if requested
        header_html = ""
        if show_header and chart_title:
            header_html = f"""
            <div style="{CHART_HEADER_STYLE}">
                <span style="
                    color: #E8E8E8;
                    font-family: 'Inter', 'Segoe UI', sans-serif;
                    font-size: 14px;
                    font-weight: 500;
                ">📊 {chart_title}</span>
                <span style="
                    color: #8B919E;
                    font-family: 'Inter', 'Segoe UI', sans-serif;
                    font-size: 11px;
                ">STATOUR Premium</span>
            </div>
            """

        wrapped_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                }}
                .chart-container {{
                    {CHART_CONTAINER_STYLE}
                }}
                .chart-content {{
                    background: #0F1117;
                }}
            </style>
        </head>
        <body>
            <div class="chart-container">
                {header_html}
                <div class="chart-content">
                    {chart_html}
                </div>
            </div>
        </body>
        </html>
        """

        components.html(wrapped_html, height=height + (50 if show_header else 0), scrolling=False)
        logger.debug("Chart rendered: %s", chart_path)
        return True

    except Exception as e:
        logger.error("Failed to render chart %s: %s", chart_path, e)
        st.error(f"Erreur d'affichage du graphique: {e}")
        return False


def render_multiple_charts(
    chart_paths: List[str],
    height: int = 500,
    columns: int = 1
) -> int:
    """Render multiple charts in a grid layout.
    
    Args:
        chart_paths: List of paths to chart HTML files
        height: Height of each chart container
        columns: Number of columns in the grid (1 for stacked, 2 for side-by-side)
    
    Returns:
        Number of charts successfully rendered
    """
    if not chart_paths:
        return 0
    
    rendered = 0
    
    if columns == 1 or len(chart_paths) == 1:
        # Single column layout
        for path in chart_paths:
            if render_chart(path, height=height):
                rendered += 1
    else:
        # Multi-column layout
        for i in range(0, len(chart_paths), columns):
            cols = st.columns(columns)
            for j, col in enumerate(cols):
                if i + j < len(chart_paths):
                    with col:
                        if render_chart(chart_paths[i + j], height=height - 50):
                            rendered += 1
    
    return rendered


def render_charts_from_message(content: str, max_charts: int = 4) -> int:
    """Extract and render all charts from message content.
    
    Args:
        content: Message content that may contain chart paths
        max_charts: Maximum number of charts to render
    
    Returns:
        Number of charts rendered
    """
    chart_paths = extract_chart_paths(content)[:max_charts]
    
    if not chart_paths:
        return 0
    
    # Use 2 columns for 2-4 charts, single column otherwise
    columns = 2 if 1 < len(chart_paths) <= 4 else 1
    return render_multiple_charts(chart_paths, columns=columns)


def render_chart_from_message(content: str) -> bool:
    """Try to extract and render a single chart from message content.
    
    For backward compatibility.
    """
    chart_path = extract_chart_path(content)
    if chart_path:
        return render_chart(chart_path)
    return False