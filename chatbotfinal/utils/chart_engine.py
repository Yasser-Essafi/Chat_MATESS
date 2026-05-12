"""
STATOUR Premium Chart Engine
==============================
Beautiful, interactive, and premium-styled Plotly chart generation
for the Ministère du Tourisme du Maroc (MTAESS).

Provides:
  - Premium color palettes (Morocco-inspired, modern gradients)
  - Layout templates with dark/light themes
  - 15+ chart types including maps, heatmaps, treemaps, etc.
  - Morocco region choropleth support
  - Consistent typography and spacing

Usage:
    from utils.chart_engine import (
        ChartEngine, CHART_TYPES, PREMIUM_COLORS, MOROCCO_REGIONS_GEO
    )
    engine = ChartEngine()
    fig = engine.bar(df, x="region", y="arrivees", title="Arrivées par région")
    fig.write_html("chart.html")
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

import pandas as pd
import numpy as np

# Plotly imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

from config.settings import CHARTS_DIR
from utils.logger import get_logger

logger = get_logger("statour.charts")


# ══════════════════════════════════════════════════════════════════════════════
# Premium Color Palettes
# ══════════════════════════════════════════════════════════════════════════════

PREMIUM_COLORS = {
    # Morocco-inspired palette: desert gold, atlas green, mediterranean blue, terracotta
    "primary": [
        "#C41E3A",  # Moroccan red
        "#006233",  # Atlas green
        "#D4AF37",  # Desert gold
        "#1E3A5F",  # Deep mediterranean blue
        "#E07A5F",  # Terracotta
        "#3D405B",  # Slate
        "#81B29A",  # Sage
        "#F2CC8F",  # Sand
        "#5E4B35",  # Dark earth
        "#9B2335",  # Deep crimson
    ],
    # Gradient palettes for continuous scales
    "morocco_gradient": [
        "#006233", "#1E8449", "#52BE80", "#A9DFBF",
        "#F9E79F", "#F4D03F", "#D4AC0D", "#C41E3A"
    ],
    "ocean_gradient": [
        "#0B3D4C", "#1A5F7A", "#2E8BC0", "#57A0D3",
        "#85C1E9", "#AED6F1", "#D6EAF8", "#EBF5FB"
    ],
    "sunset_gradient": [
        "#2C003E", "#5B2C6F", "#8E44AD", "#C0392B",
        "#E67E22", "#F39C12", "#F1C40F", "#F9E79F"
    ],
    # Monochromatic professional
    "slate": ["#1E293B", "#334155", "#475569", "#64748B", "#94A3B8", "#CBD5E1"],
    # High contrast for accessibility
    "accessible": [
        "#003f5c", "#2f4b7c", "#665191", "#a05195",
        "#d45087", "#f95d6a", "#ff7c43", "#ffa600"
    ],
}

# Default template colors
def _get_color_cycle(palette: str = "primary") -> List[str]:
    return PREMIUM_COLORS.get(palette, PREMIUM_COLORS["primary"])


# ══════════════════════════════════════════════════════════════════════════════
# Morocco Regions GeoJSON (simplified for choropleth)
# ══════════════════════════════════════════════════════════════════════════════

MOROCCO_REGIONS_GEO = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {"name": "Tanger-Tétouan-Al Hoceïma", "id": "TTA"}, "geometry": {"type": "Polygon", "coordinates": [[[-5.5, 35.5], [-5.0, 35.5], [-5.0, 36.0], [-5.5, 36.0], [-5.5, 35.5]]]}},
        {"type": "Feature", "properties": {"name": "L'Oriental", "id": "ORI"}, "geometry": {"type": "Polygon", "coordinates": [[[-2.5, 34.5], [-2.0, 34.5], [-2.0, 35.0], [-2.5, 35.0], [-2.5, 34.5]]]}},
        {"type": "Feature", "properties": {"name": "Fès-Meknès", "id": "FME"}, "geometry": {"type": "Polygon", "coordinates": [[[-5.0, 33.5], [-4.5, 33.5], [-4.5, 34.5], [-5.0, 34.5], [-5.0, 33.5]]]}},
        {"type": "Feature", "properties": {"name": "Rabat-Salé-Kénitra", "id": "RSK"}, "geometry": {"type": "Polygon", "coordinates": [[[-6.5, 34.0], [-6.0, 34.0], [-6.0, 34.5], [-6.5, 34.5], [-6.5, 34.0]]]}},
        {"type": "Feature", "properties": {"name": "Béni Mellal-Khénifra", "id": "BMK"}, "geometry": {"type": "Polygon", "coordinates": [[[-6.5, 32.0], [-6.0, 32.0], [-6.0, 32.5], [-6.5, 32.5], [-6.5, 32.0]]]}},
        {"type": "Feature", "properties": {"name": "Casablanca-Settat", "id": "CAS"}, "geometry": {"type": "Polygon", "coordinates": [[[-8.0, 33.0], [-7.5, 33.0], [-7.5, 33.5], [-8.0, 33.5], [-8.0, 33.0]]]}},
        {"type": "Feature", "properties": {"name": "Marrakech-Safi", "id": "MSA"}, "geometry": {"type": "Polygon", "coordinates": [[[-8.5, 31.5], [-8.0, 31.5], [-8.0, 32.5], [-8.5, 32.5], [-8.5, 31.5]]]}},
        {"type": "Feature", "properties": {"name": "Drâa-Tafilalet", "id": "DTA"}, "geometry": {"type": "Polygon", "coordinates": [[[-6.0, 30.5], [-5.0, 30.5], [-5.0, 31.5], [-6.0, 31.5], [-6.0, 30.5]]]}},
        {"type": "Feature", "properties": {"name": "Souss-Massa", "id": "SMA"}, "geometry": {"type": "Polygon", "coordinates": [[[-9.5, 29.5], [-9.0, 29.5], [-9.0, 30.5], [-9.5, 30.5], [-9.5, 29.5]]]}},
        {"type": "Feature", "properties": {"name": "Guelmim-Oued Noun", "id": "GON"}, "geometry": {"type": "Polygon", "coordinates": [[[-10.5, 28.5], [-10.0, 28.5], [-10.0, 29.0], [-10.5, 29.0], [-10.5, 28.5]]]}},
        {"type": "Feature", "properties": {"name": "Laâyoune-Sakia El Hamra", "id": "LSH"}, "geometry": {"type": "Polygon", "coordinates": [[[-13.5, 27.0], [-12.0, 27.0], [-12.0, 28.0], [-13.5, 28.0], [-13.5, 27.0]]]}},
        {"type": "Feature", "properties": {"name": "Dakhla-Oued Ed-Dahab", "id": "DOD"}, "geometry": {"type": "Polygon", "coordinates": [[[-17.0, 23.0], [-16.0, 23.0], [-16.0, 24.0], [-17.0, 24.0], [-17.0, 23.0]]]}},
    ]
}

# Region name normalisation map for matching data to geojson
REGION_NAME_MAP = {
    "tanger-tetouan-al hoceima": "Tanger-Tétouan-Al Hoceïma",
    "tanger tetouan al hoceima": "Tanger-Tétouan-Al Hoceïma",
    "tanger-tétouan-al hoceïma": "Tanger-Tétouan-Al Hoceïma",
    "oriental": "L'Oriental",
    "l'oriental": "L'Oriental",
    "fes-meknes": "Fès-Meknès",
    "fès-meknès": "Fès-Meknès",
    "fes meknes": "Fès-Meknès",
    "rabat-sale-kenitra": "Rabat-Salé-Kénitra",
    "rabat-salé-kénitra": "Rabat-Salé-Kénitra",
    "rabat sale kenitra": "Rabat-Salé-Kénitra",
    "beni mellal-khenifra": "Béni Mellal-Khénifra",
    "béni mellal-khénifra": "Béni Mellal-Khénifra",
    "beni mellal khenifra": "Béni Mellal-Khénifra",
    "casablanca-settat": "Casablanca-Settat",
    "marrakech-safi": "Marrakech-Safi",
    "draa-tafilalet": "Drâa-Tafilalet",
    "drâa-tafilalet": "Drâa-Tafilalet",
    "souss-massa": "Souss-Massa",
    "guelmim-oued noun": "Guelmim-Oued Noun",
    "laayoune-sakia el hamra": "Laâyoune-Sakia El Hamra",
    "laâyoune-sakia el hamra": "Laâyoune-Sakia El Hamra",
    "dakhla-oued ed-dahab": "Dakhla-Oued Ed-Dahab",
}


def normalize_region_name(name: str) -> str:
    """Normalize a region name to match the GeoJSON properties."""
    if not name or not isinstance(name, str):
        return ""
    key = name.strip().lower()
    return REGION_NAME_MAP.get(key, name.strip())


# ══════════════════════════════════════════════════════════════════════════════
# Chart Type Registry
# ══════════════════════════════════════════════════════════════════════════════

CHART_TYPES = [
    "bar", "line", "area", "pie", "donut", "treemap", "sunburst",
    "heatmap", "choropleth", "bubble", "scatter", "radar",
    "waterfall", "funnel", "combo", "density_heatmap"
]


# ══════════════════════════════════════════════════════════════════════════════
# Premium Layout Templates
# ══════════════════════════════════════════════════════════════════════════════

class ChartTheme:
    """Base theme configuration for all charts."""

    # Typography
    FONT_FAMILY = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
    FONT_FAMILY_TITLE = "'Playfair Display', Georgia, 'Times New Roman', serif"
    FONT_SIZE_TITLE = 22
    FONT_SIZE_SUBTITLE = 14
    FONT_SIZE_AXIS = 12
    FONT_SIZE_TICK = 11
    FONT_SIZE_LEGEND = 11
    FONT_SIZE_ANNOTATION = 11

    # Spacing
    MARGIN = dict(l=64, r=40, t=100, b=64)
    MARGIN_COMPACT = dict(l=48, r=32, t=80, b=48)

    # Colors - Dark premium theme (default)
    BG_COLOR = "#0F1117"
    PAPER_COLOR = "#161A23"
    GRID_COLOR = "#2D3139"
    TEXT_COLOR = "#E8E8E8"
    TEXT_COLOR_MUTED = "#8B919E"
    AXIS_COLOR = "#3D4350"
    ACCENT_COLOR = "#D4AF37"

    @classmethod
    def apply(cls, fig: go.Figure, title: str = "", subtitle: str = "",
              height: int = 600, compact: bool = False) -> go.Figure:
        """Apply the premium dark theme to a figure."""
        margin = cls.MARGIN_COMPACT if compact else cls.MARGIN

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b><br><span style='font-size:{cls.FONT_SIZE_SUBTITLE}px;color:{cls.TEXT_COLOR_MUTED}'>{subtitle}</span>" if subtitle else f"<b>{title}</b>",
                font=dict(family=cls.FONT_FAMILY_TITLE, size=cls.FONT_SIZE_TITLE, color=cls.TEXT_COLOR),
                x=0.02, xanchor="left",
            ),
            font=dict(family=cls.FONT_FAMILY, size=cls.FONT_SIZE_AXIS, color=cls.TEXT_COLOR),
            paper_bgcolor=cls.PAPER_COLOR,
            plot_bgcolor=cls.BG_COLOR,
            margin=margin,
            height=height,
            hoverlabel=dict(
                bgcolor=cls.PAPER_COLOR,
                font=dict(family=cls.FONT_FAMILY, size=13, color=cls.TEXT_COLOR),
                bordercolor=cls.GRID_COLOR,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.18 if not compact else -0.22,
                xanchor="center",
                x=0.5,
                font=dict(size=cls.FONT_SIZE_LEGEND, color=cls.TEXT_COLOR_MUTED),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
            ),
            colorway=PREMIUM_COLORS["primary"],
        )

        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=cls.GRID_COLOR,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor=cls.AXIS_COLOR,
            tickfont=dict(size=cls.FONT_SIZE_TICK, color=cls.TEXT_COLOR_MUTED),
            title_font=dict(size=cls.FONT_SIZE_AXIS, color=cls.TEXT_COLOR),
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=cls.GRID_COLOR,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor=cls.AXIS_COLOR,
            tickfont=dict(size=cls.FONT_SIZE_TICK, color=cls.TEXT_COLOR_MUTED),
            title_font=dict(size=cls.FONT_SIZE_AXIS, color=cls.TEXT_COLOR),
        )

        return fig


class LightTheme(ChartTheme):
    """Light variant of the premium theme."""
    BG_COLOR = "#FAFBFC"
    PAPER_COLOR = "#FFFFFF"
    GRID_COLOR = "#E8ECF1"
    TEXT_COLOR = "#1A1D23"
    TEXT_COLOR_MUTED = "#5A6270"
    AXIS_COLOR = "#C5CDD8"
    ACCENT_COLOR = "#C41E3A"


# ══════════════════════════════════════════════════════════════════════════════
# Chart Engine
# ══════════════════════════════════════════════════════════════════════════════

class ChartEngine:
    """
    Premium chart generation engine for STATOUR.

    Provides beautiful, interactive Plotly charts with consistent styling.
    All methods return a plotly.graph_objects.Figure that can be further
    customized before saving.
    """

    def __init__(self, theme: str = "dark"):
        if not PLOTLY_OK:
            raise ImportError("Plotly is required for ChartEngine")
        self.theme = LightTheme() if theme == "light" else ChartTheme()
        self.colors = PREMIUM_COLORS["primary"]
        self._chart_count = 0

    def _next_path(self, prefix: str = "chart") -> str:
        """Generate a unique chart file path."""
        self._chart_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{ts}_{self._chart_count}.html"
        path = os.path.join(CHARTS_DIR, filename).replace("\\", "/")
        os.makedirs(CHARTS_DIR, exist_ok=True)
        return path

    def _apply_theme(self, fig: go.Figure, title: str, subtitle: str = "",
                     height: int = 600, compact: bool = False) -> go.Figure:
        """Apply the configured theme to a figure."""
        return self.theme.apply(fig, title, subtitle, height, compact)

    def _format_hover(self, fig: go.Figure, template: str = "%{x}<br>%{y:,.0f}<extra></extra>") -> go.Figure:
        """Apply consistent hover template to all traces."""
        fig.update_traces(hovertemplate=template)
        return fig

    # ──────────────────────────────────────────────────────────────────────
    # Core Chart Types
    # ──────────────────────────────────────────────────────────────────────

    def bar(self, df: pd.DataFrame, x: str, y: str, color: Optional[str] = None,
            title: str = "", subtitle: str = "", horizontal: bool = False,
            height: int = 600, sort: bool = True, **kwargs) -> go.Figure:
        """Premium bar chart with gradient coloring option."""
        df = df.copy()
        if sort and not horizontal:
            df = df.sort_values(by=y, ascending=True)
        elif sort and horizontal:
            df = df.sort_values(by=y, ascending=False)

        orientation = "h" if horizontal else "v"
        x_arg = y if horizontal else x
        y_arg = x if horizontal else y

        fig = px.bar(
            df, x=x_arg, y=y_arg, color=color,
            orientation=orientation,
            color_discrete_sequence=self.colors,
            **kwargs
        )

        fig = self._apply_theme(fig, title, subtitle, height)
        # Note: marker_cornerradius is the correct property (no underscore before radius)
        # but it was added in Plotly 5.x. Use try/except for compatibility with older versions.
        try:
            fig.update_traces(
                marker_line_width=0,
                marker_cornerradius=4 if not horizontal else 0,
                texttemplate="%{y:,.0f}" if horizontal else "%{x:,.0f}",
                textposition="outside",
                textfont=dict(size=11, color=self.theme.TEXT_COLOR_MUTED),
                hovertemplate=("%{y}<br>%{x:,.0f}<extra></extra>" if horizontal
                               else "%{x}<br>%{y:,.0f}<extra></extra>"),
            )
        except (ValueError, TypeError):
            # Fallback for older Plotly versions that don't support cornerradius
            fig.update_traces(
                marker_line_width=0,
                texttemplate="%{y:,.0f}" if horizontal else "%{x:,.0f}",
                textposition="outside",
                textfont=dict(size=11, color=self.theme.TEXT_COLOR_MUTED),
                hovertemplate=("%{y}<br>%{x:,.0f}<extra></extra>" if horizontal
                               else "%{x}<br>%{y:,.0f}<extra></extra>"),
            )
        return fig

    def line(self, df: pd.DataFrame, x: str, y: str, color: Optional[str] = None,
             title: str = "", subtitle: str = "", height: int = 600,
             markers: bool = True, area: bool = False, **kwargs) -> go.Figure:
        """Premium line chart with smooth curves and markers."""
        fig = px.line(
            df, x=x, y=y, color=color,
            color_discrete_sequence=self.colors,
            **kwargs
        )

        fig = self._apply_theme(fig, title, subtitle, height)
        fig.update_traces(
            line=dict(width=3),
            mode="lines+markers" if markers else "lines",
            marker=dict(size=8, line=dict(width=2, color=self.theme.BG_COLOR)),
            hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>",
        )

        if area:
            for trace in fig.data:
                trace.fill = "tozeroy"
                trace.fillcolor = trace.line.color.replace(")", ",0.15)").replace("rgb", "rgba") if "rgb" in trace.line.color else trace.line.color

        fig.update_layout(hovermode="x unified")
        return fig

    def area(self, df: pd.DataFrame, x: str, y: str, color: Optional[str] = None,
             title: str = "", subtitle: str = "", height: int = 600, **kwargs) -> go.Figure:
        """Stacked area chart."""
        return self.line(df, x, y, color, title, subtitle, height, markers=False, area=True, **kwargs)

    def pie(self, df: pd.DataFrame, names: str, values: str,
            title: str = "", subtitle: str = "", height: int = 550,
            donut: bool = False, hole: float = 0.55, **kwargs) -> go.Figure:
        """Premium pie or donut chart."""
        fig = px.pie(
            df, names=names, values=values,
            color_discrete_sequence=self.colors,
            **kwargs
        )

        fig = self._apply_theme(fig, title, subtitle, height, compact=True)
        fig.update_traces(
            textinfo="label+percent",
            textposition="outside",
            textfont=dict(size=12, color=self.theme.TEXT_COLOR),
            marker=dict(line=dict(color=self.theme.BG_COLOR, width=2)),
            pull=[0.02] * len(df),
            hovertemplate="<b>%{label}</b><br>%{value:,.0f}<br>%{percent}<extra></extra>",
        )

        if donut:
            fig.update_traces(hole=hole)
            # Add center annotation
            total = df[values].sum()
            fig.add_annotation(
                text=f"<b>{total:,.0f}</b><br>Total",
                showarrow=False,
                font=dict(size=16, color=self.theme.TEXT_COLOR, family=self.theme.FONT_FAMILY),
                x=0.5, y=0.5,
            )

        return fig

    def donut(self, df: pd.DataFrame, names: str, values: str,
              title: str = "", subtitle: str = "", height: int = 550, **kwargs) -> go.Figure:
        """Premium donut chart."""
        return self.pie(df, names, values, title, subtitle, height, donut=True, **kwargs)

    def treemap(self, df: pd.DataFrame, path: List[str], values: str,
                title: str = "", subtitle: str = "", height: int = 600, **kwargs) -> go.Figure:
        """Interactive treemap for hierarchical data."""
        fig = px.treemap(
            df, path=path, values=values,
            color_discrete_sequence=self.colors,
            **kwargs
        )

        fig = self._apply_theme(fig, title, subtitle, height, compact=True)
        fig.update_traces(
            textfont=dict(size=13, color="white"),
            hovertemplate="<b>%{label}</b><br>%{value:,.0f}<extra></extra>",
            texttemplate="<b>%{label}</b><br>%{value:,.0f}",
            marker=dict(line=dict(color=self.theme.BG_COLOR, width=2)),
        )
        return fig

    def sunburst(self, df: pd.DataFrame, path: List[str], values: str,
                 title: str = "", subtitle: str = "", height: int = 600, **kwargs) -> go.Figure:
        """Sunburst chart for hierarchical data."""
        fig = px.sunburst(
            df, path=path, values=values,
            color_discrete_sequence=self.colors,
            **kwargs
        )

        fig = self._apply_theme(fig, title, subtitle, height, compact=True)
        fig.update_traces(
            textfont=dict(size=12, color="white"),
            hovertemplate="<b>%{label}</b><br>%{value:,.0f}<extra></extra>",
            insidetextorientation="radial",
        )
        return fig

    def heatmap(self, df: pd.DataFrame, x: str, y: str, z: str,
                title: str = "", subtitle: str = "", height: int = 600,
                colorscale: str = "morocco", **kwargs) -> go.Figure:
        """Premium heatmap with custom colorscale."""
        pivot = df.pivot(index=y, columns=x, values=z)

        cs = PREMIUM_COLORS.get(colorscale, PREMIUM_COLORS["morocco_gradient"])
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=cs,
            hovertemplate="%{y} | %{x}<br>%{z:,.0f}<extra></extra>",
            colorbar=dict(
                title=dict(text=z, font=dict(size=12, color=self.theme.TEXT_COLOR)),
                tickfont=dict(size=11, color=self.theme.TEXT_COLOR_MUTED),
                thickness=16,
                len=0.8,
            ),
        ))

        fig = self._apply_theme(fig, title, subtitle, height)
        fig.update_xaxes(side="top")
        return fig

    def density_heatmap(self, df: pd.DataFrame, x: str, y: str,
                        title: str = "", subtitle: str = "", height: int = 600, **kwargs) -> go.Figure:
        """2D density heatmap for correlation/scatter density."""
        fig = px.density_heatmap(
            df, x=x, y=y,
            color_continuous_scale=PREMIUM_COLORS["morocco_gradient"],
            **kwargs
        )
        fig = self._apply_theme(fig, title, subtitle, height)
        fig.update_traces(
            hovertemplate="%{x}<br>%{y}<br>Count: %{z}<extra></extra>",
        )
        return fig

    def bubble(self, df: pd.DataFrame, x: str, y: str, size: str,
               color: Optional[str] = None, title: str = "", subtitle: str = "",
               height: int = 600, **kwargs) -> go.Figure:
        """Bubble chart with size encoding."""
        fig = px.scatter(
            df, x=x, y=y, size=size, color=color,
            color_discrete_sequence=self.colors,
            size_max=50,
            **kwargs
        )

        fig = self._apply_theme(fig, title, subtitle, height)
        fig.update_traces(
            marker=dict(line=dict(width=1, color=self.theme.BG_COLOR)),
            hovertemplate="<b>%{text}</b><br>%{x}<br>%{y:,.0f}<br>Size: %{marker.size:,.0f}<extra></extra>",
        )
        return fig

    def scatter(self, df: pd.DataFrame, x: str, y: str,
                color: Optional[str] = None, title: str = "", subtitle: str = "",
                height: int = 600, trendline: bool = False, **kwargs) -> go.Figure:
        """Scatter plot with optional trendline."""
        fig = px.scatter(
            df, x=x, y=y, color=color,
            color_discrete_sequence=self.colors,
            trendline="ols" if trendline else None,
            **kwargs
        )

        fig = self._apply_theme(fig, title, subtitle, height)
        fig.update_traces(
            marker=dict(size=10, line=dict(width=1, color=self.theme.BG_COLOR)),
            hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>",
        )
        return fig

    def radar(self, df: pd.DataFrame, categories: str, values: str,
              group: Optional[str] = None, title: str = "", subtitle: str = "",
              height: int = 550, **kwargs) -> go.Figure:
        """Radar / spider chart for multi-dimensional comparison."""
        fig = go.Figure()

        if group:
            groups = df[group].unique()
            for i, g in enumerate(groups):
                sub = df[df[group] == g]
                fig.add_trace(go.Scatterpolar(
                    r=sub[values].tolist() + [sub[values].iloc[0]],
                    theta=sub[categories].tolist() + [sub[categories].iloc[0]],
                    fill="toself",
                    name=str(g),
                    line=dict(color=self.colors[i % len(self.colors)], width=2),
                    fillcolor=self.colors[i % len(self.colors)].replace(")", ",0.2)").replace("rgb", "rgba") if "rgb" in self.colors[i % len(self.colors)] else self.colors[i % len(self.colors)],
                ))
        else:
            fig.add_trace(go.Scatterpolar(
                r=df[values].tolist() + [df[values].iloc[0]],
                theta=df[categories].tolist() + [df[categories].iloc[0]],
                fill="toself",
                name=values,
                line=dict(color=self.colors[0], width=2),
            ))

        fig = self._apply_theme(fig, title, subtitle, height, compact=True)
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, gridcolor=self.theme.GRID_COLOR, tickfont=dict(size=10)),
                angularaxis=dict(gridcolor=self.theme.GRID_COLOR, tickfont=dict(size=11)),
                bgcolor=self.theme.BG_COLOR,
            ),
            showlegend=True,
        )
        return fig

    def waterfall(self, df: pd.DataFrame, x: str, y: str,
                  title: str = "", subtitle: str = "", height: int = 550, **kwargs) -> go.Figure:
        """Waterfall chart for showing cumulative changes."""
        measure = ["relative"] * (len(df) - 1) + ["total"]
        fig = go.Figure(go.Waterfall(
            name="",
            orientation="v",
            measure=measure,
            x=df[x].tolist(),
            y=df[y].tolist(),
            textposition="outside",
            text=[f"{v:,.0f}" for v in df[y]],
            connector=dict(line=dict(color=self.theme.GRID_COLOR, width=1)),
            increasing=dict(marker=dict(color="#52BE80")),
            decreasing=dict(marker=dict(color="#E07A5F")),
            totals=dict(marker=dict(color="#1E3A5F")),
        ))

        fig = self._apply_theme(fig, title, subtitle, height)
        return fig

    def funnel(self, df: pd.DataFrame, x: str, y: str,
               title: str = "", subtitle: str = "", height: int = 550, **kwargs) -> go.Figure:
        """Funnel chart for pipeline/conversion visualization."""
        fig = go.Figure(go.Funnel(
            y=df[x].tolist(),
            x=df[y].tolist(),
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(
                color=self.colors[:len(df)],
                line=dict(color=self.theme.BG_COLOR, width=2),
            ),
            connector=dict(line=dict(color=self.theme.GRID_COLOR, width=1)),
        ))

        fig = self._apply_theme(fig, title, subtitle, height, compact=True)
        return fig

    def combo(self, df: pd.DataFrame, x: str, bar_y: str, line_y: str,
              title: str = "", subtitle: str = "", height: int = 600, **kwargs) -> go.Figure:
        """Combo chart: bars + line on dual y-axes."""
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=df[x], y=df[bar_y],
                name=bar_y,
                marker_color=self.colors[0],
                marker_line_width=0,
                hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=df[x], y=df[line_y],
                name=line_y,
                mode="lines+markers",
                line=dict(color=self.colors[1], width=3),
                marker=dict(size=8),
                hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>",
            ),
            secondary_y=True,
        )

        fig = self._apply_theme(fig, title, subtitle, height)
        fig.update_layout(
            barmode="group",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        )
        return fig

    def choropleth(self, df: pd.DataFrame, locations: str, z: str,
                   title: str = "", subtitle: str = "", height: int = 650,
                   color_scale: str = "morocco", **kwargs) -> go.Figure:
        """
        Choropleth map for Morocco regions.
        The `locations` column should contain region names that will be normalized.
        """
        df = df.copy()
        df[locations] = df[locations].apply(normalize_region_name)

        cs = PREMIUM_COLORS.get(color_scale, PREMIUM_COLORS["morocco_gradient"])

        fig = go.Figure(go.Choroplethmapbox(
            geojson=MOROCCO_REGIONS_GEO,
            locations=df[locations].tolist(),
            z=df[z].tolist(),
            featureidkey="properties.name",
            colorscale=cs,
            marker_opacity=0.85,
            marker_line_width=1,
            marker_line_color="white",
            colorbar=dict(
                title=dict(text=z, font=dict(size=12, color=self.theme.TEXT_COLOR)),
                tickfont=dict(size=11, color=self.theme.TEXT_COLOR_MUTED),
                thickness=16,
                len=0.7,
                x=0.96,
            ),
            hovertemplate="<b>%{location}</b><br>%{z:,.0f}<extra></extra>",
        ))

        fig = self._apply_theme(fig, title, subtitle, height)
        fig.update_layout(
            mapbox=dict(
                style="carto-darkmatter" if isinstance(self.theme, ChartTheme) and self.theme.BG_COLOR == "#0F1117" else "carto-positron",
                zoom=5,
                center=dict(lat=31.8, lon=-7.1),
            ),
            margin=dict(l=0, r=0, t=80, b=0),
        )
        return fig

    # ──────────────────────────────────────────────────────────────────────
    # Convenience: Save with premium config
    # ──────────────────────────────────────────────────────────────────────

    def save(self, fig: go.Figure, path: Optional[str] = None,
             prefix: str = "chart", include_plotlyjs: str = "cdn",
             config: Optional[Dict] = None) -> str:
        """Save figure to HTML with premium interactivity config."""
        if path is None:
            path = self._next_path(prefix)

        default_config = {
            "responsive": True,
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": os.path.splitext(os.path.basename(path))[0],
                "height": fig.layout.height or 600,
                "width": fig.layout.width or 900,
                "scale": 2,
            },
        }
        if config:
            default_config.update(config)

        fig.write_html(
            path,
            include_plotlyjs=include_plotlyjs,
            full_html=True,
            config=default_config,
        )
        logger.debug("Chart saved: %s", path)
        return path


# ══════════════════════════════════════════════════════════════════════════════
# Convenience Functions (module-level for easy sandbox injection)
# ══════════════════════════════════════════════════════════════════════════════

_engine: Optional[ChartEngine] = None


def _get_engine() -> ChartEngine:
    global _engine
    if _engine is None:
        _engine = ChartEngine(theme="dark")
    return _engine


def chart_bar(df, x, y, color=None, title="", subtitle="", horizontal=False, height=600, **kwargs):
    """Generate a premium bar chart and return the figure."""
    return _get_engine().bar(df, x, y, color, title, subtitle, horizontal, height, **kwargs)


def chart_line(df, x, y, color=None, title="", subtitle="", height=600, markers=True, **kwargs):
    """Generate a premium line chart and return the figure."""
    return _get_engine().line(df, x, y, color, title, subtitle, height, markers, **kwargs)


def chart_area(df, x, y, color=None, title="", subtitle="", height=600, **kwargs):
    """Generate a premium area chart and return the figure."""
    return _get_engine().area(df, x, y, color, title, subtitle, height, **kwargs)


def chart_pie(df, names, values, title="", subtitle="", height=550, **kwargs):
    """Generate a premium pie chart and return the figure."""
    return _get_engine().pie(df, names, values, title, subtitle, height, **kwargs)


def chart_donut(df, names, values, title="", subtitle="", height=550, **kwargs):
    """Generate a premium donut chart and return the figure."""
    return _get_engine().donut(df, names, values, title, subtitle, height, **kwargs)


def chart_treemap(df, path, values, title="", subtitle="", height=600, **kwargs):
    """Generate a premium treemap and return the figure."""
    return _get_engine().treemap(df, path, values, title, subtitle, height, **kwargs)


def chart_sunburst(df, path, values, title="", subtitle="", height=600, **kwargs):
    """Generate a premium sunburst chart and return the figure."""
    return _get_engine().sunburst(df, path, values, title, subtitle, height, **kwargs)


def chart_heatmap(df, x, y, z, title="", subtitle="", height=600, **kwargs):
    """Generate a premium heatmap and return the figure."""
    return _get_engine().heatmap(df, x, y, z, title, subtitle, height, **kwargs)


def chart_density_heatmap(df, x, y, title="", subtitle="", height=600, **kwargs):
    """Generate a premium density heatmap and return the figure."""
    return _get_engine().density_heatmap(df, x, y, title, subtitle, height, **kwargs)


def chart_bubble(df, x, y, size, color=None, title="", subtitle="", height=600, **kwargs):
    """Generate a premium bubble chart and return the figure."""
    return _get_engine().bubble(df, x, y, size, color, title, subtitle, height, **kwargs)


def chart_scatter(df, x, y, color=None, title="", subtitle="", height=600, trendline=False, **kwargs):
    """Generate a premium scatter chart and return the figure."""
    return _get_engine().scatter(df, x, y, color, title, subtitle, height, trendline, **kwargs)


def chart_radar(df, categories, values, group=None, title="", subtitle="", height=550, **kwargs):
    """Generate a premium radar chart and return the figure."""
    return _get_engine().radar(df, categories, values, group, title, subtitle, height, **kwargs)


def chart_waterfall(df, x, y, title="", subtitle="", height=550, **kwargs):
    """Generate a premium waterfall chart and return the figure."""
    return _get_engine().waterfall(df, x, y, title, subtitle, height, **kwargs)


def chart_funnel(df, x, y, title="", subtitle="", height=550, **kwargs):
    """Generate a premium funnel chart and return the figure."""
    return _get_engine().funnel(df, x, y, title, subtitle, height, **kwargs)


def chart_combo(df, x, bar_y, line_y, title="", subtitle="", height=600, **kwargs):
    """Generate a premium combo chart and return the figure."""
    return _get_engine().combo(df, x, bar_y, line_y, title, subtitle, height, **kwargs)


def chart_choropleth(df, locations, z, title="", subtitle="", height=650, **kwargs):
    """Generate a premium Morocco choropleth map and return the figure."""
    return _get_engine().choropleth(df, locations, z, title, subtitle, height, **kwargs)


def chart_save(fig, path=None, prefix="chart", **kwargs):
    """Save a chart figure to HTML with premium settings."""
    return _get_engine().save(fig, path, prefix, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Sandbox-safe namespace builder
# ══════════════════════════════════════════════════════════════════════════════

def get_chart_namespace() -> Dict[str, Any]:
    """
    Return a dict of chart functions ready to inject into the analytics sandbox.
    Usage:
        exec_globals.update(get_chart_namespace())
    """
    namespace = {
        # Chart engine class
        "ChartEngine": ChartEngine,
        "PREMIUM_COLORS": PREMIUM_COLORS,
        "CHART_TYPES": CHART_TYPES,
        "MOROCCO_REGIONS_GEO": MOROCCO_REGIONS_GEO,
        "normalize_region_name": normalize_region_name,
        # Convenience functions
        "chart_bar": chart_bar,
        "chart_line": chart_line,
        "chart_area": chart_area,
        "chart_pie": chart_pie,
        "chart_donut": chart_donut,
        "chart_treemap": chart_treemap,
        "chart_sunburst": chart_sunburst,
        "chart_heatmap": chart_heatmap,
        "chart_density_heatmap": chart_density_heatmap,
        "chart_bubble": chart_bubble,
        "chart_scatter": chart_scatter,
        "chart_radar": chart_radar,
        "chart_waterfall": chart_waterfall,
        "chart_funnel": chart_funnel,
        "chart_combo": chart_combo,
        "chart_choropleth": chart_choropleth,
        "chart_save": chart_save,
    }
    
    # Add make_subplots for complex multi-chart layouts
    if PLOTLY_OK:
        namespace["make_subplots"] = make_subplots
    
    return namespace
