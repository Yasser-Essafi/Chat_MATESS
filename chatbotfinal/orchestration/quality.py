"""Quality checks for orchestration outputs."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List


def _norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def requested_chart_count(message: str) -> int:
    norm = _norm_text(message)
    count = 0
    for raw in re.findall(r"\b([2-4])\s+(?:graph|chart|courbe|visual)", norm):
        count = max(count, int(raw))
    word_counts = {"deux": 2, "trois": 3, "quatre": 4}
    for word, value in word_counts.items():
        if re.search(rf"\b{word}\s+(?:graph|chart|courbe|visual)", norm):
            count = max(count, value)
    if any(k in norm for k in ["plusieurs graph", "multiple chart", "multiple graph", "des graphiques"]):
        count = max(count, 2)
    if any(k in norm for k in ["dashboard", "tableau de bord"]):
        count = max(count, 3)
    if any(k in norm for k in ["graph", "chart", "courbe", "visualis", "diagramme", "heatmap"]):
        count = max(count, 1)
    return min(count, 4)


def is_compound_request(message: str) -> bool:
    norm = _norm_text(message)
    connectors = [" et ", " plus ", " avec ", " ainsi que ", " puis ", "compare", "analyse"]
    deliverables = ["graph", "chart", "courbe", "tableau", "analyse", "recommand", "paragraphe"]
    return sum(1 for item in deliverables if item in norm) >= 2 or any(item in norm for item in connectors)


@dataclass
class QualityReport:
    missing: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing


def inspect_deliverables(message: str, response: str, chart_paths: List[str]) -> QualityReport:
    missing: List[str] = []
    requested_charts = requested_chart_count(message)
    if requested_charts and len(chart_paths or []) < requested_charts:
        missing.append(f"{requested_charts - len(chart_paths or [])} graphique(s) supplementaire(s)")
    if is_compound_request(message):
        heading_count = len(re.findall(r"(?m)^##\s+", response or ""))
        paragraph_count = len([p for p in re.split(r"\n\s*\n", response or "") if len(p.strip()) > 80])
        if heading_count < 2 and paragraph_count < 2:
            missing.append("analyse narrative structuree")
    norm = _norm_text(message)
    if "pourquoi" in norm and not any(k in _norm_text(response) for k in ["cause", "facteur", "explication", "pourquoi"]):
        missing.append("explication causale")
    if any(k in norm for k in ["top management", "ministre", "decision", "decisionnel"]) and "decision" not in _norm_text(response):
        missing.append("decisions recommandees")
    return QualityReport(missing=missing[:4])
