"""Conversation follow-up resolution for the orchestration graph."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


def _norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def _latest_line(context: str, prefix: str) -> Optional[str]:
    pattern = re.compile(rf"^\s*{re.escape(prefix)}:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
    matches = pattern.findall(context or "")
    return matches[-1].strip() if matches else None


def latest_data_agent(context: str) -> Optional[str]:
    for agent in ("PREDICTION", "ANALYTICS", "RESEARCHER"):
        if _latest_line(context, agent):
            return agent.lower()
    return None


def is_clear_social_turn(message: str) -> bool:
    norm = _norm_text(message).strip()
    return bool(re.fullmatch(
        r"(bonjour|bonsoir|salut|hello|hi|hey|merci|merci beaucoup|ok merci|"
        r"au revoir|bye|goodbye|a bientot|bonne journee|bonne soiree)[\s!?.]*",
        norm,
    ))


def is_short_followup(message: str) -> bool:
    norm = _norm_text(message).strip()
    words = norm.split()
    if not words or len(words) > 10:
        return False
    if is_clear_social_turn(message):
        return False
    if re.search(r"\b(et|pour|meme|pareil|idem|aussi|seulement|uniquement|en)\b", norm):
        return True
    if re.search(r"\b(19\d{2}|20\d{2}|mre|tes|apf|marrakech|casablanca|tanger|agadir)\b", norm):
        return True
    return False


def context_has_data_turn(context: str) -> bool:
    norm = _norm_text(context)
    return bool(
        latest_data_agent(context) in {"analytics", "prediction"}
        or re.search(
            r"\b(apf|hebergement|hotel|nuitee|arrivee|mre|tes|prediction|prevision|top\s*\d+|classement)\b",
            norm,
        )
    )


def resolve_followup(message: str, conversation_context: str) -> str:
    """Expand short analytical follow-ups with the prior request shape.

    The expansion is intentionally generic: it gives the planner the previous
    analytical request and asks it to preserve period, metric, ranking/chart
    shape, geography and segment unless the current turn changes them.
    """
    if not conversation_context or not is_short_followup(message):
        return message
    if not context_has_data_turn(conversation_context):
        return message

    previous_user = _latest_line(conversation_context, "USER")
    if not previous_user:
        return message

    return (
        f"{message}\n\n"
        "[Contexte de suivi STATOUR]\n"
        f"Demande analytique precedente: {previous_user}\n"
        "Conserver le meme type de livrable, la meme periode, la meme metrique, "
        "les memes dimensions et le meme niveau de detail, sauf si le nouveau "
        "message modifie explicitement l'un de ces elements."
    )
