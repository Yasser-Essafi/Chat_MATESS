"""
Humanizer End-Node — Warm Conversational Response Layer
========================================================
Every response passes through the Humanizer before reaching the user.
Ensures consistent, warm, human tone regardless of which tools were used.

For simple queries: generates a direct, friendly answer
For complex queries: structures a deep report in engaging, clear French
"""

import re
import time
from typing import Optional, Dict, Any, List

from utils.base_agent import get_shared_client, _prepare_messages, _should_skip_reasoning_effort
from utils.logger import get_logger
from config.settings import (
    AZURE_OPENAI_DEPLOYMENT,
    MODEL_IS_REASONING,
    RESEARCHER_REASONING_EFFORT,
    RESEARCHER_MAX_COMPLETION_TOKENS,
    ANALYTICS_MAX_COMPLETION_TOKENS,
)
from .followup import is_short_followup, context_has_data_turn

logger = get_logger("statour.orchestration.humanizer")


# ══════════════════════════════════════════════════════════════════════════════
# Humanizer System Prompts
# ══════════════════════════════════════════════════════════════════════════════

_HUMANIZER_SIMPLE_PROMPT = """Tu es un(e) analyste touristique senior du Ministère du Tourisme du Maroc (MTAESS), chaleureux(se) et passionné(e).

PERSONNALITÉ:
- Tu parles comme un(e) collègue expert(e) de confiance, pas comme un robot
- Tu es naturel(le), direct(e), et engageant(e)
- Tu utilises un langage vivant et accessible
- Tu es concis(e) : 2-5 phrases maximum pour les réponses simples
- Tu adaptes ta langue à celle de l'utilisateur (français par défaut)

POUR LES SALUTATIONS:
- Sois chaleureux(se) et accueillant(e)
- Mentionne brièvement ce que tu peux faire, sans lister de manière robotique
- Reste naturel(le) : pas de bullet points systématiques

POUR LES REMERCIEMENTS/AU REVOIR:
- Réponse courte et chaleureuse
- Propose subtilement de l'aide future

POUR LES QUESTIONS SIMPLES:
- Réponds directement avec le fait demandé
- Ajoute un mot de contexte si pertinent
- Pas de structure formelle, juste une réponse naturelle

JAMAIS:
- De ton bureaucratique ou template-heavy
- De "je suis un assistant IA" ou "en tant qu'IA"
- De bullet points pour une réponse simple
- De formulations robotiques ("Voici les informations demandées", "Je vous informe que")
- D'emojis excessifs (1 maximum par message, et uniquement si naturel)"""


_HUMANIZER_COMPLEX_PROMPT = """Tu es un(e) analyste touristique senior du Ministère du Tourisme du Maroc (MTAESS).
Tu rédiges des analyses claires, engageantes et professionnelles.

PERSONNALITÉ:
- Expert(e) passionné(e) par les données touristiques
- Tu t'adresses à des collègues du Ministère comme un partenaire de confiance
- Ton ton est professionnel MAIS chaleureux et accessible
- Tu racontes l'histoire que les données révèlent, pas juste des chiffres

STRUCTURE POUR LES ANALYSES COMPLEXES:
1. Accroche contextuelle (1-2 phrases qui captent l'attention avec le fait marquant)
2. Corps d'analyse (données structurées, mais avec des transitions naturelles)
3. Éclairage/perspective (ce que ça signifie concrètement)

STYLE D'ÉCRITURE:
- Utilise des transitions naturelles entre les sections
- Préfère les phrases complètes aux bullet points quand possible
- Quand tu utilises des tableaux, introduis-les avec une phrase contextuelle
- Les chiffres importants méritent d'être mis en valeur avec du **gras**
- Termine par une phrase d'ouverture (pas de "N'hésitez pas à me solliciter")

RÈGLES MÉTIER STRICTES:
- "nationalité" dans les données = TOUJOURS "pays de résidence" (jamais nationalité ethnique)
- "arrivées" est AMBIGU: distinguer APF (frontières) vs hébergement (check-ins hôteliers)
- 2026 n'a que Jan-Fév: ne JAMAIS comparer 2026 complet avec d'autres années complètes
- Citer la source des données: (APF — base interne), (Hébergement — STDN+Estimatif)

JAMAIS:
- De copier-coller brut des résultats SQL
- De ton robotique ou bureaucratique
- De "Voici les résultats" sans contexte
- De comparer des périodes non comparables sans le signaler
- D'inventer des données non présentes dans le contexte fourni

CONTEXTE FOURNI:
Les données brutes et résultats d'analyse te seront donnés. Ta mission est de les transformer en une réponse humaine, engageante et précise."""


# ══════════════════════════════════════════════════════════════════════════════
# Humanizer Functions
# ══════════════════════════════════════════════════════════════════════════════

_DATA_QUESTION_RE = re.compile(
    r"\b(combien|total|arrivee|arrivees|nuitee|nuitees|touriste|touristes|"
    r"mre|tes|apf|hebergement|hotel|region|classement|top\s*\d+|20\d{2})\b",
    re.IGNORECASE,
)


def _simple_turn_needs_evidence(message: str, intent: str, conversation_context: str) -> bool:
    if intent in {"greeting", "thanks", "farewell", "platform_qa"}:
        return False
    if _DATA_QUESTION_RE.search(message or ""):
        return True
    return bool(is_short_followup(message) and context_has_data_turn(conversation_context))


def _evidence_required_response(language: str) -> str:
    if language == "en":
        return (
            "I should verify this against the STATOUR data before giving numbers. "
            "Please resend the request as a data query, or specify the metric and period."
        )
    if language == "ar":
        return "يجب التحقق من بيانات STATOUR قبل إعطاء أرقام. يرجى تحديد المؤشر والفترة."
    return (
        "Je dois verifier les donnees STATOUR avant de donner des chiffres. "
        "Precisez la metrique et la periode si besoin, puis je lancerai l'analyse avec source."
    )


def _has_numeric_claim(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))


def humanize_simple(
    message: str,
    intent: str,
    conversation_context: str = "",
    language: str = "fr",
) -> str:
    """
    Generate a warm, direct response for simple queries.

    Args:
        message: User's original message
        intent: Detected intent (greeting, thanks, farewell, platform_qa, general_question)
        conversation_context: Recent conversation context
        language: Detected language

    Returns:
        Humanized response string
    """
    start = time.time()
    client = get_shared_client()
    needs_evidence = _simple_turn_needs_evidence(message, intent, conversation_context)
    if needs_evidence:
        return _evidence_required_response(language)

    user_content = f"Message utilisateur: {message}"
    if intent:
        user_content += f"\nIntent détecté: {intent}"
    if conversation_context:
        user_content += f"\nContexte: {conversation_context[:300]}"
    user_content += f"\nLangue de reponse obligatoire: {language or 'fr'}"

    messages = [
        {"role": "system", "content": _HUMANIZER_SIMPLE_PROMPT},
        {"role": "user", "content": user_content},
    ]

    kwargs: Dict[str, Any] = {
        "model": AZURE_OPENAI_DEPLOYMENT,
        "messages": _prepare_messages(messages),
        "max_completion_tokens": 500,
    }
    if MODEL_IS_REASONING and not _should_skip_reasoning_effort():
        kwargs["reasoning_effort"] = "minimal"

    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content.strip() if response.choices else ""
        duration = (time.time() - start) * 1000
        logger.debug("Humanize[simple] %.0fms: %s", duration, content[:80])
        return content or _fallback_simple(intent, language)
    except Exception as e:
        logger.warning("Humanizer simple failed: %s", str(e)[:100])
        return _fallback_simple(intent, language)


def humanize_complex(
    message: str,
    evidence_text: str,
    chart_paths: Optional[List[str]] = None,
    conversation_context: str = "",
    language: str = "fr",
) -> str:
    """
    Transform raw evidence/analysis into an engaging, human response.

    Args:
        message: User's original question
        evidence_text: Combined evidence from executor (text summaries)
        chart_paths: List of chart file paths generated
        conversation_context: Recent conversation context
        language: Detected language

    Returns:
        Humanized analytical response
    """
    start = time.time()
    client = get_shared_client()

    user_parts = [
        f"QUESTION DE L'UTILISATEUR: {message}",
        f"\nDONNÉES ET RÉSULTATS BRUTS:\n{evidence_text[:4000]}",
    ]

    if chart_paths:
        user_parts.append(f"\nGRAPHIQUES GÉNÉRÉS: {len(chart_paths)} graphique(s) interactif(s) disponible(s)")
        user_parts.append("Note: les graphiques seront affichés automatiquement, tu n'as pas besoin de les décrire en détail.")

    if conversation_context:
        user_parts.append(f"\nCONTEXTE CONVERSATION: {conversation_context[:300]}")

    user_parts.append(f"\nLangue de reponse obligatoire: {language or 'fr'}.")

    messages = [
        {"role": "system", "content": _HUMANIZER_COMPLEX_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]

    kwargs: Dict[str, Any] = {
        "model": AZURE_OPENAI_DEPLOYMENT,
        "messages": _prepare_messages(messages),
        "max_completion_tokens": ANALYTICS_MAX_COMPLETION_TOKENS,
    }
    if MODEL_IS_REASONING and not _should_skip_reasoning_effort():
        kwargs["reasoning_effort"] = RESEARCHER_REASONING_EFFORT

    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content.strip() if response.choices else ""
        duration = (time.time() - start) * 1000
        logger.info("Humanize[complex] %.0fms, %d chars", duration, len(content))

        if content:
            return content

    except Exception as e:
        logger.warning("Humanizer complex failed: %s — returning raw evidence", str(e)[:100])

    # Fallback: return evidence text cleaned up minimally
    return _fallback_complex(message, evidence_text)


# ══════════════════════════════════════════════════════════════════════════════
# Fallbacks (no LLM needed)
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_simple(intent: str, language: str) -> str:
    """Fallback responses when LLM is unavailable."""
    if language == "ar":
        fallbacks = {
            "greeting": "مرحبا! أنا محلل السياحة بمنصة STATOUR. كيف يمكنني مساعدتك؟",
            "thanks": "على الرحب والسعة! لا تتردد في السؤال إذا احتجت شيئا آخر.",
            "farewell": "إلى اللقاء! أتمنى لك يوما سعيدا.",
        }
    elif language == "en":
        fallbacks = {
            "greeting": "Hello! I'm the tourism analyst at STATOUR. How can I help you today?",
            "thanks": "You're welcome! Feel free to ask if you need anything else.",
            "farewell": "Goodbye! Have a great day.",
        }
    else:
        fallbacks = {
            "greeting": "Bonjour ! Je suis l'analyste tourisme de STATOUR. Comment puis-je vous aider aujourd'hui ?",
            "thanks": "Avec plaisir ! N'hésitez pas si vous avez d'autres questions.",
            "farewell": "Au revoir ! Bonne continuation.",
            "platform_qa": "STATOUR est la plateforme statistique du Ministère du Tourisme du Maroc. "
                          "Elle centralise les données sur les arrivées aux frontières, l'hébergement touristique, "
                          "et les prévisions sectorielles. Je peux vous aider à explorer ces données.",
            "general_question": "Je suis là pour vous aider avec les statistiques du tourisme marocain. "
                               "Posez-moi une question sur les arrivées, les nuitées, les prévisions ou les actualités du secteur.",
        }
    return fallbacks.get(intent, fallbacks.get("general_question", "Comment puis-je vous aider ?"))


def _fallback_complex(message: str, evidence_text: str) -> str:
    """Minimal formatting of evidence when humanizer LLM fails."""
    # Clean up the evidence: remove internal markers, excessive whitespace
    cleaned = evidence_text.strip()
    if not cleaned:
        return "Les données demandées n'ont pas pu être récupérées. Veuillez reformuler votre question."

    # Add a minimal intro
    return f"Voici les résultats pour votre demande :\n\n{cleaned[:3000]}"
