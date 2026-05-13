"""
STATOUR Executive Intelligence Agent.

This agent is designed for ministerial decision support, not simple summary.
It combines Fabric evidence, RAG/web evidence, a structured evidence matrix,
gap detection, hypothesis testing, a red-team review and a final quality gate.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import AZURE_OPENAI_DEPLOYMENT
from utils.base_agent import get_shared_client, _prepare_messages
from utils.intent_extractor import IntentExtractor
from utils.logger import get_logger
from utils.mvp_services import STATIC_SIGNALS

logger = get_logger("statour.executive")


EXECUTIVE_SYSTEM_PROMPT = """Tu es STATOUR Executive Intelligence Agent, analyste stratégique pour le Ministère du Tourisme du Maroc.

Ta mission n'est pas de résumer des données. Ta mission est de produire une lecture décisionnelle, robuste, auditable et utile à un ministre, un secrétaire général ou un directeur central.

Principes obligatoires :
1. Commence par une thèse exécutive claire : favorable / fragile / contrastée / critique / insuffisamment documentée.
2. Sépare toujours données internes mesurées, signaux externes, hypothèses, interprétations et décisions recommandées.
3. Ne compare jamais des périodes non homogènes.
4. Ne mélange jamais APF, hébergement, nuitées, arrivées hôtelières, recettes, capacité et taux d'occupation.
5. Pour toute demande large, analyse au minimum : performance globale, marchés émetteurs, destinations/régions, hébergement/nuitées, connectivité aérienne, concurrence internationale, risques géopolitiques/macroéconomiques, opportunités court/moyen terme, implications de politique publique.
6. Chaque recommandation doit être liée à une preuve ou à une hypothèse explicite.
7. Si les données sont partielles, dis ce qu'on peut conclure et ce qu'on ne peut pas conclure.
8. Ajoute une section "Décisions recommandées" avec actions priorisées.
9. Ajoute une section "Questions à trancher par le ministère" si des arbitrages sont nécessaires.
10. Termine par un niveau de confiance justifié, pas seulement faible/moyen/élevé.

Ne révèle jamais de raisonnement caché ou chain-of-thought. Expose seulement une synthèse auditable : données utilisées, hypothèses, limites, sources, et justification de confiance.

Format obligatoire :
## Synthèse exécutive
## Diagnostic stratégique
## Analyse causale
## Implications pour le ministère
## Scénarios
## Décisions recommandées
## Questions à trancher par le ministère
## Données manquantes / limites
## Sources utilisées
## Confiance
"""


RED_TEAM_PROMPT = """Tu es le critique interne STATOUR. Audite la note avant livraison.
Retourne uniquement des constats courts, sans raisonnement caché.

Vérifie :
- La note est-elle trop descriptive ?
- Contient-elle une thèse centrale ?
- Les périodes sont-elles homogènes ?
- APF / hébergement / nuitées / recettes sont-ils séparés ?
- Les recommandations sont-elles liées à des preuves ?
- Les décisions sont-elles actionnables cette semaine, ce trimestre, cette année ?
- Les limites et données manquantes sont-elles explicites ?
- La confiance est-elle justifiée ?
"""


QUALITY_GATE_PROMPT = """Réécris la note en corrigeant les faiblesses du red-team.
Ne révèle pas de raisonnement caché. Garde uniquement la note finale structurée.
Toute recommandation doit être liée à une preuve ou à une hypothèse explicite.
"""


_EXECUTIVE_PATTERNS = [
    "pourquoi", "impact", "analyse", "analyser", "expliquer", "explication",
    "facteur", "facteurs", "cause", "causes", "risque", "risques",
    "recommand", "action", "actions", "scénario", "scenario",
    "top management", "direction", "décision", "decision", "stratégie",
    "strategie", "war", "guerre", "conflit", "géopolitique", "geopolitique",
    "situation", "diagnostic", "politique", "policies", "mesures",
]


@dataclass
class GapReport:
    missing_dimensions: List[str]
    requires_second_pass: bool
    queries: List[str]


def is_executive_insight_request(message: str) -> bool:
    msg = (message or "").lower()
    return any(p in msg for p in _EXECUTIVE_PATTERNS)


class ExecutiveInsightAgent:
    """Strategic tourism intelligence layer for ministry decision support."""

    agent_key = "executive_insight"
    agent_icon = ""
    agent_name = "Analyste Exécutif"

    def __init__(self, analytics_agent=None, researcher_agent=None):
        self.analytics_agent = analytics_agent
        self.researcher_agent = researcher_agent
        self.client = get_shared_client()
        self.deployment = AZURE_OPENAI_DEPLOYMENT
        self.intent_extractor = IntentExtractor()

    @staticmethod
    def _emit(step_callback: Optional[Callable[[Dict[str, Any]], None]], stage: str, label: str, detail: str = "") -> None:
        if step_callback:
            step_callback({"stage": stage, "label": label, "detail": detail})

    @staticmethod
    def _extract_chart_path(text: str) -> Optional[str]:
        if not text:
            return None
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        charts_dir = os.path.realpath(os.path.join(project_root, "charts"))
        patterns = [
            r"Chart:\s*(.+?\.html)",
            r"((?:[^\s]*/charts/|charts/)[^\s]+\.html)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            path = match.group(1).strip()
            if not os.path.isabs(path):
                path = os.path.join(project_root, path)
            real_path = os.path.realpath(path)
            if real_path.startswith(charts_dir) and os.path.exists(real_path):
                return real_path
        return None

    @staticmethod
    def _extract_chart_paths(text: str, limit: int = 4) -> List[str]:
        if not text:
            return []
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        charts_dir = os.path.realpath(os.path.join(project_root, "charts"))
        paths: List[str] = []
        for pattern in [r"Chart:\s*(.+?\.html)", r"((?:[^\s]*/charts/|charts/)[^\s]+\.html)"]:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                path = match.group(1).strip().strip("`),.;")
                if not os.path.isabs(path):
                    path = os.path.join(project_root, path)
                real_path = os.path.realpath(path)
                if real_path.startswith(charts_dir) and os.path.exists(real_path) and real_path not in paths:
                    paths.append(real_path)
                    if len(paths) >= limit:
                        return paths
        return paths

    @staticmethod
    def _internal_unavailable(internal_text: str) -> bool:
        t = (internal_text or "").lower()
        return any(
            marker in t
            for marker in [
                "aucune table",
                "fabric connection unavailable",
                "fabric connexion",
                "fabric n'est pas",
                "analyse interne indisponible",
                "no tables catalogued",
                "aucune table fabric",
            ]
        )

    def plan_analysis(self, message: str, intent: Any) -> Dict[str, Any]:
        msg = (message or "").lower()
        broad = any(k in msg for k in ["situation", "diagnostic", "analyse globale", "tourisme du maroc", "secteur"])
        policy = any(k in msg for k in ["politique", "policies", "mesures", "strategie", "stratégie", "feuille de route"])
        historical = len(re.findall(r"\b(19\d{2}|20\d{2})\b", msg)) >= 2
        causal = any(k in msg for k in ["pourquoi", "cause", "facteur", "impact", "risque", "expliquer"])

        dimensions = ["performance globale", "APF arrivées", "hébergement/nuitées", "marchés émetteurs"]
        if broad:
            dimensions.extend([
                "régions/destinations", "connectivité aérienne", "concurrence internationale",
                "risques macro/géopolitiques", "investissement/capacité", "politique publique",
            ])
        if policy:
            dimensions.extend(["politiques publiques", "réformes", "arbitrages budgétaires"])
        if historical:
            dimensions.extend(["série historique", "ruptures structurelles", "changements de politique"])
        if causal:
            dimensions.extend(["hypothèses causales", "facteurs temporaires vs structurels"])

        # Deduplicate while preserving order.
        seen = set()
        dimensions = [d for d in dimensions if not (d in seen or seen.add(d))]
        return {
            "question_type": "national_tourism_diagnostic" if broad else "targeted_executive_analysis",
            "decision_level": "ministerial",
            "required_dimensions": dimensions,
            "minimum_evidence_required": [
                "KPI interne comparable",
                "segmentation marché",
                "source externe indépendante",
                "limites et données manquantes",
                "décisions priorisées",
            ],
            "needs_policy_lens": policy or broad,
            "needs_historical_lens": historical,
            "needs_causal_lens": causal or broad,
            "intent": getattr(intent, "__dict__", {}),
        }

    def collect_internal_evidence_v2(self, message: str, plan: Dict[str, Any], domain_context: Optional[str] = None) -> Dict[str, Any]:
        if not self.analytics_agent:
            return {"text": "Aucun agent analytique disponible.", "chart_path": None}

        prompt = (
            f"{message}\n\n"
            "[MODE EXECUTIVE EVIDENCE V2]\n"
            "Objectif: produire uniquement les preuves internes Fabric utiles à une note ministérielle.\n"
            f"Dimensions à couvrir si disponibles: {', '.join(plan['required_dimensions'])}.\n"
            "Règles obligatoires:\n"
            "- Requêtes Fabric read-only via sql(query) uniquement.\n"
            "- Agréger dans SQL; ne jamais faire SELECT * sans TOP.\n"
            "- Séparer APF, hébergement, nuitées, arrivées hôtelières, recettes, taux d'occupation.\n"
            "- Comparer seulement des périodes homogènes; si 2026 est partielle, aligner les années sur les mêmes mois disponibles.\n"
            "- Identifier marchés gagnants/perdants si les colonnes le permettent.\n"
            "- Lister clairement les données absentes.\n"
            "- Rester factuel: pas d'explication externe, pas de recommandation.\n"
        )
        try:
            text = self.analytics_agent.chat(prompt, domain_context=domain_context)
            chart_paths = self._extract_chart_paths(text)
            return {"text": text, "chart_path": chart_paths[0] if chart_paths else None, "chart_paths": chart_paths}
        except Exception as e:
            logger.warning("Internal evidence failed: %s", e)
            return {"text": f"Analyse interne indisponible: {str(e)[:180]}", "chart_path": None, "chart_paths": []}

    def _search(self, query: str, max_results: int = 4) -> List[Dict[str, str]]:
        searcher = getattr(self.researcher_agent, "searcher", None)
        if not searcher:
            return []
        try:
            return searcher.search(query, max_results=max_results, use_trusted_only=True) or []
        except Exception as e:
            logger.debug("Executive search failed for %s: %s", query[:80], e)
            return []

    def collect_external_context_v2(self, message: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        base = message.strip()
        queries = [
            f"{base} Maroc tourisme actualité facteurs externes",
            "Maroc tourisme connectivité aérienne marchés émetteurs ONMT ONDA 2025 2026",
            "Maroc tourisme Vision 2030 feuille de route investissement hôtellerie",
        ]
        if plan.get("needs_policy_lens"):
            queries.extend([
                "Maroc politiques publiques tourisme feuille de route 2023 2026 ministère tourisme",
                "Maroc tourisme impact territorial emploi devises durabilité régions",
            ])
        if plan.get("needs_historical_lens"):
            queries.extend([
                f"{base} historique politiques tourisme Maroc",
                "Maroc tourisme stratégie 2010 Vision 2020 Vision 2030 politiques publiques",
            ])
        if "concurrence internationale" in plan["required_dimensions"]:
            queries.append("Maroc tourisme concurrence Espagne Turquie Egypte Tunisie Portugal 2025")
        if "risques macro/géopolitiques" in plan["required_dimensions"]:
            queries.append("tourisme international risques géopolitiques inflation pouvoir achat marchés européens 2025 2026")

        sources: List[Dict[str, str]] = []
        context_lines: List[str] = []
        seen_urls = set()
        for query in queries[:8]:
            for item in self._search(query, max_results=3):
                url = item.get("url") or ""
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                title = item.get("title") or item.get("source") or "Source"
                content = item.get("content") or ""
                sources.append({"title": title, "url": url, "source": item.get("source") or title})
                context_lines.append(f"- [{title}] {content[:650]} ({url})")
                if len(sources) >= 14:
                    break
            if len(sources) >= 14:
                break

        rag_context = ""
        try:
            rag = getattr(self.researcher_agent, "rag", None)
            if rag:
                rag_results = rag.search(message, n_results=5)
                rag_context = "\n".join(
                    f"- Base interne MTAESS: {r.get('content', '')[:500]}"
                    for r in rag_results
                )
        except Exception as e:
            logger.debug("Executive RAG context failed: %s", e)

        if not sources:
            sources = [
                {"title": s["title"], "url": s["url"], "source": s["source"]}
                for s in STATIC_SIGNALS
            ]
            context_lines = [
                f"- [{s['title']}] {s['content']} ({s['url']})"
                for s in STATIC_SIGNALS
            ]

        context = "\n".join([rag_context] + context_lines if rag_context else context_lines)
        return {"context": context, "sources": sources, "queries": queries[:8]}

    def build_evidence_matrix(self, plan: Dict[str, Any], internal: Dict[str, Any], external: Dict[str, Any]) -> Dict[str, Any]:
        internal_text = internal.get("text") or ""
        external_text = external.get("context") or ""
        claims = [
            {
                "claim": "Preuves internes disponibles",
                "source_type": "internal",
                "evidence": internal_text[:1800],
                "confidence": "medium" if not self._internal_unavailable(internal_text) else "low",
                "limitations": "Couverture limitée aux tables Fabric cataloguées et à la fraîcheur disponible.",
            },
            {
                "claim": "Signaux externes collectés",
                "source_type": "external",
                "evidence": external_text[:2200],
                "confidence": "medium" if external.get("sources") else "low",
                "limitations": "Les sources externes indiquent des facteurs plausibles mais ne prouvent pas seules la causalité.",
            },
        ]
        return {
            "plan": plan,
            "claims": claims,
            "sources": external.get("sources", []),
            "queries": external.get("queries", []),
            "chart_path": internal.get("chart_path"),
            "chart_paths": internal.get("chart_paths", []),
        }

    def detect_gaps(self, evidence_matrix: Dict[str, Any], data_freshness: Optional[Dict[str, Any]] = None) -> GapReport:
        text = json.dumps(evidence_matrix, ensure_ascii=False).lower()
        missing = []
        required = evidence_matrix["plan"]["required_dimensions"]
        checks = {
            "connectivité aérienne": ["aérien", "aerien", "air", "onda", "connectivité", "connectivity"],
            "concurrence internationale": ["espagne", "turquie", "egypte", "tunisie", "portugal", "concurr"],
            "risques macro/géopolitiques": ["géopolitique", "geopolitique", "inflation", "risque", "guerre", "conflit"],
            "régions/destinations": ["région", "region", "marrakech", "agadir", "tanger", "fès", "fes"],
            "politique publique": ["politique", "feuille de route", "vision", "ministère"],
            "série historique": ["historique", "1980", "1990", "2000", "world bank"],
        }
        for dim in required:
            terms = checks.get(dim)
            if terms and not any(term in text for term in terms):
                missing.append(dim)

        freshness = data_freshness or {}
        for key, item in freshness.items():
            months = item.get("months_latest_year") if isinstance(item, dict) else None
            if months and len(months) < 12:
                missing.append(f"{key}: année récente partielle")

        queries = []
        for gap in missing[:5]:
            queries.append(f"{gap} Maroc tourisme données récentes sources officielles")
        return GapReport(
            missing_dimensions=missing,
            requires_second_pass=bool(missing and len(evidence_matrix.get("sources", [])) < 12),
            queries=queries,
        )

    def second_pass_research(self, gaps: GapReport) -> Dict[str, Any]:
        sources = []
        lines = []
        seen_urls = set()
        for query in gaps.queries[:4]:
            for item in self._search(query, max_results=3):
                url = item.get("url") or ""
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                title = item.get("title") or item.get("source") or "Source"
                content = item.get("content") or ""
                sources.append({"title": title, "url": url, "source": item.get("source") or title})
                lines.append(f"- [{title}] {content[:650]} ({url})")
        return {"context": "\n".join(lines), "sources": sources}

    @staticmethod
    def merge_evidence(evidence_matrix: Dict[str, Any], additional: Dict[str, Any]) -> Dict[str, Any]:
        if additional.get("context"):
            evidence_matrix["claims"].append({
                "claim": "Recherche ciblée complémentaire",
                "source_type": "external_second_pass",
                "evidence": additional["context"][:2500],
                "confidence": "medium",
                "limitations": "Recherche ciblée déclenchée par lacunes détectées.",
            })
        urls = {s.get("url") for s in evidence_matrix.get("sources", [])}
        for src in additional.get("sources", []):
            if src.get("url") not in urls:
                evidence_matrix.setdefault("sources", []).append(src)
                urls.add(src.get("url"))
        return evidence_matrix

    def generate_hypotheses(self, evidence_matrix: Dict[str, Any]) -> List[Dict[str, str]]:
        return [
            {
                "hypothesis": "La dynamique est portée par des facteurs structurels et par des effets saisonniers/marchés.",
                "test": "Comparer APF, hébergement, nuitées et marchés sur périodes homogènes.",
            },
            {
                "hypothesis": "Le risque principal est la surestimation d'une tendance annuelle à partir de données partielles.",
                "test": "Vérifier la fraîcheur et les mois couverts par table.",
            },
            {
                "hypothesis": "La croissance des arrivées ne garantit pas automatiquement recettes, nuitées et dispersion régionale.",
                "test": "Croiser hébergement, recettes, capacité, régions et marchés émetteurs.",
            },
        ]

    def test_hypotheses(self, hypotheses: List[Dict[str, str]], evidence_matrix: Dict[str, Any]) -> List[Dict[str, str]]:
        text = json.dumps(evidence_matrix, ensure_ascii=False).lower()
        tested = []
        for h in hypotheses:
            status = "à confirmer"
            if any(term in text for term in ["jan", "fév", "fev", "partielle", "months_latest_year"]):
                if "partielles" in h["hypothesis"] or "partielle" in h["hypothesis"]:
                    status = "soutenue par la fraîcheur des données"
            if any(term in text for term in ["apf", "hébergement", "hebergement", "nuitées", "nuitees"]):
                if "arrivées" in h["hypothesis"] or "arrivees" in h["hypothesis"]:
                    status = "plausible, à quantifier"
            tested.append({**h, "status": status})
        return tested

    def _confidence_assessment(
        self,
        internal_text: str,
        sources: List[Dict[str, str]],
        gaps: GapReport,
        data_freshness: Optional[Dict[str, Any]],
    ) -> Dict[str, str]:
        score = 0
        reasons = []
        if not self._internal_unavailable(internal_text):
            score += 2
            reasons.append("données internes Fabric disponibles")
        else:
            reasons.append("données internes limitées ou indisponibles")
        if len(sources) >= 8:
            score += 2
            reasons.append("plusieurs sources externes utilisées")
        elif sources:
            score += 1
            reasons.append("quelques sources externes utilisées")
        if gaps.missing_dimensions:
            score -= 1
            reasons.append("lacunes détectées: " + ", ".join(gaps.missing_dimensions[:4]))
        freshness = data_freshness or {}
        if any(isinstance(v, dict) and len(v.get("months_latest_year", [])) < 12 for v in freshness.values()):
            score -= 1
            reasons.append("année récente partielle")
        level = "élevée" if score >= 3 else "moyenne" if score >= 1 else "faible"
        return {"level": level, "justification": "; ".join(reasons[:5])}

    def _llm_json_or_text(self, system: str, payload: Dict[str, Any], max_tokens: int = 1800) -> str:
        messages = _prepare_messages([
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ])
        resp = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            max_completion_tokens=max_tokens,
            reasoning_effort="low",
        )
        return (resp.choices[0].message.content or "").strip() if resp.choices else ""

    def synthesize_executive_brief(
        self,
        message: str,
        plan: Dict[str, Any],
        evidence_matrix: Dict[str, Any],
        hypotheses: List[Dict[str, str]],
        gaps: GapReport,
        confidence: Dict[str, str],
        data_freshness: Optional[Dict[str, Any]],
    ) -> str:
        payload = {
            "question": message,
            "analysis_plan": plan,
            "evidence_matrix": evidence_matrix,
            "tested_hypotheses": hypotheses,
            "gaps": gaps.__dict__,
            "confidence": confidence,
            "data_freshness": data_freshness or {},
            "instruction": "Produis une note ministérielle stratégique. Ne fais pas un résumé descriptif.",
        }
        try:
            text = self._llm_json_or_text(EXECUTIVE_SYSTEM_PROMPT, payload, max_tokens=3200)
            if text:
                return text
        except Exception as e:
            logger.warning("Executive synthesis failed: %s", e)
        return self._fallback_response(evidence_matrix, gaps, confidence)

    def red_team_review(self, draft: str, evidence_matrix: Dict[str, Any], original_question: str) -> str:
        payload = {
            "question": original_question,
            "draft": draft[:6000],
            "evidence_matrix": evidence_matrix,
        }
        try:
            critique = self._llm_json_or_text(RED_TEAM_PROMPT, payload, max_tokens=900)
            return critique or "Aucune critique générée."
        except Exception as e:
            logger.warning("Red-team review failed: %s", e)
            return self._deterministic_critique(draft)

    @staticmethod
    def _deterministic_critique(draft: str) -> str:
        issues = []
        required = [
            "## Synthèse exécutive",
            "## Décisions recommandées",
            "## Données manquantes / limites",
            "## Confiance",
        ]
        for section in required:
            if section.lower() not in draft.lower():
                issues.append(f"Section manquante: {section}")
        if "immédiat" not in draft.lower() and "3 mois" not in draft.lower():
            issues.append("Actions non priorisées par horizon.")
        return "\n".join(issues) or "Critique déterministe: structure minimale présente."

    def rewrite_with_quality_gate(
        self,
        draft: str,
        critique: str,
        evidence_matrix: Dict[str, Any],
        confidence: Dict[str, str],
    ) -> str:
        required_sections = [
            "## Synthèse exécutive",
            "## Diagnostic stratégique",
            "## Décisions recommandées",
            "## Données manquantes / limites",
            "## Sources utilisées",
            "## Confiance",
        ]
        needs_rewrite = any(section.lower() not in draft.lower() for section in required_sections)
        needs_rewrite = needs_rewrite or any(term in critique.lower() for term in ["manqu", "descriptive", "générique", "non prior"])
        if not needs_rewrite:
            return self._ensure_confidence_line(draft, confidence)
        payload = {
            "draft": draft[:6500],
            "critique": critique,
            "evidence_matrix": evidence_matrix,
            "confidence": confidence,
            "required_sections": required_sections,
        }
        try:
            fixed = self._llm_json_or_text(QUALITY_GATE_PROMPT + "\n" + EXECUTIVE_SYSTEM_PROMPT, payload, max_tokens=3200)
            return self._ensure_confidence_line(fixed or draft, confidence)
        except Exception as e:
            logger.warning("Quality gate rewrite failed: %s", e)
            return self._ensure_confidence_line(draft, confidence)

    @staticmethod
    def _ensure_confidence_line(text: str, confidence: Dict[str, str]) -> str:
        if "## Confiance" in text:
            return text
        return (
            text.rstrip()
            + f"\n\n## Confiance\n{confidence['level']} — {confidence['justification']}"
        )

    def _fallback_response(self, evidence_matrix: Dict[str, Any], gaps: GapReport, confidence: Dict[str, str]) -> str:
        sources = evidence_matrix.get("sources", [])
        source_lines = []
        for s in sources[:8]:
            label = s.get("source") or s.get("title") or "Source"
            url = s.get("url") or ""
            source_lines.append(f"- {label}: {url}" if url else f"- {label}")
        if not source_lines:
            source_lines = [f"- {s['source']}: {s['url']}" for s in STATIC_SIGNALS[:2]]

        gaps_text = ", ".join(gaps.missing_dimensions) or "aucune lacune majeure détectée automatiquement"
        return (
            "## Synthèse exécutive\n"
            "La situation ne peut pas être qualifiée de manière définitive sans compléter certaines dimensions. "
            "Les données disponibles permettent une lecture prudente, mais la décision ministérielle doit distinguer faits mesurés, hypothèses et limites.\n\n"
            "## Diagnostic stratégique\n"
            "- Les preuves internes et externes ont été collectées, mais certaines dimensions restent incomplètes.\n"
            f"- Points à compléter: {gaps_text}.\n\n"
            "## Analyse causale\n"
            "Les facteurs causaux doivent être interprétés comme hypothèses tant qu'ils ne sont pas quantifiés par marché, période et canal.\n\n"
            "## Implications pour le ministère\n"
            "Ne pas communiquer une conclusion annuelle si les données récentes sont partielles. Prioriser une lecture homogène APF/hébergement/nuitées/recettes.\n\n"
            "## Scénarios\n"
            "- Favorable: la dynamique se confirme sur marchés et hébergement.\n"
            "- Central: croissance modérée avec disparités par marché.\n"
            "- Défavorable: choc externe ou ralentissement des marchés européens.\n\n"
            "## Décisions recommandées\n"
            "- Immédiat: produire un tableau comparatif homogène.\n"
            "- 3 mois: isoler marchés et régions en ralentissement.\n"
            "- 12 mois: intégrer recettes, emploi, capacité et durabilité au pilotage 2030.\n\n"
            "## Questions à trancher par le ministère\n"
            "- Quel niveau de prudence adopter dans la communication publique ?\n"
            "- Quels marchés prioriser si les signaux divergent ?\n\n"
            "## Données manquantes / limites\n"
            f"{gaps_text}\n\n"
            "## Sources utilisées\n"
            + "\n".join(source_lines)
            + f"\n\n## Confiance\n{confidence['level']} — {confidence['justification']}"
        )

    @staticmethod
    def _norm(text: str) -> str:
        if any(marker in (text or "") for marker in ("Ã", "Â", "â")):
            try:
                text = text.encode("latin1").decode("utf-8")
            except Exception:
                pass
        text = unicodedata.normalize("NFKD", text or "")
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.lower()

    @staticmethod
    def _month_from_text(text: str) -> Optional[int]:
        norm = ExecutiveInsightAgent._norm(text)
        months = {
            "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4,
            "mai": 5, "juin": 6, "juillet": 7, "aout": 8,
            "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12,
        }
        for name, num in months.items():
            if re.search(r"\b" + re.escape(name) + r"\b", norm):
                return num
        return None

    def _decline_precheck(self, message: str) -> Optional[Dict[str, Any]]:
        norm = self._norm(message)
        asks_decline = any(k in norm for k in ["baisse", "baisser", "diminue", "recul"])
        asks_increase = any(k in norm for k in ["hausse", "augment", "progresse", "croissance"])
        if not ("pourquoi" in norm and (asks_decline or asks_increase)):
            return None
        year_match = re.search(r"\b(20[12]\d)\b", message or "")
        month = self._month_from_text(message)
        if not year_match or not month or not self.analytics_agent:
            return None
        year = int(year_match.group(1))
        prev_year = year - 1
        db = getattr(self.analytics_agent, "_db", None)
        if not db or getattr(db, "source", None) != "fabric":
            return None
        try:
            apf = db._qualify("fact_statistiques_apf")
            hbg = db._qualify("fact_statistiqueshebergementnationaliteestimees")
            apf_df = db.safe_query(
                f"SELECT YEAR(date_stat) AS annee, SUM(mre + tes) AS arrivees "
                f"FROM {apf} WHERE YEAR(date_stat) IN ({prev_year}, {year}) AND MONTH(date_stat) = {month} "
                f"GROUP BY YEAR(date_stat) ORDER BY annee"
            )
            hbg_df = db.safe_query(
                f"SELECT YEAR(date_stat) AS annee, SUM(arrivees) AS arrivees_hotelieres, SUM(nuitees) AS nuitees "
                f"FROM {hbg} WHERE YEAR(date_stat) IN ({prev_year}, {year}) AND MONTH(date_stat) = {month} "
                f"GROUP BY YEAR(date_stat) ORDER BY annee"
            )
        except Exception as e:
            logger.debug("Executive decline precheck failed: %s", e)
            return None

        month_name = {
            1: "janvier", 2: "fevrier", 3: "mars", 4: "avril", 5: "mai", 6: "juin",
            7: "juillet", 8: "aout", 9: "septembre", 10: "octobre", 11: "novembre", 12: "decembre",
        }.get(month, str(month))
        rows = []
        has_decline = False

        def _vals(df, col):
            if df.empty:
                return None, None
            by_year = {int(r["annee"]): float(r[col] or 0) for _, r in df.iterrows()}
            return by_year.get(prev_year), by_year.get(year)

        for label, df, col in [
            ("Arrivees APF", apf_df, "arrivees"),
            ("Arrivees hotelieres", hbg_df, "arrivees_hotelieres"),
            ("Nuitees hebergement", hbg_df, "nuitees"),
        ]:
            prev_val, cur_val = _vals(df, col)
            if prev_val is None or cur_val is None or prev_val == 0:
                continue
            delta = (cur_val / prev_val - 1) * 100
            has_decline = has_decline or delta < 0
            rows.append((label, prev_val, cur_val, delta))

        if not rows:
            return None

        table = [f"| Indicateur | {prev_year} | {year} | Variation |", "| :--- | :--- | :--- | :--- |"]
        for label, prev_val, cur_val, delta in rows:
            table.append(f"| {label} | {int(round(prev_val)):,} | {int(round(cur_val)):,} | {delta:+.1f}% |")
        if (asks_decline and has_decline) or (asks_increase and any(delta > 0 for _, _, _, delta in rows)):
            direction = "baisse" if asks_decline else "hausse"
            response = (
                "## Synthese executive\n"
                f"La {direction} doit d'abord etre lue sur un perimetre precis. "
                f"En {month_name} {year}, voici la verification Fabric disponible avant toute explication causale.\n\n"
                "## Verification factuelle prealable\n"
                + "\n".join(table)
                + "\n\n## Lecture ministerielle\n"
                "- Les causes ne doivent pas etre attribuees sans croiser marche emetteur, voie, region et calendrier evenementiel.\n"
                "- Les variations APF et hebergement peuvent diverger; ne pas les fusionner dans un seul narratif.\n"
                "- Pour une note decisionnelle complete, completer avec les donnees par nationalite/voie et les signaux externes sourcables.\n\n"
                "## Sources utilisees\n"
                "- Fabric Gold STATOUR: fact_statistiques_apf\n"
                "- Fabric Gold STATOUR: fact_statistiqueshebergementnationaliteestimees\n\n"
                "## Confiance\n"
                "elevee pour la verification quantitative; moyenne pour l'interpretation causale tant que les dimensions fines ne sont pas precisees."
            )
        else:
            expected = "baisse" if asks_decline else "hausse"
            response = (
                "## Synthese executive\n"
                f"Il n'y a **pas de {expected} mesuree** en {month_name} {year} sur les principaux perimetres internes disponibles. "
                "La question repose donc sur une hypothese a corriger avant toute analyse causale ou communication.\n\n"
                "## Verification factuelle prealable\n"
                + "\n".join(table)
                + "\n\n## Implication pour le ministere\n"
                "- Ne pas communiquer une explication nationale tant que le mouvement n'est pas mesure sur le perimetre vise.\n"
                "- Si le mouvement est observe localement, demander le perimetre exact: APF, hebergement, region, poste-frontiere, marche emetteur ou nuitees.\n\n"
                "## Donnees manquantes / limites\n"
                "Cette verification couvre les donnees Fabric disponibles pour le mois homogene N vs N-1. Elle ne prouve pas l'absence de mouvement sur un sous-segment local non precise.\n\n"
                "## Sources utilisees\n"
                "- Fabric Gold STATOUR: fact_statistiques_apf\n"
                "- Fabric Gold STATOUR: fact_statistiqueshebergementnationaliteestimees\n\n"
                "## Confiance\n"
                "elevee pour le constat quantitatif sur ces perimetres; moyenne si la question visait un sous-segment non precise."
            )
        return {
            "agent": self.agent_key,
            "agent_icon": self.agent_icon,
            "agent_name": self.agent_name,
            "response": response,
            "sources": [
                {"title": "Fabric Gold STATOUR APF", "source": "fact_statistiques_apf", "url": ""},
                {"title": "Fabric Gold STATOUR Hebergement", "source": "fact_statistiqueshebergementnationaliteestimees", "url": ""},
            ],
            "confidence": "elevee",
            "confidence_detail": {"level": "elevee", "justification": "precheck Fabric homogene N vs N-1"},
            "data_freshness": {},
            "intent": {},
            "analysis_plan": {},
            "gaps": {"missing_dimensions": [], "requires_second_pass": False, "queries": []},
            "chart_path": None,
            "chart_paths": [],
        }

    def run(
        self,
        message: str,
        domain_context: Optional[str] = None,
        data_freshness: Optional[Dict[str, Any]] = None,
        step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        self._emit(step_callback, "intent", "Cadrage décisionnel", "Qualification du niveau de décision, du type de question et des dimensions à couvrir.")
        intent = self.intent_extractor.extract(message)
        plan = self.plan_analysis(message, intent)

        self._emit(step_callback, "plan", "Plan d'analyse", ", ".join(plan["required_dimensions"][:8]))
        self._emit(step_callback, "database", "Collecte Fabric", "KPI internes, périodes comparables, marchés, APF/hébergement si disponibles.")
        internal = self.collect_internal_evidence_v2(message, plan, domain_context=domain_context)

        self._emit(step_callback, "search", "Recherche externe multi-axes", "Sources officielles/trustées, connectivité, risques, concurrence, politiques publiques.")
        external = self.collect_external_context_v2(message, plan)

        self._emit(step_callback, "evidence", "Matrice de preuves", "Structuration des faits, sources, hypothèses et limites.")
        evidence_matrix = self.build_evidence_matrix(plan, internal, external)

        self._emit(step_callback, "gaps", "Détection des lacunes", "Vérification des dimensions insuffisamment couvertes.")
        gaps = self.detect_gaps(evidence_matrix, data_freshness=data_freshness)

        if gaps.requires_second_pass:
            self._emit(step_callback, "second_pass", "Recherche ciblée complémentaire", ", ".join(gaps.missing_dimensions[:4]))
            additional = self.second_pass_research(gaps)
            evidence_matrix = self.merge_evidence(evidence_matrix, additional)

        self._emit(step_callback, "hypotheses", "Hypothèses stratégiques", "Séparation du structurel, temporaire, mesuré et supposé.")
        hypotheses = self.test_hypotheses(self.generate_hypotheses(evidence_matrix), evidence_matrix)

        confidence = self._confidence_assessment(
            internal.get("text", ""),
            evidence_matrix.get("sources", []),
            gaps,
            data_freshness,
        )

        self._emit(step_callback, "analysis", "Synthèse ministérielle", "Production d'une note décisionnelle avec scénarios et décisions.")
        draft = self.synthesize_executive_brief(
            message=message,
            plan=plan,
            evidence_matrix=evidence_matrix,
            hypotheses=hypotheses,
            gaps=gaps,
            confidence=confidence,
            data_freshness=data_freshness,
        )

        self._emit(step_callback, "red_team", "Red-team interne", "Critique de la note: thèse, causalité, limites, décisions, comparabilité.")
        critique = self.red_team_review(draft, evidence_matrix, message)

        self._emit(step_callback, "quality", "Quality gate", "Réécriture si la note est trop descriptive, générique ou incomplète.")
        response = self.rewrite_with_quality_gate(draft, critique, evidence_matrix, confidence)

        return {
            "agent": self.agent_key,
            "agent_icon": self.agent_icon,
            "agent_name": self.agent_name,
            "response": response,
            "sources": evidence_matrix.get("sources", []),
            "confidence": confidence["level"],
            "confidence_detail": confidence,
            "data_freshness": data_freshness or {},
            "intent": getattr(intent, "__dict__", {}),
            "analysis_plan": plan,
            "gaps": gaps.__dict__,
            "chart_path": evidence_matrix.get("chart_path"),
            "chart_paths": evidence_matrix.get("chart_paths", []),
        }
