"""Default STATOUR orchestration nodes."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Callable, Optional

from .contracts import NodeContext, NodeResult, TraceStep
from .registry import NodeRegistry


def _norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def is_human_advisor_request(message: str) -> bool:
    """Detect opinion/advice turns that should feel like a human exchange."""
    norm = _norm_text(message)
    if not norm.strip() or norm.strip().startswith("/"):
        return False
    if any(k in norm for k in ["graph", "chart", "courbe", "tableau", "sql", "dataset"]):
        return False
    if re.search(r"\b(20[12]\d|2030)\b", norm):
        return False
    if any(k in norm for k in ["top management", "ministre", "ministeriel", "decisionnel"]):
        return False

    second_person = any(
        k in norm
        for k in [
            "tu pense", "tu penses", "t'en pense", "t en pense",
            "ton avis", "a ton avis", "à ton avis", "tu vois",
            "tu me recommande", "tu recommande", "tu recommandes",
            "tu ma rien recommander", "tu m'as rien recommande",
            "tu m as rien recommande", "tu ma rien recommende",
        ]
    )
    advice_intent = any(
        k in norm
        for k in [
            "recommande", "recommend", "conseille", "avis",
            "tu ferais quoi", "on devrait faire quoi", "que faire",
            "tu pense quoi", "tu penses quoi",
        ]
    )
    tourism_context = any(k in norm for k in ["tourisme", "touriste", "maroc", "destination", "secteur"])
    return tourism_context and (second_person or advice_intent)


@dataclass
class CallableNode:
    """Adapter for an existing agent callable."""

    key: str
    agent_name: str
    call: Callable[[NodeContext], NodeResult]
    predicate: Optional[Callable[[NodeContext], bool]] = None
    agent_icon: str = ""

    def can_handle(self, context: NodeContext) -> bool:
        return bool(self.predicate(context)) if self.predicate else False

    def run(self, context: NodeContext) -> NodeResult:
        result = self.call(context)
        if not result.agent:
            result.agent = self.key
        if not result.agent_name:
            result.agent_name = self.agent_name
        if not result.agent_icon:
            result.agent_icon = self.agent_icon
        return result


class CommandNode:
    """Slash command node for orchestrator and analytics commands."""

    key = "command"

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def can_handle(self, context: NodeContext) -> bool:
        return (context.message or "").strip().startswith("/")

    def run(self, context: NodeContext) -> NodeResult:
        message = (context.message or "").strip()
        lower = message.lower()
        orch = self.orchestrator

        cmd_result = orch.handle_orchestrator_commands(message)
        if cmd_result:
            return NodeResult(
                agent="normal",
                agent_name="Orchestrateur",
                response=cmd_result,
                trace=[TraceStep("command", "Commande orchestrateur", agent="command").to_dict()],
            )

        analytics_cmds = {
            "/datasets": lambda: orch.analytics_agent.list_datasets(),
            "/stats": lambda: orch.analytics_agent.quick_stats(),
            "/schema": lambda: orch.analytics_agent.get_schema(),
            "/columns": lambda: orch.analytics_agent.get_columns(),
            "/sample": lambda: orch.analytics_agent.get_sample(),
        }
        if lower in analytics_cmds:
            return NodeResult(
                agent="analytics",
                agent_name="Analyste de Donnees",
                response=analytics_cmds[lower](),
                trace=[TraceStep("command", "Commande analytics", agent="command").to_dict()],
            )
        if lower.startswith("/switch "):
            return NodeResult(
                agent="analytics",
                agent_name="Analyste de Donnees",
                response=orch.analytics_agent.switch_dataset(message.split(None, 1)[1]),
                trace=[TraceStep("command", "Changement dataset", agent="command").to_dict()],
            )
        return NodeResult(
            agent="normal",
            agent_name="Orchestrateur",
            response="Commande inconnue. Utilisez /help pour voir les commandes disponibles.",
            trace=[TraceStep("command", "Commande inconnue", status="error", agent="command").to_dict()],
            errors=[{"stage": "command", "message": "unknown_command"}],
        )


class HumanAdvisorNode:
    """Conversational tourism advisor for opinion and recommendation turns."""

    key = "human_advisor"

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def can_handle(self, context: NodeContext) -> bool:
        return is_human_advisor_request(context.message)

    def run(self, context: NodeContext) -> NodeResult:
        message = context.message or ""
        norm = _norm_text(message)
        complained = any(k in norm for k in ["rien recomm", "pas recomm", "tu ma rien", "tu m'as rien"])
        opener = (
            "Tu as raison: la reponse d'avant etait trop rapport, pas assez conseil."
            if complained
            else "Mon avis, franchement:"
        )
        response = (
            f"{opener}\n\n"
            "Le tourisme au Maroc est dans une tres bonne dynamique, mais le vrai sujet maintenant ce n'est plus seulement "
            "d'attirer plus de monde. C'est d'attirer mieux, de mieux repartir les flux, et de transformer les arrivees en "
            "nuitées, recettes et emplois plus stables.\n\n"
            "Ce que je recommanderais en priorite:\n\n"
            "1. Desaturer Marrakech et Agadir en poussant des circuits Casablanca-Rabat, Tanger-Tetouan et interieur du pays. "
            "Le Maroc a assez de matiere pour vendre autre chose que le triptyque soleil, medina, resort.\n\n"
            "2. Monter en qualite sur l'experience: signaletique, mobilite locale, langues, proprete, information digitale, "
            "service apres-arrivee. Le pays sait attirer; il doit maintenant rendre le sejour plus fluide.\n\n"
            "3. Travailler les saisons creuses avec des offres MICE, culture, sport, bien-etre et city-break. C'est la que les "
            "capacites hotelieres peuvent mieux respirer et que les recettes deviennent moins fragiles.\n\n"
            "4. Ne pas piloter seulement avec les arrivees. Je regarderais surtout les nuitées, la DMS, les recettes, la "
            "repartition par ville et la satisfaction. Beaucoup d'arrivees avec peu de nuitées, ce n'est pas forcement une victoire.\n\n"
            "Donc oui, je suis optimiste, mais pas en mode autopilote. Le Maroc a une fenetre enorme avant 2030; le risque, "
            "c'est de confondre croissance naturelle et strategie. La bonne bataille maintenant, c'est la qualite, la repartition "
            "et la valeur par visiteur."
        )
        return NodeResult(
            agent=self.key,
            agent_name="Conseiller Tourisme",
            response=response,
            trace=[TraceStep("human_advisor", "Avis conversationnel", agent=self.key).to_dict()],
            confidence="advisory",
            data_scope_note="Avis qualitatif; aucun graphique genere sans demande explicite.",
        )


def build_default_registry(orchestrator) -> NodeRegistry:
    """Build the default registry around the existing agents."""

    def _wrap_text(agent_key: str, agent_name: str, func):
        return CallableNode(
            key=agent_key,
            agent_name=agent_name,
            call=lambda ctx: NodeResult(agent=agent_key, agent_name=agent_name, response=func(ctx)),
        )

    registry = NodeRegistry()
    registry.register(CommandNode(orchestrator))
    registry.register(HumanAdvisorNode(orchestrator))
    registry.register(_wrap_text("normal", "Assistant General", lambda ctx: orchestrator.normal_agent.chat(ctx.message)))
    registry.register(_wrap_text(
        "analytics",
        "Analyste de Donnees",
        lambda ctx: orchestrator.analytics_agent.chat(ctx.message, domain_context=ctx.domain_context),
    ))
    registry.register(_wrap_text("researcher", "Chercheur Tourisme", lambda ctx: orchestrator.researcher_agent.chat(ctx.message)))

    def _prediction(ctx: NodeContext) -> NodeResult:
        if not orchestrator.prediction_agent:
            return NodeResult(
                agent="analytics",
                agent_name="Analyste de Donnees",
                response=orchestrator.analytics_agent.chat(ctx.message, domain_context=ctx.domain_context),
            )
        return NodeResult.from_legacy_dict(orchestrator.prediction_agent.chat(ctx.message))

    registry.register(CallableNode("prediction", "Previsionniste STATOUR", _prediction))

    def _executive(ctx: NodeContext) -> NodeResult:
        result = orchestrator.executive_agent.run(
            ctx.message,
            domain_context=ctx.domain_context,
            data_freshness=ctx.metadata.get("data_freshness") or {},
        )
        return NodeResult.from_legacy_dict(result)

    registry.register(CallableNode("executive_insight", "Analyste Executif", _executive))
    return registry
