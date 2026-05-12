"""STATOUR Orchestration — Plan-Execute-Review-Humanize Pipeline.

The existing Orchestrator remains the compatibility facade. This package
provides:
- Typed contracts and a node registry (legacy compatibility)
- Triage, Planner, Executor, Reviewer, Humanizer nodes (new flow)
- Graph runner that ties all nodes into a coherent pipeline
"""

from .contracts import Artifact, NodeContext, NodeResult, TraceStep
from .registry import NodeRegistry, OrchestrationNode
from .triage import triage, TriageResult
from .planner import generate_plan, ExecutionPlan, PlanStep
from .executor import Executor, ExecutionResult, Evidence
from .reviewer import review, ReviewResult
from .humanizer import humanize_simple, humanize_complex
from .graph import run_graph

__all__ = [
    # Legacy contracts
    "Artifact",
    "NodeContext",
    "NodeRegistry",
    "NodeResult",
    "OrchestrationNode",
    "TraceStep",
    # New flow nodes
    "triage",
    "TriageResult",
    "generate_plan",
    "ExecutionPlan",
    "PlanStep",
    "Executor",
    "ExecutionResult",
    "Evidence",
    "review",
    "ReviewResult",
    "humanize_simple",
    "humanize_complex",
    "run_graph",
]
