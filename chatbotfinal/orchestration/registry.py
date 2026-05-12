"""Node registry used by the STATOUR orchestrator."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Protocol

from .contracts import NodeContext, NodeResult


class OrchestrationNode(Protocol):
    """Minimal interface implemented by all orchestration nodes."""

    key: str

    def can_handle(self, context: NodeContext) -> bool:
        ...

    def run(self, context: NodeContext) -> NodeResult:
        ...


class NodeRegistry:
    """Register, select, and run orchestration nodes by key."""

    def __init__(self, nodes: Optional[Iterable[OrchestrationNode]] = None):
        self._nodes: Dict[str, OrchestrationNode] = {}
        for node in nodes or []:
            self.register(node)

    def register(self, node: OrchestrationNode) -> OrchestrationNode:
        if not getattr(node, "key", None):
            raise ValueError("Node key is required")
        if node.key in self._nodes:
            raise ValueError(f"Node already registered: {node.key}")
        self._nodes[node.key] = node
        return node

    def get(self, key: str) -> OrchestrationNode:
        try:
            return self._nodes[key]
        except KeyError as exc:
            raise KeyError(f"Unknown orchestration node: {key}") from exc

    def keys(self) -> list[str]:
        return list(self._nodes.keys())

    def select(self, context: NodeContext, preferred_key: Optional[str] = None) -> OrchestrationNode:
        if preferred_key:
            return self.get(preferred_key)
        for node in self._nodes.values():
            if node.can_handle(context):
                return node
        raise LookupError("No orchestration node can handle this context")

    def run(self, context: NodeContext, preferred_key: Optional[str] = None) -> NodeResult:
        return self.select(context, preferred_key=preferred_key).run(context)
