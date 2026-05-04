"""
STATOUR Session & History Management
"""

import os
import sys
import json
import uuid
import tempfile
from datetime import datetime
from typing import Optional, List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import get_logger

logger = get_logger("statour.ui.session")

# History persistence directory
HISTORY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "chat_history"
)
os.makedirs(HISTORY_DIR, exist_ok=True)


class Message:
    """A single chat message."""

    def __init__(
        self,
        role: str,
        content: str,
        agent: Optional[str] = None,
        agent_icon: Optional[str] = None,
        agent_name: Optional[str] = None,
        chart_path: Optional[str] = None,
        rerouted: bool = False,
        classification_time_ms: float = 0,
        total_time_ms: float = 0,
        timestamp: Optional[str] = None,
    ):
        self.role = role
        self.content = content
        self.agent = agent
        self.agent_icon = agent_icon
        self.agent_name = agent_name
        self.chart_path = chart_path
        self.rerouted = rerouted
        self.classification_time_ms = classification_time_ms
        self.total_time_ms = total_time_ms
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        # Derive a relative chart_url from the absolute chart_path so the
        # frontend can reload charts when revisiting old conversations.
        chart_url = (
            "/charts/" + os.path.basename(self.chart_path)
            if self.chart_path and os.path.exists(self.chart_path)
            else None
        )
        return {
            "role": self.role,
            "content": self.content,
            "agent": self.agent,
            "agent_icon": self.agent_icon,
            "agent_name": self.agent_name,
            "chart_path": self.chart_path,
            "chart_url": chart_url,
            "rerouted": self.rerouted,
            "classification_time_ms": self.classification_time_ms,
            "total_time_ms": self.total_time_ms,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        # chart_url is a computed field in to_dict(), not a constructor param — strip it
        d = {k: v for k, v in data.items() if k != "chart_url"}
        return cls(**d)


class Conversation:
    """A single conversation (list of messages with metadata)."""

    def __init__(
        self,
        conversation_id: Optional[str] = None,
        title: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        self.id = conversation_id or str(uuid.uuid4())[:8]
        self.title = title or "Nouvelle conversation"
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = self.created_at
        self.messages: List[Message] = []

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()

        # Auto-title from first user message
        if self.title == "Nouvelle conversation" and message.role == "user":
            title = message.content[:60]
            if len(message.content) > 60:
                title += "..."
            self.title = title

    def get_last_agent(self) -> Optional[str]:
        for msg in reversed(self.messages):
            if msg.role == "assistant" and msg.agent:
                return msg.agent
        return None

    def get_charts(self) -> List[str]:
        charts = []
        for msg in self.messages:
            if msg.chart_path and os.path.exists(msg.chart_path):
                charts.append(msg.chart_path)
        return charts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        conv = cls(
            conversation_id=data.get("id"),
            title=data.get("title"),
            created_at=data.get("created_at"),
        )
        conv.updated_at = data.get("updated_at", conv.created_at)
        conv.messages = [
            Message.from_dict(m) for m in data.get("messages", [])
        ]
        return conv


class SessionManager:
    """
    Manages multiple conversations with persistence.
    Saves/loads conversations as JSON files.
    """

    def __init__(self):
        self.conversations: Dict[str, Conversation] = {}
        self.active_conversation_id: Optional[str] = None
        self._load_all_conversations()

    # ──────────────────────────────────────────────────────────────────
    # Active Conversation
    # ──────────────────────────────────────────────────────────────────

    def get_active_conversation(self) -> Optional[Conversation]:
        if self.active_conversation_id:
            return self.conversations.get(self.active_conversation_id)
        return None

    def new_conversation(self) -> Conversation:
        conv = Conversation()
        self.conversations[conv.id] = conv
        self.active_conversation_id = conv.id
        self._save_conversation(conv)
        logger.info("New conversation created: %s", conv.id)
        return conv

    def switch_conversation(self, conversation_id: str) -> Optional[Conversation]:
        if conversation_id in self.conversations:
            self.active_conversation_id = conversation_id
            return self.conversations[conversation_id]
        return None

    def delete_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

            # Delete file
            filepath = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    logger.warning("Failed to delete history file: %s", e)

            # Reset active if needed
            if self.active_conversation_id == conversation_id:
                if self.conversations:
                    self.active_conversation_id = list(self.conversations.keys())[-1]
                else:
                    self.active_conversation_id = None

            return True
        return False

    def add_message_to_active(self, message: Message) -> None:
        conv = self.get_active_conversation()
        if conv is None:
            conv = self.new_conversation()
        conv.add_message(message)
        self._save_conversation(conv)

    # ──────────────────────────────────────────────────────────────────
    # Conversation List (sorted by most recent)
    # ──────────────────────────────────────────────────────────────────

    def get_conversation_list(self) -> List[Dict[str, Any]]:
        convs = []
        for conv in self.conversations.values():
            last_agent = conv.get_last_agent()
            convs.append({
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "message_count": len(conv.messages),
                "last_agent": last_agent,
            })
        convs.sort(key=lambda x: x["updated_at"], reverse=True)
        return convs

    # ──────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────

    def _save_conversation(self, conv: Conversation) -> None:
        """Save conversation with atomic write to prevent corruption."""
        filepath = os.path.join(HISTORY_DIR, f"{conv.id}.json")
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(dir=HISTORY_DIR, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(conv.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, filepath)
        except Exception as e:
            logger.error("Failed to save conversation %s: %s", conv.id, e)
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _load_all_conversations(self) -> None:
        """Load all conversation JSON files from HISTORY_DIR."""
        if not os.path.isdir(HISTORY_DIR):
            return

        json_files = sorted(
            [f for f in os.listdir(HISTORY_DIR) if f.endswith(".json")],
            key=lambda f: os.path.getmtime(os.path.join(HISTORY_DIR, f)),
            reverse=True,
        )

        for filename in json_files:
            filepath = os.path.join(HISTORY_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                conv = Conversation.from_dict(data)
                self.conversations[conv.id] = conv
            except Exception as e:
                logger.warning("Failed to load %s: %s", filename, e)

        if self.conversations:
            # Set most recent as active
            most_recent = max(
                self.conversations.values(),
                key=lambda c: c.updated_at,
            )
            self.active_conversation_id = most_recent.id
            logger.info(
                "Loaded %d conversations from history",
                len(self.conversations),
            )

    def clear_all_history(self) -> int:
        count = len(self.conversations)
        self.conversations.clear()
        self.active_conversation_id = None

        for f in os.listdir(HISTORY_DIR):
            if f.endswith(".json"):
                try:
                    os.remove(os.path.join(HISTORY_DIR, f))
                except Exception:
                    pass

        return count