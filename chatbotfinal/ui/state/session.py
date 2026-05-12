"""
STATOUR Session & History Management
"""

import os
import sys
import json
import uuid
import tempfile
import time
import threading
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
        message_id: Optional[str] = None,
        agent: Optional[str] = None,
        agent_icon: Optional[str] = None,
        agent_name: Optional[str] = None,
        chart_path: Optional[str] = None,
        chart_paths: Optional[List[str]] = None,
        rerouted: bool = False,
        classification_time_ms: float = 0,
        total_time_ms: float = 0,
        sources: Optional[List[Dict[str, Any]]] = None,
        confidence: Optional[str] = None,
        data_freshness: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        status: str = "done",
        trace: Optional[List[Dict[str, Any]]] = None,
        fallbacks: Optional[List[Dict[str, Any]]] = None,
        errors: Optional[List[Dict[str, Any]]] = None,
        error: Optional[Any] = None,
        timestamp: Optional[str] = None,
    ):
        paths: List[str] = []
        for path in chart_paths or []:
            if path and path not in paths:
                paths.append(path)
        if chart_path and chart_path not in paths:
            paths.insert(0, chart_path)

        self.id = message_id or str(uuid.uuid4())
        self.message_id = self.id
        self.role = role
        self.content = content
        self.agent = agent
        self.agent_icon = agent_icon
        self.agent_name = agent_name
        self.chart_paths = paths
        self.chart_path = paths[0] if paths else None
        self.rerouted = rerouted
        self.classification_time_ms = classification_time_ms
        self.total_time_ms = total_time_ms
        self.sources = sources or []
        self.confidence = confidence
        self.data_freshness = data_freshness or {}
        self.run_id = run_id
        self.status = status or "done"
        self.trace = trace or []
        self.fallbacks = fallbacks or []
        self.errors = errors or []
        self.error = error
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        # Derive a relative chart_url from the absolute chart_path so the
        # frontend can reload charts when revisiting old conversations.
        chart_urls = [
            "/charts/" + os.path.basename(path)
            for path in self.chart_paths
            if path and os.path.exists(path)
        ]
        chart_url = chart_urls[0] if chart_urls else None
        return {
            "id": self.id,
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "agent": self.agent,
            "agent_icon": self.agent_icon,
            "agent_name": self.agent_name,
            "chart_path": self.chart_path,
            "chart_paths": self.chart_paths,
            "chart_url": chart_url,
            "chart_urls": chart_urls,
            "rerouted": self.rerouted,
            "classification_time_ms": self.classification_time_ms,
            "total_time_ms": self.total_time_ms,
            "sources": self.sources,
            "confidence": self.confidence,
            "data_freshness": self.data_freshness,
            "run_id": self.run_id,
            "status": self.status,
            "trace": self.trace,
            "fallbacks": self.fallbacks,
            "errors": self.errors,
            "error": self.error,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        # chart_url is a computed field in to_dict(), not a constructor param — strip it
        d = {k: v for k, v in data.items() if k not in {"chart_url", "chart_urls", "id"}}
        if "message_id" not in d and data.get("id"):
            d["message_id"] = data.get("id")
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
            self._retitle_from_first_user()

    def _retitle_from_first_user(self) -> None:
        for msg in self.messages:
            if msg.role == "user" and msg.content:
                title = msg.content[:60]
                if len(msg.content) > 60:
                    title += "..."
                self.title = title
                return
        self.title = "Nouvelle conversation"

    def find_message_index(self, message_id: str) -> Optional[int]:
        for idx, msg in enumerate(self.messages):
            if msg.message_id == message_id or msg.id == message_id:
                return idx
        return None

    def get_message(self, message_id: str) -> Optional[Message]:
        idx = self.find_message_index(message_id)
        return self.messages[idx] if idx is not None else None

    def previous_user_index(self, start_index: int) -> Optional[int]:
        for idx in range(min(start_index, len(self.messages) - 1), -1, -1):
            if self.messages[idx].role == "user":
                return idx
        return None

    def related_user_for_message(self, message_id: str) -> Optional[Message]:
        idx = self.find_message_index(message_id)
        if idx is None:
            return None
        msg = self.messages[idx]
        if msg.role == "user":
            return msg
        user_idx = self.previous_user_index(idx)
        return self.messages[user_idx] if user_idx is not None else None

    def truncate_after_message(self, message_id: str) -> Optional[Message]:
        idx = self.find_message_index(message_id)
        if idx is None:
            return None
        kept = self.messages[idx]
        self.messages = self.messages[: idx + 1]
        self.updated_at = datetime.now().isoformat()
        self._retitle_from_first_user()
        return kept

    def update_user_message(self, message_id: str, content: str) -> Optional[Message]:
        msg = self.get_message(message_id)
        if msg is None or msg.role != "user":
            return None
        msg.content = content
        msg.timestamp = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self._retitle_from_first_user()
        return msg

    def get_last_agent(self) -> Optional[str]:
        for msg in reversed(self.messages):
            if msg.role == "assistant" and msg.agent:
                return msg.agent
        return None

    def get_charts(self) -> List[str]:
        charts = []
        for msg in self.messages:
            for path in msg.chart_paths:
                if path and os.path.exists(path):
                    charts.append(path)
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
        self._lock = threading.RLock()
        self._load_all_conversations()

    # ──────────────────────────────────────────────────────────────────
    # Active Conversation
    # ──────────────────────────────────────────────────────────────────

    def get_active_conversation(self) -> Optional[Conversation]:
        if self.active_conversation_id:
            return self.conversations.get(self.active_conversation_id)
        return None

    def new_conversation(self, activate: bool = True) -> Conversation:
        with self._lock:
            conv = Conversation()
            self.conversations[conv.id] = conv
            if activate:
                self.active_conversation_id = conv.id
            self._save_conversation(conv)
            logger.info("New conversation created: %s", conv.id)
            return conv

    def ensure_conversation(
        self,
        conversation_id: Optional[str] = None,
        activate: bool = False,
    ) -> Conversation:
        """Return an existing conversation or create the requested one.

        API calls should pass activate=False so stateless requests do not
        mutate the UI-global active conversation pointer.
        """
        with self._lock:
            if conversation_id:
                conv = self.conversations.get(conversation_id)
                if conv is None:
                    conv = Conversation(conversation_id=conversation_id)
                    self.conversations[conv.id] = conv
                    self._save_conversation(conv)
                    logger.info("Conversation created from client id: %s", conv.id)
                if activate:
                    self.active_conversation_id = conv.id
                return conv
            return self.new_conversation(activate=activate)

    def switch_conversation(self, conversation_id: str) -> Optional[Conversation]:
        with self._lock:
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
        with self._lock:
            conv = self.get_active_conversation()
            if conv is None:
                conv = self.new_conversation()
            conv.add_message(message)
            self._save_conversation(conv)

    def add_message_to_conversation(self, conversation_id: str, message: Message) -> Conversation:
        """Append a message to a specific conversation without switching active UI state."""
        with self._lock:
            conv = self.ensure_conversation(conversation_id, activate=False)
            conv.add_message(message)
            self._save_conversation(conv)
            return conv

    def save_conversation(self, conv: Conversation) -> None:
        self._save_conversation(conv)

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        return self.conversations.get(conversation_id)

    def fork_from_message(self, conversation_id: str, message_id: str) -> Optional[Message]:
        conv = self.get_conversation(conversation_id)
        if not conv:
            return None
        user_msg = conv.related_user_for_message(message_id)
        if not user_msg:
            return None
        kept = conv.truncate_after_message(user_msg.message_id)
        self.active_conversation_id = conversation_id
        self._save_conversation(conv)
        return kept

    def edit_user_message(
        self,
        conversation_id: str,
        message_id: str,
        content: str,
    ) -> Optional[Message]:
        conv = self.get_conversation(conversation_id)
        if not conv:
            return None
        user_msg = conv.related_user_for_message(message_id)
        if not user_msg:
            return None
        edited = conv.update_user_message(user_msg.message_id, content)
        if not edited:
            return None
        conv.truncate_after_message(edited.message_id)
        self.active_conversation_id = conversation_id
        self._save_conversation(conv)
        return edited

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
        with self._lock:
            filepath = os.path.join(HISTORY_DIR, f"{conv.id}.json")
            tmp_path = None
            payload = ""
            try:
                payload = json.dumps(conv.to_dict(), ensure_ascii=False, indent=2)
                fd, tmp_path = tempfile.mkstemp(dir=HISTORY_DIR, suffix=".tmp")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(payload)
                for attempt in range(3):
                    try:
                        os.replace(tmp_path, filepath)
                        tmp_path = None
                        return
                    except PermissionError:
                        if attempt == 2:
                            raise
                        time.sleep(0.2 * (attempt + 1))
            except Exception as e:
                try:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(payload)
                    logger.warning(
                        "Saved conversation %s with direct-write fallback after atomic save failed: %s",
                        conv.id,
                        e,
                    )
                except Exception as fallback_error:
                    logger.error(
                        "Failed to save conversation %s: %s; fallback also failed: %s",
                        conv.id,
                        e,
                        fallback_error,
                    )
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
