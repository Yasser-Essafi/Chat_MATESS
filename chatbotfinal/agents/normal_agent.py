"""
STATOUR Normal Agent — Fixed
==============================
General-purpose conversational agent for the STATOUR platform.
Handles greetings, help, platform questions, and general conversation.

Fixes from original:
- Uses BaseAgent (eliminates duplicated code)
- History trimming (was completely missing — caused context overflow)
- Thread-safe chat via lock
- Empty response handling with user message rollback
- Structured logging
- Proper error handling

Usage:
    from agents.normal_agent import NormalAgent

    agent = NormalAgent()
    response = agent.chat("Bonjour, qu'est-ce que STATOUR?")
    agent.reset_conversation()
"""

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    SYSTEM_PROMPTS,
    AGENT_TEMPERATURE,
    AGENT_MAX_COMPLETION_TOKENS,
    NORMAL_REASONING_EFFORT,
)
from utils.base_agent import BaseAgent, detect_language, sanitize_input
from utils.logger import get_logger

logger = get_logger("statour.normal")


# ══════════════════════════════════════════════════════════════════════════════
# Greeting / Farewell Detection (handled at agent level for fast response)
# ══════════════════════════════════════════════════════════════════════════════

# Greetings split by language — used to pick the correct response WITHOUT calling detect_language,
# because detect_language("Salut") → "en" (greetings are not in the French grammar-word markers).
_FR_GREETINGS = frozenset({
    "bonjour", "bonsoir", "salut", "salam", "coucou",
    "bjr", "bsr", "yo", "hola",  # colloquial / ambiguous → default FR for STATOUR context
})
_EN_GREETINGS = frozenset({
    "hello", "hi", "hey", "good morning", "good evening",
})
_AR_GREETINGS = frozenset({
    "مرحبا", "السلام عليكم", "سلام", "اهلا",
})
_GREETINGS = _FR_GREETINGS | _EN_GREETINGS | _AR_GREETINGS

# Exact-match farewells (lowercase)
_FAREWELLS = frozenset({
    "bye", "goodbye", "au revoir", "ciao", "adieu", "bbye",
    "à bientôt", "a bientot", "bslama", "مع السلامة",
})

# Thank-you messages (lowercase)
_THANKS = frozenset({
    "merci", "thanks", "thank you", "thx", "ty",
    "merci beaucoup", "thank you very much",
    "شكرا", "شكرا جزيلا",
})

# Greeting responses by language
_GREETING_RESPONSES = {
    "fr": (
        "Bonjour ! 👋 Je suis l'Assistant Général de STATOUR, "
        "la plateforme statistique du Ministère du Tourisme, "
        "de l'Artisanat et de l'Économie Sociale du Maroc.\n\n"
        "Je peux vous aider avec :\n"
        "• 📊 Des questions sur les données touristiques\n"
        "• 🔍 Des recherches sur l'actualité du tourisme\n"
        "• ℹ️ Des explications sur la plateforme STATOUR\n\n"
        "Comment puis-je vous aider ?"
    ),
    "en": (
        "Hello! 👋 I'm the General Assistant of STATOUR, "
        "the statistical platform of Morocco's Ministry of Tourism.\n\n"
        "I can help you with:\n"
        "• 📊 Tourism data questions\n"
        "• 🔍 Tourism news and research\n"
        "• ℹ️ STATOUR platform explanations\n\n"
        "How can I help you?"
    ),
    "ar": (
        "مرحبا! 👋 أنا المساعد العام لمنصة ستاتور، "
        "المنصة الإحصائية لوزارة السياحة المغربية.\n\n"
        "يمكنني مساعدتك في:\n"
        "• 📊 أسئلة حول بيانات السياحة\n"
        "• 🔍 أخبار وأبحاث السياحة\n"
        "• ℹ️ شرح منصة ستاتور\n\n"
        "كيف يمكنني مساعدتك؟"
    ),
}

_FAREWELL_RESPONSES = {
    "fr": (
        "Au revoir ! 👋 N'hésitez pas à revenir si vous avez "
        "d'autres questions sur le tourisme marocain. Bonne journée ! 🇲🇦"
    ),
    "en": (
        "Goodbye! 👋 Feel free to come back if you have "
        "more questions about Moroccan tourism. Have a great day! 🇲🇦"
    ),
    "ar": (
        "مع السلامة! 👋 لا تتردد في العودة إذا كان لديك "
        "أسئلة أخرى حول السياحة المغربية. يوم سعيد! 🇲🇦"
    ),
}

_THANKS_RESPONSES = {
    "fr": (
        "Je vous en prie ! 😊 N'hésitez pas si vous avez "
        "d'autres questions. Je suis là pour vous aider."
    ),
    "en": (
        "You're welcome! 😊 Don't hesitate if you have "
        "more questions. I'm here to help."
    ),
    "ar": (
        "على الرحب والسعة! 😊 لا تتردد إذا كان لديك "
        "أسئلة أخرى. أنا هنا لمساعدتك."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Normal Agent
# ══════════════════════════════════════════════════════════════════════════════

class NormalAgent(BaseAgent):
    """
    General-purpose conversational agent for the STATOUR platform.

    Handles:
        - Greetings, farewells, thank-you messages (instant, no LLM call)
        - General questions about STATOUR
        - Tourism terminology and concepts
        - Platform navigation help
        - Routing suggestions to specialized agents

    Inherits from BaseAgent:
        - Shared AzureOpenAI client
        - History trimming
        - Thread-safe LLM calls
        - Reset/length methods
    """

    def __init__(self):
        super().__init__(
            system_prompt=SYSTEM_PROMPTS["normal"],
            agent_name="Assistant Général STATOUR",
            temperature=AGENT_TEMPERATURE,
            max_tokens=AGENT_MAX_COMPLETION_TOKENS,
            reasoning_effort=NORMAL_REASONING_EFFORT,
        )
        logger.info("%s ready", self.agent_name)

    # ──────────────────────────────────────────────────────────────────────
    # Quick Response Detection (no LLM needed)
    # ──────────────────────────────────────────────────────────────────────

    def _get_quick_response(self, message: str) -> Optional[str]:
        """
        Check if the message is a simple greeting/farewell/thanks
        that can be answered instantly without calling the LLM.

        Args:
            message: User message (raw, not sanitized).

        Returns:
            Quick response string, or None if LLM is needed.
        """
        msg = message.lower().strip()

        # ── Determine language for greeting/farewell/thanks ──
        # We do NOT use detect_language() here because greetings like "Salut",
        # "Bonjour", "Salam" are not in the grammar-word markers → would return "en".
        # Instead, look up the word in language-specific sets.
        def _greeting_lang(word: str) -> str:
            if word in _AR_GREETINGS:
                return "ar"
            if word in _EN_GREETINGS:
                return "en"
            return "fr"  # Default: French (STATOUR primary language)

        # Exact match greetings
        if msg in _GREETINGS:
            return _GREETING_RESPONSES[_greeting_lang(msg)]

        # Exact match farewells
        if msg in _FAREWELLS:
            lang = detect_language(message)
            return _FAREWELL_RESPONSES.get(lang, _FAREWELL_RESPONSES["fr"])

        # Exact match thanks
        if msg in _THANKS:
            lang = detect_language(message)
            return _THANKS_RESPONSES.get(lang, _THANKS_RESPONSES["fr"])

        # Greeting with extra words (e.g., "bonjour comment ça va")
        words = msg.split()
        if len(words) <= 5 and words[0] in _GREETINGS:
            return _GREETING_RESPONSES[_greeting_lang(words[0])]

        return None

    # ──────────────────────────────────────────────────────────────────────
    # Main Chat (thread-safe)
    # ──────────────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return a response.

        Workflow:
            1. Check for quick responses (greetings, farewells, thanks)
            2. Sanitize input
            3. Trim conversation history
            4. Call LLM
            5. Handle empty responses gracefully

        Args:
            user_message: The user's input text.

        Returns:
            Agent response string. Never raises — returns error message on failure.
        """
        with self._lock:
            return self._chat_internal(user_message)

    def _chat_internal(self, user_message: str) -> str:
        """Internal chat logic (must be called with self._lock held)."""

        # ── Step 1: Quick response (no LLM needed) ──
        quick = self._get_quick_response(user_message)
        if quick:
            self._append_exchange(user_message, quick)
            logger.debug("Quick response for: %s", user_message[:50])
            return quick

        # ── Step 2: Sanitize input ──
        sanitized = sanitize_input(user_message)
        if not sanitized:
            error_msg = "⚠️ Message invalide. Veuillez reformuler votre question."
            logger.warning("Empty message after sanitization: %s", user_message[:50])
            return error_msg

        # ── Step 3: Trim history ──
        self._trim_history()

        # ── Step 4: Add user message and call LLM ──
        self.conversation_history.append({
            "role": "user",
            "content": sanitized,
        })

        result = self._call_llm()

        # ── Step 5: Handle response ──
        if result:
            self.conversation_history.append({
                "role": "assistant",
                "content": result,
            })
            logger.debug(
                "Response generated (%d chars) for: %s",
                len(result), sanitized[:50],
            )
            return result

        # ── Step 6: Empty response — rollback and retry once ──
        logger.warning(
            "Empty LLM response for: %s — retrying...",
            sanitized[:50],
        )

        # Retry with a simpler prompt
        result = self._call_llm_with_retry(max_retries=1)

        if result:
            self.conversation_history.append({
                "role": "assistant",
                "content": result,
            })
            return result

        # ── Step 7: Complete failure — rollback user message ──
        self._pop_last_user_message()
        logger.error("Failed to generate response for: %s", sanitized[:80])

        lang = detect_language(user_message)
        error_messages = {
            "fr": "⚠️ Je n'ai pas pu générer une réponse. Veuillez reformuler votre question.",
            "en": "⚠️ I couldn't generate a response. Please rephrase your question.",
            "ar": "⚠️ لم أتمكن من إنشاء إجابة. يرجى إعادة صياغة سؤالك.",
        }
        return error_messages.get(lang, error_messages["fr"])


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Interactive CLI for testing the Normal Agent."""
    # Validate configuration
    try:
        from config.settings import validate_config
        validate_config(require_tavily=False)
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return

    print("=" * 60)
    print("🏛️  STATOUR — Assistant Général")
    print("   Ministère du Tourisme du Maroc")
    print("=" * 60)
    print()
    print("Commands: /reset  /stats  /quit")
    print("=" * 60)
    print()

    agent = NormalAgent()

    while True:
        try:
            user_input = input("👤 Vous: ").strip()
            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd == "/quit":
                print("\n👋 Au revoir!")
                break

            if cmd == "/reset":
                print(f"\n{agent.reset_conversation()}\n")
                continue

            if cmd == "/stats":
                stats = agent.get_history_stats()
                print(f"\n📊 Stats:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                print()
                continue

            response = agent.chat(user_input)
            print(f"\n🤖 {agent.agent_name}:\n{response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"\n❌ Erreur inattendue: {e}\n")
            logger.error("CLI error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()