"""Simple conversation memory with Database for the agent."""
from __future__ import annotations

from .db import load_chat_history, save_chat_history


class ConversationMemory:
    """Хранит историю диалога и контекст между шагами."""

    def __init__(self, system_prompt: str, max_turns: int = 10):
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.history: list[dict] = [{"role": "system", "content": system_prompt}]

    def add_user_message(self, content: str):
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str | None, tool_calls: list | None = None):
        """Добавить сообщение ассистента.

        Если есть tool_calls, content может быть None (OpenAI API так и требует —
        пустая строка вместе с tool_calls иногда вызывает валидационные ошибки).
        Если tool_calls нет — content приводится к строке (None → "").
        """
        if tool_calls:
            msg: dict = {
                "role": "assistant",
                "content": content if content else None,
                "tool_calls": [
                    tc.model_dump() if hasattr(tc, "model_dump") else tc
                    for tc in tool_calls
                ],
            }
        else:
            msg = {"role": "assistant", "content": content or ""}
        self.history.append(msg)

    def get_display_history(self) -> list[dict]:
        """User + assistant text messages only — for chat UI rendering."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.history
            if m["role"] in ("user", "assistant") and m.get("content")
        ]

    def add_tool_result(self, tool_call_id: str, content: str):
        self.history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        })

    def get_messages(self) -> list[dict]:
        """Возвращает сообщения для отправки в LLM, обрезая старые, если нужно."""
        # Оставляем system + последние (max_turns * 2) сообщений (пользователь+ассистент)
        if len(self.history) <= self.max_turns * 2 + 1:
            return self.history.copy()
        # Сохраняем system + последние сообщения
        return [self.history[0]] + self.history[-(self.max_turns * 2):]

    def clear(self):
        """Очистить память, оставив только system prompt."""
        self.history = [{"role": "system", "content": self.system_prompt}]


class PersistedMemory(ConversationMemory):
    """Расширенная память с автосохранением в SQLite."""
    def __init__(self, chat_id: str, system_prompt: str, max_turns: int = 10):
        super().__init__(system_prompt, max_turns)
        self.chat_id = chat_id
        self._load_from_db()

    def _load_from_db(self):
        db_history = load_chat_history(self.chat_id)
        if not db_history:
            return

        self.history = db_history

        # Гарантируем, что system prompt всегда первый
        if not self.history or self.history[0].get("role") != "system":
            self.history.insert(0, {"role": "system", "content": self.system_prompt})

        # Удаляем висящие assistant tool_call-сообщения с конца истории.
        while (
            len(self.history) > 1
            and self.history[-1].get("role") == "assistant"
            and self.history[-1].get("tool_calls")
        ):
            self.history.pop()

    def save(self):
        """Сохраняет историю в БД — без результатов tool-вызовов (они могут быть огромными)."""
        without_results = [m for m in self.history if m.get("role") != "tool"]
        save_chat_history(self.chat_id, without_results)
