"""Structured JSONL logging for agent runs and tool calls.

Активируется переменной окружения `GEO_LOG_RUNS=1`. Без неё работает как no-op
и не создаёт никаких файлов. Метка прогона задаётся через `RUN_LABEL` —
используется для regression-сравнения разных конфигураций (промпт, num_ctx,
параметры сэмплирования и т.п.) на одной и той же модели.

Логи:
    logs/runs.jsonl       — одна строка на вызов agent.run()
    logs/tool_calls.jsonl — одна строка на каждый tool call

Связь по `run_id`. Директорию можно переопределить через `GEO_LOG_DIR`.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import ValidationError

# Pydantic-схемы тулов — нужны для валидации аргументов
from lib.data_types.agent_tools_schema import (
    DistanceRequest,
    FilterRequest,
    GeocodeRequest,
    HeatmapRequest,
    NearestHexesRequest,
    NearestPlacesRequest,
    OpportunityGridRequest,
    RankPlacesRequest,
    SearchByNameRequest,
    SearchPlacesRequest,
)

_TOOL_SCHEMAS = {
    "geocode": GeocodeRequest,
    "search_places": SearchPlacesRequest,
    "nearest_places": NearestPlacesRequest,
    "search_by_name": SearchByNameRequest,
    "rank_places": RankPlacesRequest,
    "nearest_hexes": NearestHexesRequest,
    "opportunity_grid": OpportunityGridRequest,
    "filter_places": FilterRequest,
    "compute_distance": DistanceRequest,
    "build_heatmap": HeatmapRequest,
}


def _enabled() -> bool:
    return os.getenv("GEO_LOG_RUNS") == "1"


def _log_dir() -> Path:
    p = Path(os.getenv("GEO_LOG_DIR", "logs"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _append_jsonl(path: Path, record: dict) -> None:
    """Atomic append одной строки JSON; пишем построчно, чтобы не терять данные
    при крэше посередине benchmark."""
    line = json.dumps(record, ensure_ascii=False, default=str)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _validate_tool_args(tool_name: str, args: dict) -> bool:
    """Прошли ли args валидацию Pydantic-схемы соответствующего тула."""
    schema = _TOOL_SCHEMAS.get(tool_name)
    if schema is None:
        return False
    try:
        schema(**args)
        return True
    except (ValidationError, TypeError):
        return False


def _summarize_result(result: Any) -> dict:
    """Безопасное саммари результата тула — без полного дампа в лог."""
    if isinstance(result, dict) and "error" in result:
        return {"status": "error", "error": str(result["error"])[:200]}
    if isinstance(result, list):
        return {"status": "ok", "type": "list", "len": len(result)}
    if isinstance(result, dict):
        return {"status": "ok", "type": "dict", "keys": list(result.keys())[:10]}
    return {"status": "ok", "type": type(result).__name__}


class RunLogger:
    """Контекст-менеджер для логирования одного вызова agent.run().

    Если `GEO_LOG_RUNS != "1"` — все методы становятся no-op, файлы не создаются.

    Использование:
        with RunLogger(query=q, chat_id=cid, model=m) as logger:
            ...
            logger.log_llm_call(usage=resp.usage, latency_ms=dt)
            logger.log_tool_call(name, args, result)
            ...
            logger.set_final_answer(text, terminated_by="answer")
    """

    def __init__(self, query: str, chat_id: str, model: str):
        self.enabled = _enabled()
        self.run_id = uuid.uuid4().hex
        self.query = query
        self.chat_id = chat_id
        self.model = model
        self.label = os.getenv("RUN_LABEL", "default")

        self._t_start: float = 0.0
        self._llm_latency_ms: float = 0.0
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._tool_sequence: list[str] = []
        self._final_answer: str = ""
        self._terminated_by: str = "unknown"
        self._exception: str | None = None

    def __enter__(self) -> "RunLogger":
        self._t_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self.enabled:
            return
        if exc_type is not None:
            self._terminated_by = "exception"
            self._exception = f"{exc_type.__name__}: {exc_val}"
        total_ms = (time.perf_counter() - self._t_start) * 1000.0

        record = {
            "run_id": self.run_id,
            "label": self.label,
            "model": self.model,
            "chat_id": self.chat_id,
            "user_query": self.query,
            "final_answer": self._final_answer[:2000],
            "tool_sequence": self._tool_sequence,
            "n_tool_calls": len(self._tool_sequence),
            "total_latency_ms": round(total_ms, 1),
            "llm_latency_ms": round(self._llm_latency_ms, 1),
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "terminated_by": self._terminated_by,
            "exception": self._exception,
        }
        _append_jsonl(_log_dir() / "runs.jsonl", record)

    # --- API, вызываемое из agent.py ---

    def log_llm_call(self, *, usage: Any, latency_ms: float) -> None:
        """Накопить usage/latency одного chat.completions.create."""
        if not self.enabled:
            return
        self._llm_latency_ms += latency_ms
        if usage is not None:
            self._prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self._completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    def log_tool_call(self, name: str, args: dict, result: Any) -> None:
        """Записать один tool call в tool_calls.jsonl и обновить tool_sequence."""
        if not self.enabled:
            return
        self._tool_sequence.append(name)
        record = {
            "run_id": self.run_id,
            "label": self.label,
            "model": self.model,
            "tool_name": name,
            "tool_arguments": args,
            "arguments_valid": _validate_tool_args(name, args),
            "result_summary": _summarize_result(result),
        }
        _append_jsonl(_log_dir() / "tool_calls.jsonl", record)

    def set_final_answer(self, text: str, *, terminated_by: str) -> None:
        """Зафиксировать финальный ответ и причину завершения цикла.

        terminated_by ∈ {"answer", "max_iterations", "exception", "offline"}
        """
        if not self.enabled:
            return
        self._final_answer = text or ""
        self._terminated_by = terminated_by
