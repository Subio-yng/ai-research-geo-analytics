"""Minimal LLM agent: query -> choose tool -> execute tool -> answer.

Многошаговый цикл tool calling с памятью в SQLite. Совместим с OpenAI API
и локальными OpenAI-совместимыми бэкендами (Ollama / vLLM / LM Studio).

Запуск:
    python -m agent.agent "Find quiet cafe near metro"
    python -m agent.agent "Find areas with low pharmacy coverage"
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid

from dotenv import load_dotenv

from core_utils.coverage import compute_opportunity_grid
from core_utils.search import search_places

from .db import init_db, save_opportunity_grid
from .memory import PersistedMemory
from .prompts import SYSTEM_PROMPT
from .run_logger import RunLogger
from .tools import (
    _tool_opportunity_grid,
    _tool_nearest_hexes,
    _tool_distance,
    _tool_filtering,
    _tool_rank,
    _tool_search_places,
    _tool_nearest_places,
    _tool_search_by_name,
    _tool_build_heatmap,
    _tool_geocode,
)
from .tools_schema import TOOLS

load_dotenv()


TOOL_IMPL = {
    "geocode": _tool_geocode,
    "nearest_places": _tool_nearest_places,
    "nearest_hexes": _tool_nearest_hexes,
    "search_by_name": _tool_search_by_name,
    "search_places": _tool_search_places,
    "rank_places": _tool_rank,
    "opportunity_grid": _tool_opportunity_grid,
    "filter_places": _tool_filtering,
    "compute_distance": _tool_distance,
    "build_heatmap": _tool_build_heatmap,
}


def _offline_route(query: str) -> str:
    """Простая эвристика-роутинг для оффлайн-режима (без LLM)."""
    q = query.lower()
    if "underserved" in q or ("low" in q and "coverage" in q) or "lack" in q or "open" in q:
        cat = "pharmacy" if "pharmacy" in q else "cafe" if "cafe" in q else "pharmacy"
        cells = compute_opportunity_grid(category=cat, hex_resolution=8)
        return (
            f"[offline] Opportunity grid for '{cat}' ({len(cells)} hexes):\n"
            + json.dumps(cells[:5], indent=2, ensure_ascii=False, default=str)
        )

    category = None
    for c in ("cafe", "restaurant", "fastfood", "pharmacy", "bar"):
        if c in q:
            category = c
            break

    places = search_places(category=category, limit=5)
    return (
        f"[offline] First {len(places)} places (category={category}):\n"
        + json.dumps(places, indent=2, ensure_ascii=False, default=str)
    )


def _is_local_backend() -> bool:
    return bool(os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"))


def _llm_client_and_model():
    from openai import OpenAI

    base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or ("local" if base_url else None)
    model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "qwen3:14b"

    timeout = 180.0 if _is_local_backend() else 90.0
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout), model


def _llm_extra_params() -> dict:
    """Параметры под Qwen3 в Ollama. Для внешнего OpenAI extra_body игнорируется."""
    if not _is_local_backend():
        return {}
    return {
        "top_p": 0.8,
        "extra_body": {
            "top_k": 20,
            "repeat_penalty": 1.05,
            "options": {
                "num_ctx": 16384,
                "num_predict": 2048,
            },
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def _safe_parse_tool_args(raw: str | None) -> tuple[dict, str | None]:
    """Парсит JSON-аргументы тула; возвращает (args, error_message)."""
    if not raw:
        return {}, None
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}, f"expected JSON object, got {type(parsed).__name__}"
        return parsed, None
    except json.JSONDecodeError as e:
        return {}, f"invalid JSON arguments from LLM: {e.msg} (pos {e.pos})"


def run(query: str, chat_id: str) -> str:
    """Run agent with memory and multi-step tool calling."""
    init_db()

    memory = PersistedMemory(chat_id=chat_id, system_prompt=SYSTEM_PROMPT)
    memory.add_user_message(query)
    memory.save()

    # Оффлайн-режим
    if not (os.getenv("OPENAI_API_KEY") or _is_local_backend()):
        model_name = "offline-heuristic"
        with RunLogger(query=query, chat_id=chat_id, model=model_name) as logger:
            answer = _offline_route(query)
            logger.set_final_answer(answer, terminated_by="offline")
        return answer

    client, model = _llm_client_and_model()
    extra = _llm_extra_params()

    with RunLogger(query=query, chat_id=chat_id, model=model) as logger:
        max_iterations = 5
        for iteration in range(max_iterations):
            messages = memory.get_messages()

            t_llm = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=2048,
                temperature=0.3,
                **extra,
            )
            dt_ms = (time.perf_counter() - t_llm) * 1000.0
            logger.log_llm_call(usage=response.usage, latency_ms=dt_ms)

            msg = response.choices[0].message

            if not msg.tool_calls:
                answer = msg.content or ""
                memory.add_assistant_message(answer)
                memory.save()
                logger.set_final_answer(answer, terminated_by="answer")
                return answer

            memory.add_assistant_message(msg.content, msg.tool_calls)

            for call in msg.tool_calls:
                name = call.function.name
                args, parse_err = _safe_parse_tool_args(call.function.arguments)

                if parse_err:
                    result = {"error": parse_err, "hint": "Re-check tool argument JSON format."}
                    logger.log_tool_call(name, args, result)
                    memory.add_tool_result(call.id, json.dumps(result, ensure_ascii=False))
                    continue

                impl = TOOL_IMPL.get(name)
                if impl:
                    try:
                        result = impl(args)
                    except Exception as e:
                        result = {"error": f"tool '{name}' raised {type(e).__name__}: {e}"}
                else:
                    result = {"error": f"unknown tool '{name}'"}

                logger.log_tool_call(name, args, result)

                if name == "opportunity_grid" and isinstance(result, list):
                    save_opportunity_grid(
                        chat_id,
                        {"cells": [c.model_dump() for c in result], "args": dict(args)},
                    )
                    top = sorted(result, key=lambda c: c.opportunity_score, reverse=True)[:20]
                    summary = {
                        "total_cells": len(result),
                        "top_cells": [
                            c.model_dump(exclude={"boundary", "row", "col", "is_visible", "label"})
                            for c in top
                        ],
                        "note": "Full grid is in UI. Use nearest_hexes to explore specific cells.",
                    }
                    memory.add_tool_result(call.id, json.dumps(summary, ensure_ascii=False))
                else:
                    memory.add_tool_result(
                        call.id,
                        json.dumps(result, ensure_ascii=False, default=str),
                    )

            memory.save()

        # Лимит итераций исчерпан
        memory.add_user_message(
            "[System] Достигнут лимит шагов. Сформулируй финальный ответ "
            "на основе уже полученных данных от tools. Не вызывай новые tools."
        )
        t_llm = time.perf_counter()
        final = client.chat.completions.create(
            model=model,
            messages=memory.get_messages(),
            max_tokens=2048,
            temperature=0.3,
            **extra,
        )
        dt_ms = (time.perf_counter() - t_llm) * 1000.0
        logger.log_llm_call(usage=final.usage, latency_ms=dt_ms)

        final_text = final.choices[0].message.content or ""
        memory.add_assistant_message(final_text)
        memory.save()
        logger.set_final_answer(final_text, terminated_by="max_iterations")
        return final_text


def main() -> None:
    query = " ".join(sys.argv[1:])
    if not query:
        print("Usage: python -m agent.agent <query>")
        sys.exit(1)
    chat_id = os.getenv("CHAT_ID") or f"cli-{uuid.uuid4().hex[:8]}"
    print(run(query, chat_id=chat_id))


if __name__ == "__main__":
    main()
