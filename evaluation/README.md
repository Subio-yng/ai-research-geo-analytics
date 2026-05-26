# Regression benchmark for the geo agent

Бенчмарк гоняет одного агента на одной модели (`qwen3:14b` через Ollama), но
позволяет **сравнивать разные конфигурации**: промпт, параметры сэмплирования,
`num_ctx`, версии тулов. Каждый прогон помечается меткой `--label`.

## Quick start

1. Запустить Ollama и поставить модель:

```bash
   ollama pull qwen3:14b
   ollama serve
```

2. Прогнать baseline:

```bash
   python -m evaluation.evaluate --label baseline --n-repeats 3
```

   60 запросов × 3 повтора ≈ **30–60 минут на M4 24GB** для 14B-модели.
   Если запросы с followup есть — общее число runs больше 180.

3. Внести изменение (промпт, `num_ctx`, температура, набор тулов) и прогнать ещё раз
   с другой меткой:

```bash
   python -m evaluation.evaluate --label short_prompt --n-repeats 3
```

4. Сравнить:

```bash
   python -m evaluation.evaluate --analyze
```

   Получаешь `logs/comparison.csv` и `logs/comparison.md` со всеми label-ами
   рядом в одной таблице.

## Быстрая проверка (smoke test)

Перед длинным прогоном проверь, что pipeline вообще работает:

```bash
python -m evaluation.evaluate --label smoke --n-repeats 1 --queries q01 q02 q03
```

## Что меряется

- **Strict tool-seq match** — последовательность тулов точно равна expected.
- **Soft tool-set match** — все expected тулы вызваны (порядок не важен).
- **Args subset match** — ключевые поля аргументов совпали с expected.
- **Args valid rate** — доля tool calls, где аргументы прошли валидацию Pydantic-схемы.
- **Calls per run** — сколько тулов агент вызвал. Меньше = лучше, при равной точности.
- **Latency** — `total` (включая выполнение тулов) и `LLM` (только chat.completions).
- **Tokens** — prompt/completion (если бэкенд отдаёт `usage`).
- **Terminated by** — `answer` / `max_iterations` / `exception` / `offline`.

## Структура логов

`logs/runs.jsonl` — одна строка на каждый вызов `agent.run()`:

```json
{
  "run_id": "abc123...",
  "label": "baseline",
  "model": "qwen3:14b",
  "chat_id": "bench-baseline-q01-rep0-xxx",
  "user_query": "Найди кафе рядом с площадью 1905 года",
  "final_answer": "Топ-3 кафе: ...",
  "tool_sequence": ["geocode", "nearest_places"],
  "n_tool_calls": 2,
  "total_latency_ms": 8420.3,
  "llm_latency_ms": 6210.1,
  "prompt_tokens": 1845,
  "completion_tokens": 312,
  "terminated_by": "answer",
  "exception": null
}
```

`logs/tool_calls.jsonl` — одна строка на каждый tool call (связь по `run_id`):

```json
{
  "run_id": "abc123...",
  "label": "baseline",
  "model": "qwen3:14b",
  "tool_name": "geocode",
  "tool_arguments": {"location": "площадь 1905 года", "city_hint": "Екатеринбург"},
  "arguments_valid": true,
  "result_summary": {"status": "ok", "type": "dict", "keys": ["lat", "lon", "address"]}
}
```

## Бенчмарк (60 запросов)

Категории и количество запросов:

- `geocode_then_search` — geocode → nearest_places
- `opportunity_implant` — opportunity_grid со стратегией implant
- `opportunity_aggregate` — opportunity_grid со стратегией aggregate
- `name_lookup` — search_by_name по бренду
- `nearby_simple`, `broad_search`, `distance`, `geocode_only`
- `ranking`, `filtering`, `hex_analysis`, `complex_chain`
- `category_normalization`, `heatmap`
- `clarification_required` / `ambiguous` — ожидается уточняющий вопрос, `expected_tools=[]`
- `out_of_scope` — не геозапрос или prompt-injection, `expected_tools=[]`

## Перед первым прогоном

1. **Проверь `hex_id` в q41/q42** — они должны существовать в твоей сетке.
   Один раз запусти `compute_opportunity_grid(category="pharmacy")` и подставь
   реальные id из результата.
2. **Проверь геокодер вручную** — что «площадь 1905 года», «Плотинка» и др.
   реально находятся.
3. **Очисти старые логи**, если хочешь начать с чистого листа:

```bash
   rm logs/*.jsonl
```

   Иначе `--analyze` смешает старые и новые прогоны (но это нормально — разные
   labels всё равно разделены).

## Что НЕ нужно делать

- **Не меняй `LLM_MODEL` между прогонами** — бенчмарк рассчитан на сравнение
  конфигураций одной модели. Хочешь сравнить с другой моделью — это уже не
  regression-тест, а model selection, для которого нужна отдельная отчётность.
- **Не запускай два прогона параллельно** — оба будут писать в один JSONL, и
  записи могут перемешаться (хоть append и атомарен построчно).
