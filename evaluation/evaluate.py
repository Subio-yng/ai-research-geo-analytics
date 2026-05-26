"""Regression benchmark for the geo agent on a single model.

Идея: модель одна (qwen3:14b), но мы хотим сравнивать разные конфигурации —
промпт, параметры сэмплирования, num_ctx, набор тулов. Каждый прогон помечается
меткой `--label`, которая пишется в каждую запись логов. После двух+ прогонов
с разными метками `--analyze` показывает таблицу сравнения.

Типичный dev-loop:
    # baseline
    python -m evaluation.evaluate --label baseline --n-repeats 3

    # меняешь prompts.py / параметры, прогоняешь ещё раз
    python -m evaluation.evaluate --label short_prompt --n-repeats 3

    # сравнение
    python -m evaluation.evaluate --analyze

Followup:
    Записи могут содержать поле "followup": {"user_reply": ...,
    "expected_tools": [...], "expected_args_subset": {...}}.
    После основного запроса бенчмарк дошлёт user_reply в том же chat_id —
    это проверка, корректно ли агент действует после уточнения.
    В отчёте followup-проходы агрегируются под категорией "<orig>_followup".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

QUERIES_PATH = Path(__file__).resolve().parent / "benchmark_queries.json"
LOG_DIR = Path(os.getenv("GEO_LOG_DIR", "logs"))


def _setup_env(model: str, label: str) -> None:
    """Set env-vars BEFORE importing agent (run_logger reads them on import)."""
    os.environ["GEO_LOG_RUNS"] = "1"
    os.environ["LLM_MODEL"] = model
    os.environ["RUN_LABEL"] = label
    os.environ["LLM_BASE_URL"] = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
    # Локальные сервера обычно не валидируют ключ, но клиент требует непустой.
    os.environ.setdefault("OPENAI_API_KEY", "ollama")


def run_benchmark(model: str, label: str, n_repeats: int,
                  query_filter: list[str] | None = None) -> None:
    """Прогнать бенчмарк под одной меткой."""
    _setup_env(model, label)

    # Импортируем агента ПОСЛЕ установки env-переменных
    from agent.agent import run as run_agent  # noqa: E402

    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    if query_filter:
        queries = [q for q in queries if q["id"] in query_filter]

    total_steps = sum(2 if q.get("followup") else 1 for q in queries) * n_repeats
    print(f"\n=== Benchmark: model={model} label={label} ===")
    print(f"Queries: {len(queries)}, repeats: {n_repeats}, total runs: {total_steps}")
    print(f"Logs append to: {LOG_DIR}/runs.jsonl and {LOG_DIR}/tool_calls.jsonl\n")

    t0 = time.perf_counter()
    step = 0
    for rep in range(n_repeats):
        for q in queries:
            chat_id = f"bench-{label}-{q['id']}-rep{rep}-{uuid.uuid4().hex[:6]}"

            # основной запрос
            step += 1
            print(f"  [{step}/{total_steps}] {q['id']} ({q['category']}): "
                  f"{q['query'][:60]}...", end=" ", flush=True)
            t_q = time.perf_counter()
            try:
                run_agent(q["query"], chat_id=chat_id)
                print(f"ok {time.perf_counter() - t_q:.1f}s")
            except Exception as e:
                print(f"ERROR: {type(e).__name__}: {e}")
                continue

            # followup
            fu = q.get("followup")
            if not fu:
                continue
            step += 1
            print(f"  [{step}/{total_steps}] {q['id']}_fu ({q['category']}_followup): "
                  f"{fu['user_reply'][:60]}...", end=" ", flush=True)
            t_q = time.perf_counter()
            try:
                run_agent(fu["user_reply"], chat_id=chat_id)
                print(f"ok {time.perf_counter() - t_q:.1f}s")
            except Exception as e:
                print(f"ERROR: {type(e).__name__}: {e}")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed / 60:.1f} min")


# --- Analysis ---


def _load_logs() -> tuple[list[dict], list[dict]]:
    runs_path = LOG_DIR / "runs.jsonl"
    tools_path = LOG_DIR / "tool_calls.jsonl"
    if not runs_path.exists():
        raise FileNotFoundError(f"{runs_path} not found. Run benchmark first.")

    runs = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    tools = [json.loads(line) for line in tools_path.read_text(encoding="utf-8").splitlines()
             if line.strip()] if tools_path.exists() else []
    return runs, tools


def _expected_lookup() -> dict[str, dict]:
    """user_query -> expected spec (включая followup-реплики)."""
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    lookup: dict[str, dict] = {}
    for q in queries:
        lookup[q["query"]] = q
        fu = q.get("followup")
        if fu:
            lookup[fu["user_reply"]] = {
                "id": q["id"] + "_fu",
                "category": q["category"] + "_followup",
                "query": fu["user_reply"],
                "expected_tools": fu["expected_tools"],
                "expected_args_subset": fu["expected_args_subset"],
            }
    return lookup


def _check_tool_match(actual_seq: list[str], expected_seq: list[str]) -> tuple[bool, bool]:
    """(strict, soft). strict: точное равенство. soft: expected ⊆ actual (и оба пустые)."""
    strict = actual_seq == expected_seq
    if not expected_seq:
        soft = len(actual_seq) == 0
    else:
        soft = set(expected_seq).issubset(set(actual_seq))
    return strict, soft


def _check_args_subset(actual_calls: list[dict], expected: dict) -> bool:
    """Все expected_args_subset присутствуют в каком-нибудь реальном вызове тула."""
    if not expected:
        return True
    for tool_name, expected_args in expected.items():
        matching = [c for c in actual_calls if c["tool_name"] == tool_name]
        if not matching:
            return False
        if not any(
            all(k in c.get("tool_arguments", {}) and c["tool_arguments"][k] == v
                for k, v in expected_args.items())
            for c in matching
        ):
            return False
    return True


def analyze() -> None:
    """Группируем по label, считаем метрики, сохраняем CSV/MD."""
    runs, tools = _load_logs()
    expected_by_query = _expected_lookup()

    tools_by_run: dict[str, list[dict]] = defaultdict(list)
    for tc in tools:
        tools_by_run[tc["run_id"]].append(tc)

    by_label: dict[str, dict] = defaultdict(lambda: {
        "model": "",
        "n_runs": 0,
        "n_with_expected": 0,
        "strict_matches": 0,
        "soft_matches": 0,
        "args_matches": 0,
        "args_valid_rate_per_call": [],
        "n_tool_calls": [],
        "total_latency_ms": [],
        "llm_latency_ms": [],
        "prompt_tokens": [],
        "completion_tokens": [],
        "terminated_by": defaultdict(int),
        "by_category": defaultdict(lambda: {"n": 0, "strict": 0, "soft": 0}),
    })

    for run in runs:
        label = run.get("label", "default")
        agg = by_label[label]
        agg["model"] = run.get("model", "?")
        agg["n_runs"] += 1
        agg["n_tool_calls"].append(run["n_tool_calls"])
        agg["total_latency_ms"].append(run["total_latency_ms"])
        agg["llm_latency_ms"].append(run["llm_latency_ms"])
        agg["prompt_tokens"].append(run.get("prompt_tokens", 0))
        agg["completion_tokens"].append(run.get("completion_tokens", 0))
        agg["terminated_by"][run["terminated_by"]] += 1

        expected = expected_by_query.get(run["user_query"])
        if not expected:
            continue
        agg["n_with_expected"] += 1
        cat = expected["category"]

        strict, soft = _check_tool_match(run["tool_sequence"], expected["expected_tools"])
        agg["strict_matches"] += int(strict)
        agg["soft_matches"] += int(soft)
        agg["by_category"][cat]["n"] += 1
        agg["by_category"][cat]["strict"] += int(strict)
        agg["by_category"][cat]["soft"] += int(soft)

        run_tools = tools_by_run.get(run["run_id"], [])
        if run_tools:
            valid_rate = sum(1 for t in run_tools if t["arguments_valid"]) / len(run_tools)
            agg["args_valid_rate_per_call"].append(valid_rate)

        if _check_args_subset(run_tools, expected.get("expected_args_subset", {})):
            agg["args_matches"] += 1

    _print_report(by_label)
    _save_csv(by_label)
    _save_markdown(by_label)


def _pct(num, den):
    return f"{100 * num / den:.1f}%" if den else "n/a"


def _stat(values):
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{mean(values):.1f}±{stdev(values):.1f}"


def _print_report(by_label: dict) -> None:
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS (by label)")
    print("=" * 80)
    for label, agg in by_label.items():
        n = agg["n_runs"]
        ne = agg["n_with_expected"]
        print(f"\n--- Label: {label}  (model: {agg['model']}) ---")
        print(f"  Runs total: {n}, runs with expected_tools: {ne}")
        print(f"  Strict tool-seq match:  {_pct(agg['strict_matches'], ne)} "
              f"({agg['strict_matches']}/{ne})")
        print(f"  Soft tool-set match:    {_pct(agg['soft_matches'], ne)} "
              f"({agg['soft_matches']}/{ne})")
        print(f"  Args subset match:      {_pct(agg['args_matches'], ne)} "
              f"({agg['args_matches']}/{ne})")
        if agg["args_valid_rate_per_call"]:
            print(f"  Args valid (avg/run):   {mean(agg['args_valid_rate_per_call']):.1%}")
        print(f"  Tool calls per run:     {_stat(agg['n_tool_calls'])}")
        print(f"  Total latency (ms):     {_stat(agg['total_latency_ms'])}")
        print(f"  LLM latency (ms):       {_stat(agg['llm_latency_ms'])}")
        print(f"  Prompt tokens (avg):    {_stat(agg['prompt_tokens'])}")
        print(f"  Compl. tokens (avg):    {_stat(agg['completion_tokens'])}")
        print(f"  Terminated by:          {dict(agg['terminated_by'])}")
        print(f"  By category (soft match):")
        for cat, c in sorted(agg["by_category"].items()):
            print(f"    {cat:34s} {_pct(c['soft'], c['n']):>8s}  ({c['soft']}/{c['n']})")


def _save_csv(by_label: dict) -> None:
    import csv
    path = LOG_DIR / "comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "label", "model", "n_runs", "n_with_expected",
            "strict_match_pct", "soft_match_pct", "args_match_pct",
            "args_valid_rate", "avg_tool_calls", "avg_total_latency_ms",
            "avg_llm_latency_ms", "avg_prompt_tokens", "avg_completion_tokens",
        ])
        for label, agg in by_label.items():
            ne = agg["n_with_expected"]
            w.writerow([
                label, agg["model"], agg["n_runs"], ne,
                100 * agg["strict_matches"] / ne if ne else 0,
                100 * agg["soft_matches"] / ne if ne else 0,
                100 * agg["args_matches"] / ne if ne else 0,
                mean(agg["args_valid_rate_per_call"]) if agg["args_valid_rate_per_call"] else 0,
                mean(agg["n_tool_calls"]) if agg["n_tool_calls"] else 0,
                mean(agg["total_latency_ms"]) if agg["total_latency_ms"] else 0,
                mean(agg["llm_latency_ms"]) if agg["llm_latency_ms"] else 0,
                mean(agg["prompt_tokens"]) if agg["prompt_tokens"] else 0,
                mean(agg["completion_tokens"]) if agg["completion_tokens"] else 0,
            ])
    print(f"\nCSV saved: {path}")


def _save_markdown(by_label: dict) -> None:
    lines = ["# Benchmark comparison (by label)\n"]
    lines.append("| Label | Model | Runs | Strict | Soft | Args | Calls/run | Total ms |")
    lines.append("|-------|-------|------|--------|------|------|-----------|----------|")
    for label, agg in by_label.items():
        ne = agg["n_with_expected"]
        lines.append(
            f"| `{label}` | `{agg['model']}` | {agg['n_runs']} | "
            f"{_pct(agg['strict_matches'], ne)} | "
            f"{_pct(agg['soft_matches'], ne)} | "
            f"{_pct(agg['args_matches'], ne)} | "
            f"{mean(agg['n_tool_calls']):.1f} | "
            f"{mean(agg['total_latency_ms']):.0f} |"
        )

    lines.append("\n## По категориям (soft match)\n")
    categories: set[str] = set()
    for agg in by_label.values():
        categories.update(agg["by_category"].keys())
    lines.append("| Category | " + " | ".join(by_label.keys()) + " |")
    lines.append("|----------|" + "|".join(["---"] * len(by_label)) + "|")
    for cat in sorted(categories):
        row = [cat]
        for label, agg in by_label.items():
            c = agg["by_category"].get(cat, {"n": 0, "soft": 0})
            row.append(_pct(c["soft"], c["n"]))
        lines.append("| " + " | ".join(row) + " |")

    path = LOG_DIR / "comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Regression benchmark for the geo agent (single model, multiple labels)."
    )
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "qwen3:14b"),
                        help="Ollama model tag. Default: env LLM_MODEL or qwen3:14b.")
    parser.add_argument("--label", default=None,
                        help="Label for this run (e.g. baseline, short_prompt, num_ctx_24k). "
                             "Used to compare configurations in --analyze.")
    parser.add_argument("--n-repeats", type=int, default=3,
                        help="Repeats of the full benchmark suite. Default: 3.")
    parser.add_argument("--queries", nargs="*",
                        help="Run only specific query ids (e.g. q01 q02). For quick smoke tests.")
    parser.add_argument("--analyze", action="store_true",
                        help="Skip running, just analyze existing logs.")
    args = parser.parse_args()

    if args.analyze:
        analyze()
        return

    if not args.label:
        sys.exit("--label is required when running benchmark. "
                 "Use e.g. --label baseline or --label v2_short_prompt.")

    run_benchmark(args.model, args.label, args.n_repeats, args.queries)
    print("\nRun `python -m evaluation.evaluate --analyze` to compare labels.")


if __name__ == "__main__":
    main()
