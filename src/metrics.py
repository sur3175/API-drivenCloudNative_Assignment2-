# LLMOps metrics layer, shared by every sub-task of the application.
#
# Requirement 7 of the assignment asks us to "apply LLMOps principles, and measure
# at least 5 relevant metrics". Every call made to a model goes through the
# `track()` context manager below, which records one row per invocation into
# Data/metrics_log.csv so the numbers can be charted and discussed in the report.
#
# Metrics captured per invocation:
#   1. latency_sec        - wall-clock time of the model call
#   2. prompt_tokens      - tokens sent to the model
#   3. completion_tokens  - tokens produced by the model
#   4. cost_usd           - estimated cost, from the published per-token price
#   5. success            - 1 / 0, so we can compute a success (error) rate
#   6. quality            - task-specific quality score (e.g. ROUGE-L for summaries)
#   7. extra              - free-form JSON for task-specific detail

import csv
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from math import ceil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_FILE = PROJECT_ROOT / "Data" / "metrics_log.csv"

FIELDNAMES = [
    "timestamp",
    "task",
    "model",
    "latency_sec",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_usd",
    "success",
    "quality",
    "error",
    "extra",
]

# USD per 1,000,000 tokens. Source: OpenAI pricing page.
#
# The Hugging Face entries are 0.0, but for two different reasons. Locally-run
# models genuinely cost nothing per token - they burn CPU/GPU time instead, which
# the latency metric is what captures. Inference API models are billed against a
# monthly credit allowance rather than per token, so there is no per-token price to
# record; watch the call count and latency for those instead of the cost column.
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-image-1": {"input": 5.00, "output": 0.00},
    # Hugging Face - local
    "sshleifer/distilbart-cnn-12-6": {"input": 0.0, "output": 0.0},
    "facebook/bart-large-cnn": {"input": 0.0, "output": 0.0},
    # Hugging Face - Inference API
    "Qwen/Qwen2.5-7B-Instruct": {"input": 0.0, "output": 0.0},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimated USD cost of a single call. Unknown models are treated as free."""
    price = PRICING.get(model)
    if price is None:
        return 0.0
    return round(
        (prompt_tokens / 1_000_000) * price["input"]
        + (completion_tokens / 1_000_000) * price["output"],
        6,
    )


class Run:
    """Mutable record filled in by the caller while the model call is in flight."""

    def __init__(self, task: str, model: str):
        self.task = task
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.quality = None
        self.extra = {}
        self.latency_sec = 0.0
        self.success = True
        self.error = ""

    def set_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return estimate_cost(self.model, self.prompt_tokens, self.completion_tokens)

    def as_dict(self) -> dict:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task": self.task,
            "model": self.model,
            "latency_sec": round(self.latency_sec, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "success": int(self.success),
            "quality": "" if self.quality is None else round(float(self.quality), 4),
            "error": self.error,
            "extra": json.dumps(self.extra, ensure_ascii=False),
        }


@contextmanager
def track(task: str, model: str):
    """Time a model call and append one row to the metrics log.

    Usage:
        with track("summarization", "gpt-4o-mini") as run:
            response = client.responses.create(...)
            run.set_usage(response.usage.input_tokens, response.usage.output_tokens)
    """
    run = Run(task, model)
    started = time.perf_counter()
    try:
        yield run
    except Exception as exc:
        run.success = False
        run.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        run.latency_sec = time.perf_counter() - started
        log_run(run)


def log_run(run: Run) -> None:
    """Append a run to Data/metrics_log.csv, creating the file with a header."""
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    is_new = not METRICS_FILE.exists() or METRICS_FILE.stat().st_size == 0
    with METRICS_FILE.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(run.as_dict())


def load_metrics() -> list:
    """Read the metrics log back as a list of dicts (empty if nothing logged yet)."""
    if not METRICS_FILE.exists():
        return []
    with METRICS_FILE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarise_metrics(task: str = None) -> dict:
    """Aggregate the log into the headline numbers we report in the document."""
    rows = [r for r in load_metrics() if task is None or r["task"] == task]
    if not rows:
        return {}

    def nums(field):
        out = []
        for r in rows:
            if r.get(field):
                try:
                    out.append(float(r[field]))
                except ValueError:
                    pass
        return out

    latencies = sorted(nums("latency_sec"))
    qualities = nums("quality")
    successes = [float(r["success"]) for r in rows if r.get("success")]

    return {
        "calls": len(rows),
        "avg_latency_sec": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        # Nearest-rank p95: the smallest value at or above 95% of observations.
        "p95_latency_sec": round(latencies[min(len(latencies) - 1, ceil(0.95 * len(latencies)) - 1)], 3) if latencies else 0.0,
        "total_tokens": int(sum(nums("total_tokens"))),
        "total_cost_usd": round(sum(nums("cost_usd")), 6),
        "success_rate": round(sum(successes) / len(rows), 4) if successes else 0.0,
        "avg_quality": round(sum(qualities) / len(qualities), 4) if qualities else None,
    }
