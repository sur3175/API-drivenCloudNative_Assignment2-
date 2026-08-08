"""Benchmark the summarisation backends against each other.

Runs every available backend over the same document with identical settings and
prints a comparison table. This is the LLMOps evidence for the report: the numbers
in the Word document should come from a run of this script, not from memory.

    python scripts/benchmark_summarization.py
    python scripts/benchmark_summarization.py --backends hf_local hf_api
    python scripts/benchmark_summarization.py --input Data/my_notes.txt

Every run also appends to Data/metrics_log.csv via the shared metrics layer.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.summarization import BACKENDS, DEFAULT_STYLE, summarize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Data/sample_lecture_notes.txt")
    parser.add_argument("--backends", nargs="+", default=["hf_local", "hf_api"])
    parser.add_argument("--style", default=DEFAULT_STYLE)
    parser.add_argument("--target-words", type=int, default=150)
    args = parser.parse_args()

    source = (ROOT / args.input).read_text(encoding="utf-8")
    print(f"Input: {args.input} ({len(source.split())} words)")
    print(f"Style: {args.style} · target {args.target_words} words\n")

    rows = []
    for backend in args.backends:
        if backend not in BACKENDS:
            print(f"  {backend}: unknown backend, skipping")
            continue
        try:
            result = summarize(
                source,
                style=args.style,
                target_words=args.target_words,
                backend=backend,
            )
        except Exception as exc:
            print(f"  {backend}: FAILED - {type(exc).__name__}: {exc}")
            continue
        rows.append((backend, result))
        print(f"  {backend}: done in {result['latency_sec']}s")

    if not rows:
        print("\nNo backend produced a summary.")
        return 1

    header = (
        f"\n{'backend':<10} {'model':<32} {'latency':>9} {'tokens':>7} "
        f"{'cost $':>9} {'words':>6} {'compr':>7} {'R-1':>6} {'R-L':>6} {'style by':>9}"
    )
    print(header)
    print("-" * len(header))
    for backend, r in rows:
        print(
            f"{backend:<10} {r['model'][:32]:<32} {r['latency_sec']:>8.2f}s "
            f"{r['total_tokens']:>7} {r['cost_usd']:>9.5f} {r['summary_words']:>6} "
            f"{r['compression_ratio']:>6.1f}x {r['rouge1']:>6.3f} {r['rougeL']:>6.3f} "
            f"{r['style_applied_by']:>9}"
        )

    print(
        "\nNote: ROUGE here is measured against the SOURCE, not a human reference "
        "summary.\nIt rewards reusing the source's own wording, so an extractive "
        "model scores higher\nthan an abstractive one that paraphrases well. Read it "
        "as a faithfulness signal,\nnot as a ranking of summary quality."
    )

    for backend, r in rows:
        print(f"\n--- {backend} ({r['model']}) ---\n{r['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
