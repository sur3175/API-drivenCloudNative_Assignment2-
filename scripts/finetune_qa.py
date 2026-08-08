"""Fine-tune t5-small on SciQ for the Education domain (assignment req. 8).

Domain fit: SciQ is 11,679 crowdsourced science exam questions, each with a support
paragraph - school science material, the same Education domain the rest of the
application targets.

Two tasks are supported.

`--task qgen` (default) - PRACTICE QUESTION GENERATION.

    input   ->  generate question: answer: <correct_answer>  context: <support>
    target  ->  <question>

    Turns a passage of notes into exam-style practice questions, which is a genuine
    study-assistant feature and something base t5-small cannot do: the prefix is not
    in its pre-training mixture, so it falls back to copying the input.

`--task qa` - GROUNDED QUESTION ANSWERING.

    input   ->  question: <q>  context: <support>
    target  ->  <correct_answer>

    Kept because it is a useful negative result, not because it is a good demo.
    Base t5-small already scores 70.0% EM / 81.2% F1 on this, since T5's original
    pre-training mixture includes SQuAD in exactly this format. There is almost no
    headroom, so fine-tuning here demonstrates very little. Measured, not assumed -
    see data/finetuning_results_qa.json.

A plain PyTorch loop is used instead of Trainer: this box is CPU-only, the model is
small, and the loop keeps the moving parts visible for the viva.

    python scripts/finetune_qa.py                    # train + evaluate qgen
    python scripts/finetune_qa.py --evaluate-only    # re-run evaluation
    python scripts/finetune_qa.py --task qa          # the negative-result baseline
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE_MODEL = "t5-small"
MODELS_DIR = ROOT / "models"

MAX_INPUT_TOKENS = 320

# Task definitions: how to build the prompt, what the target is, and how long the
# generated text is allowed to be. Questions need more room than short answers.
TASKS = {
    "qgen": {
        "prompt": lambda r: (
            f"generate question: answer: {r['correct_answer'].strip()}  "
            f"context: {r['support'].strip()}"
        ),
        "target": lambda r: r["question"].strip(),
        "max_target_tokens": 48,
        "output_dir": MODELS_DIR / "t5-small-sciq-qgen",
        "results": ROOT / "data" / "finetuning_results.json",
    },
    "qa": {
        "prompt": lambda r: (
            f"question: {r['question'].strip()}  context: {r['support'].strip()}"
        ),
        "target": lambda r: r["correct_answer"].strip(),
        "max_target_tokens": 24,
        "output_dir": MODELS_DIR / "t5-small-sciq-qa",
        "results": ROOT / "data" / "finetuning_results_qa.json",
    },
}


# --------------------------------------------------------------------------- #
# SQuAD-style metrics
# --------------------------------------------------------------------------- #

def normalise(text: str) -> str:
    """Lowercase, strip articles and punctuation - the standard SQuAD normalisation."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, truth: str) -> float:
    return float(normalise(prediction) == normalise(truth))


def token_f1(prediction: str, truth: str) -> float:
    """Token-overlap F1. Partial credit where exact match is too strict."""
    pred_tokens = normalise(prediction).split()
    truth_tokens = normalise(truth).split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, truth: str) -> float:
    """LCS-based F1. Order-sensitive, so it rewards getting the phrasing right and
    not just the bag of words - which matters for generated questions."""
    pred = normalise(prediction).split()
    ref = normalise(truth).split()
    if not pred or not ref:
        return 0.0
    matcher = SequenceMatcher(None, pred, ref, autojunk=False)
    lcs = sum(b.size for b in matcher.get_matching_blocks())
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def copy_rate(prediction: str, source_context: str) -> float:
    """How much of the output is lifted verbatim from the input context.

    Base t5-small, given a prefix it was never trained on, tends to echo the input
    instead of generating. This makes that failure mode measurable rather than
    something you have to eyeball in the samples.
    """
    pred = normalise(prediction).split()
    if not pred:
        return 0.0
    ctx = set(normalise(source_context).split())
    return sum(1 for w in pred if w in ctx) / len(pred)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_sciq(train_size: int, eval_size: int):
    """SciQ, with the support-less rows dropped - they cannot be answered from context."""
    from datasets import load_dataset

    train = load_dataset("allenai/sciq", split="train")
    test = load_dataset("allenai/sciq", split="test")

    def usable(row):
        return bool(row["support"].strip()) and bool(row["correct_answer"].strip())

    train = train.filter(usable)
    test = test.filter(usable)
    print(f"  usable train rows: {len(train)}, usable test rows: {len(test)}")

    train = train.select(range(min(train_size, len(train))))
    test = test.select(range(min(eval_size, len(test))))
    return train, test


class SciQDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, task):
        self.rows = rows
        self.tokenizer = tokenizer
        self.task = task

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        model_inputs = self.tokenizer(
            self.task["prompt"](row),
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = self.tokenizer(
            self.task["target"](row),
            max_length=self.task["max_target_tokens"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).input_ids
        # -100 tells the loss to ignore padding positions.
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": model_inputs.input_ids[0],
            "attention_mask": model_inputs.attention_mask[0],
            "labels": labels[0],
        }


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #

def train(train_rows, task, epochs, batch_size, lr):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model.train()

    loader = DataLoader(
        SciQDataset(train_rows, tokenizer, task), batch_size=batch_size, shuffle=True
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(loader) * epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimiser, start_factor=1.0, end_factor=0.1, total_iters=total_steps
    )

    print(f"\nTraining: {len(train_rows)} examples, {epochs} epoch(s), "
          f"batch {batch_size}, {total_steps} steps")
    started = time.perf_counter()
    step = 0
    running = 0.0
    for epoch in range(epochs):
        for batch in loader:
            optimiser.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            running += out.loss.item()
            step += 1
            if step % 10 == 0 or step == total_steps:
                elapsed = time.perf_counter() - started
                per_step = elapsed / step
                print(f"  step {step}/{total_steps}  loss {running / 10:.4f}  "
                      f"{per_step:.2f}s/step  eta {(total_steps - step) * per_step / 60:.1f}m",
                      flush=True)
                running = 0.0
    minutes = (time.perf_counter() - started) / 60

    task["output_dir"].mkdir(parents=True, exist_ok=True)
    model.save_pretrained(task["output_dir"])
    tokenizer.save_pretrained(task["output_dir"])
    print(f"\nTrained in {minutes:.1f} min. Saved to {task['output_dir']}")
    return minutes


@torch.no_grad()
def evaluate(model_path, test_rows, task, label):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    ems, f1s, rouges, copies, latencies, samples = [], [], [], [], [], []
    for i, row in enumerate(test_rows):
        prompt = task["prompt"](row)
        inputs = tokenizer(
            prompt, max_length=MAX_INPUT_TOKENS, truncation=True, return_tensors="pt"
        )
        started = time.perf_counter()
        output = model.generate(
            **inputs, max_new_tokens=task["max_target_tokens"], num_beams=2
        )
        latencies.append(time.perf_counter() - started)
        prediction = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        truth = task["target"](row)
        ems.append(exact_match(prediction, truth))
        f1s.append(token_f1(prediction, truth))
        rouges.append(rouge_l(prediction, truth))
        copies.append(copy_rate(prediction, row["support"]))
        if i < 8:
            samples.append({"expected": truth, "predicted": prediction})

    def pct(values):
        return round(100 * sum(values) / len(values), 2)

    result = {
        "model": label,
        "examples": len(test_rows),
        "exact_match": pct(ems),
        "token_f1": pct(f1s),
        "rouge_l": pct(rouges),
        "copy_rate": pct(copies),
        "avg_latency_sec": round(sum(latencies) / len(latencies), 3),
        "samples": samples,
    }
    print(f"\n{label}: EM {result['exact_match']}%  F1 {result['token_f1']}%  "
          f"ROUGE-L {result['rouge_l']}%  copied {result['copy_rate']}%  "
          f"({result['avg_latency_sec']}s/item)")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASKS), default="qgen")
    parser.add_argument("--train-size", type=int, default=1600)
    parser.add_argument("--eval-size", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    task = TASKS[args.task]
    print(f"Task: {args.task}")
    print("Loading SciQ...")
    train_rows, test_rows = load_sciq(args.train_size, args.eval_size)

    minutes = None
    if not args.evaluate_only:
        minutes = train(train_rows, task, args.epochs, args.batch_size, args.lr)
    elif not task["output_dir"].exists():
        sys.exit(
            f"No fine-tuned model at {task['output_dir']}. "
            "Run without --evaluate-only first."
        )

    print("\nEvaluating on held-out SciQ test data...")
    before = evaluate(BASE_MODEL, test_rows, task, f"{BASE_MODEL} (base)")
    after = evaluate(task["output_dir"], test_rows, task, f"{BASE_MODEL} (fine-tuned)")

    results = {
        "task": args.task,
        "dataset": "allenai/sciq",
        "base_model": BASE_MODEL,
        "train_examples": len(train_rows),
        "epochs": args.epochs,
        "train_minutes": round(minutes, 1) if minutes else None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "before": before,
        "after": after,
        "delta": {
            key: round(after[key] - before[key], 2)
            for key in ("exact_match", "token_f1", "rouge_l", "copy_rate")
        },
    }
    task["results"].parent.mkdir(parents=True, exist_ok=True)
    task["results"].write_text(json.dumps(results, indent=2), encoding="utf-8")

    cols = ("exact_match", "token_f1", "rouge_l", "copy_rate")
    print("\n" + "=" * 74)
    print(f"{'':<22}" + "".join(f"{c.replace('_', ' '):>13}" for c in cols))
    print(f"{'base t5-small':<22}" + "".join(f"{before[c]:>13}" for c in cols))
    print(f"{'fine-tuned':<22}" + "".join(f"{after[c]:>13}" for c in cols))
    print(f"{'change':<22}" + "".join(f"{results['delta'][c]:>+13}" for c in cols))
    print("=" * 74)
    print(f"\nResults written to {task['results']}")


if __name__ == "__main__":
    main()
