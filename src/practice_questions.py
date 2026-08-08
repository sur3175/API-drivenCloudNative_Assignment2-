# Sub-task: Practice Question Generation - the fine-tuned model (assignment req. 8).
#
# Turns a passage of study notes into exam-style practice questions, using the
# t5-small fine-tuned on SciQ by scripts/finetune_qa.py.
#
# The model is trained on
#     generate question: answer: <answer>  context: <passage>  ->  <question>
# so it needs an answer to build each question around. At training time that comes
# from the dataset; here the answers are picked out of the user's own notes by
# `candidate_answers` below, so the student only has to supply the material.

import os
import re
from pathlib import Path

from src.metrics import track

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "t5-small-sciq-qgen"
MODEL_NAME = "t5-small-sciq-qgen (fine-tuned)"
BASE_MODEL = "t5-small"

MAX_INPUT_TOKENS = 320
MAX_TARGET_TOKENS = 48

# Enough to keep the candidate-answer picker off function words. A full stopword
# list would be a dependency; this covers what actually shows up at the top of a
# frequency count for English prose.
STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "for", "are", "was", "were", "have",
    "has", "had", "not", "but", "they", "them", "their", "there", "then", "than",
    "which", "when", "what", "who", "whom", "whose", "how", "why", "can", "could",
    "would", "should", "will", "shall", "may", "might", "must", "into", "onto", "over",
    "under", "about", "after", "before", "between", "because", "while", "each", "every",
    "some", "any", "all", "one", "two", "its", "it's", "you", "your", "our", "his",
    "her", "him", "she", "he", "we", "us", "does", "did", "done", "being", "been",
    "such", "only", "also", "more", "most", "other", "another", "same", "own", "very",
    "just", "even", "still", "much", "many", "both", "cannot", "these", "those",
    # Frequent in prose and useless as an answer to a question.
    "without", "within", "through", "throughout", "however", "therefore", "instead",
    "rather", "always", "never", "often", "usually", "make", "makes", "made", "means",
    "meaning", "used", "using", "uses", "need", "needs", "needed", "given", "gives",
    "take", "takes", "become", "becomes", "comes", "goes", "keep", "keeps", "call",
    "calls", "called", "work", "works", "thing", "things", "example", "examples",
    "something", "anything", "everything", "nothing", "here", "where", "whereas",
}

# Acronyms are strong answers (API, SLO, CNCF) but too short for the general length
# rule, so they are matched separately.
MIN_TERM_LENGTH = 5

_model_cache = None


def is_available() -> bool:
    """True when the fine-tuned weights exist on disk."""
    return (MODEL_DIR / "config.json").exists()


def _load():
    global _model_cache
    if _model_cache is None:
        if not is_available():
            raise RuntimeError(
                f"No fine-tuned model at {MODEL_DIR}. Train it first:\n"
                f"    python scripts/finetune_qa.py --task qgen"
            )
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
        model.eval()
        _model_cache = (tokenizer, model)
    return _model_cache


def candidate_answers(context: str, count: int) -> list:
    """Pick terms from the notes worth building a question around.

    Frequency over content words, preferring longer terms - a crude but serviceable
    keyword extractor that needs no extra dependency. Terms are returned in the
    order they first appear so the generated questions follow the notes.
    """
    freq = {}
    display = {}
    for word in re.findall(r"[A-Za-z][A-Za-z'-]{1,}", context):
        acronym = word.isupper() and len(word) >= 3
        if not acronym and (len(word) < MIN_TERM_LENGTH or word.lower() in STOPWORDS):
            continue
        key = word.lower()
        freq[key] = freq.get(key, 0) + (2 if acronym else 1)
        # Keep the acronym's original casing; questions read better with "API" than "api".
        display.setdefault(key, word if acronym else key)

    # Score by frequency, nudged by length so "orchestration" outranks "state".
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1] * (1 + len(kv[0]) / 20), kv[0]))
    picked = [display[term] for term, _ in ranked[:count]]

    lowered = context.lower()
    picked.sort(key=lambda t: lowered.find(t.lower()))
    return picked


def _generate_one(tokenizer, model, answer: str, context: str) -> str:
    import torch

    prompt = f"generate question: answer: {answer.strip()}  context: {context.strip()}"
    inputs = tokenizer(
        prompt, max_length=MAX_INPUT_TOKENS, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_TARGET_TOKENS,
            num_beams=4,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def generate_questions(context: str, num_questions: int = 5, answers: list = None) -> dict:
    """Generate practice questions from `context`.

    Returns a dict with:
        questions   - list of {"question": str, "answer": str}
        model       - which weights produced them
        latency_sec, prompt_tokens, completion_tokens, total_tokens, cost_usd

    Raises ValueError on input too short, RuntimeError if the model isn't trained.
    """
    context = (context or "").strip()
    if len(context.split()) < 30:
        raise ValueError(
            "Please provide at least 30 words of study material to generate "
            "questions from."
        )

    tokenizer, model = _load()
    targets = answers or candidate_answers(context, num_questions)
    if not targets:
        raise ValueError("Could not find any terms in the material to ask about.")

    with track("practice_questions", MODEL_NAME) as run:
        questions = []
        for answer in targets[:num_questions]:
            question = _generate_one(tokenizer, model, answer, context)
            if question:
                questions.append({"question": question, "answer": answer})
        run.set_usage(
            int(len(context.split()) * 1.4 * len(questions)),
            int(sum(len(q["question"].split()) for q in questions) * 1.4),
        )
        run.extra = {
            "context_words": len(context.split()),
            "requested": num_questions,
            "produced": len(questions),
        }
        result = {
            "questions": questions,
            "model": MODEL_NAME,
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
            "total_tokens": run.total_tokens,
            "cost_usd": run.cost_usd,
        }

    result["latency_sec"] = round(run.latency_sec, 3)
    return result
