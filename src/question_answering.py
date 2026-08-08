# Sub-task: Question Answering (NLP category).
#
# Domain: Education. A student pastes study material - the same lecture notes or
# textbook section the summariser works on - and asks questions about it. Answers are
# grounded in that material rather than in the model's own knowledge, which is the
# behaviour a study assistant needs: if the notes do not cover it, the honest answer
# is "not in the material", not a confident guess.
#
# Two backends, mirroring the summarisation sub-task:
#   * "hf_local" -> distilbert-base-cased-distilled-squad, extractive. Selects the
#                   answer span from the context, so it cannot hallucinate. Runs
#                   locally, no key, no cost.
#   * "hf_api"   -> Qwen2.5-7B-Instruct via the Inference API, generative. Reads
#                   better and handles questions whose answer is spread across
#                   several sentences, but has to be held to the context by prompt.

import os
import re

from config import create_client
from src.metrics import track

OPENAI_MODEL = "gpt-4o-mini"
HF_LOCAL_MODEL = "distilbert-base-cased-distilled-squad"
HF_API_MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"
HF_API_MODEL = os.getenv("HF_API_MODEL") or HF_API_MODEL_DEFAULT

# What the generative backends are told. The refusal instruction is the important
# part: without it the model answers from its own knowledge and the answer stops
# being evidence of what the study material says.
SYSTEM_PROMPT = (
    "You answer student questions using only the study material provided. "
    "Quote or paraphrase the material; never add facts from your own knowledge. "
    "If the material does not contain the answer, reply exactly: "
    "The study material does not cover this. "
    "Keep answers under 80 words unless the question needs a list."
)

NO_ANSWER = "The study material does not cover this."

# distilbert's encoder takes 512 tokens. The pipeline slides a window over longer
# contexts and keeps the best-scoring span; these are its window and overlap.
MAX_SEQ_LEN = 384
DOC_STRIDE = 128

_qa_cache = None


def _words(text: str) -> list:
    return re.findall(r"[a-z0-9']+", text.lower())


def groundedness(answer: str, context: str) -> float:
    """Fraction of the answer's content words that appear in the context.

    The quality metric for the generative backends. An extractive answer scores 1.0
    by construction; a generative answer that drifts into the model's own knowledge
    drops well below it, which is exactly the failure we want visible in the metrics.
    """
    answer_words = [w for w in _words(answer) if len(w) > 3]
    if not answer_words:
        return 0.0
    context_words = set(_words(context))
    hits = sum(1 for w in answer_words if w in context_words)
    return round(hits / len(answer_words), 4)


def _get_qa_pipeline():
    global _qa_cache
    if _qa_cache is None:
        from transformers import pipeline

        _qa_cache = pipeline("question-answering", model=HF_LOCAL_MODEL)
    return _qa_cache


def _answer_hf_local(question: str, context: str, run) -> dict:
    """Extractive span selection. Returns the span plus the model's confidence."""
    qa = _get_qa_pipeline()
    result = qa(
        question=question,
        context=context,
        max_seq_len=MAX_SEQ_LEN,
        doc_stride=DOC_STRIDE,
        handle_impossible_answer=True,
    )
    answer = (result.get("answer") or "").strip()
    score = float(result.get("score", 0.0))

    # An empty span, or one the model is barely confident in, is a refusal. Reporting
    # a low-confidence guess as an answer is worse than saying the notes don't cover it.
    if not answer or score < 0.10:
        answer = NO_ANSWER
        evidence = ""
    else:
        evidence = _surrounding_sentence(context, result.get("start", 0), result.get("end", 0))

    run.set_usage(int(len(_words(context)) * 1.4), int(len(_words(answer)) * 1.4))
    return {"answer": answer, "confidence": round(score, 4), "evidence": evidence}


def _surrounding_sentence(context: str, start: int, end: int) -> str:
    """The sentence containing the extracted span, as the citation shown to the user."""
    left = context.rfind(".", 0, start)
    right = context.find(".", end)
    left = 0 if left == -1 else left + 1
    right = len(context) if right == -1 else right + 1
    return context[left:right].strip()


def _chat(client, model, question, context, run):
    """One chat-completion round trip, shared by the API-based backends."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Study material:\n---\n{context}\n---\n\nQuestion: {question}"
                ),
            },
        ],
        max_tokens=300,
        temperature=0.2,
    )
    usage = getattr(response, "usage", None)
    if usage is not None:
        run.set_usage(
            getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)
        )
    return (response.choices[0].message.content or "").strip()


def _answer_hf_api(question: str, context: str, run) -> dict:
    from huggingface_hub import InferenceClient

    token = _hf_token()
    if not token:
        raise RuntimeError(
            "HF_API_KEY not found. Add it to .env, or use the local backend, "
            "which needs no key."
        )
    answer = _chat(InferenceClient(api_key=token), HF_API_MODEL, question, context, run)
    return {"answer": answer, "confidence": None, "evidence": ""}


def _answer_openai(question: str, context: str, run) -> dict:
    answer = _chat(create_client("openai"), OPENAI_MODEL, question, context, run)
    return {"answer": answer, "confidence": None, "evidence": ""}


def _hf_token() -> str:
    return os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN") or ""


BACKENDS = {
    "hf_local": {"fn": _answer_hf_local, "model": HF_LOCAL_MODEL, "extractive": True},
    "hf_api": {"fn": _answer_hf_api, "model": HF_API_MODEL, "extractive": False},
    "openai": {"fn": _answer_openai, "model": OPENAI_MODEL, "extractive": False},
}

DEFAULT_BACKEND = "hf_local"


def answer_question(question: str, context: str, backend: str = DEFAULT_BACKEND) -> dict:
    """Answer `question` from `context` and return the answer with its metrics.

    Returns a dict with:
        answer         - the answer text, or NO_ANSWER when the material doesn't cover it
        model          - which model produced it
        extractive     - True if the answer is a span copied from the context
        confidence     - span probability (extractive backends only, else None)
        groundedness   - fraction of answer content words found in the context
        evidence       - the sentence the span came from (extractive only)
        answered       - False when the backend declined to answer
        latency_sec, prompt_tokens, completion_tokens, total_tokens, cost_usd

    Raises ValueError on empty input.
    """
    question = (question or "").strip()
    context = (context or "").strip()
    if not question:
        raise ValueError("Please enter a question.")
    if not context:
        raise ValueError(
            "Please provide the study material to answer from - paste it into the "
            "context box or upload a document."
        )
    if len(context.split()) < 20:
        raise ValueError("The study material is too short to answer from.")
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from {list(BACKENDS)}.")

    spec = BACKENDS[backend]
    with track("question_answering", spec["model"]) as run:
        out = spec["fn"](question, context, run)
        answer = out["answer"]
        answered = answer.strip().rstrip(".") != NO_ANSWER.rstrip(".")
        score = groundedness(answer, context) if answered else 1.0
        run.quality = score
        run.extra = {
            "backend": backend,
            "question": question[:200],
            "context_words": len(context.split()),
            "answered": answered,
            "confidence": out["confidence"],
        }
        result = {
            "answer": answer,
            "model": spec["model"],
            "extractive": spec["extractive"],
            "confidence": out["confidence"],
            "groundedness": score,
            "evidence": out["evidence"],
            "answered": answered,
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
            "total_tokens": run.total_tokens,
            "cost_usd": run.cost_usd,
        }

    result["latency_sec"] = round(run.latency_sec, 3)
    return result
