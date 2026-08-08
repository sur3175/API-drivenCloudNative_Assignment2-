# Sub-task: Text Summarisation (NLP category).
#
# Domain: Education. The summariser condenses long-form study material - lecture
# transcripts, textbook chapters, research papers, student essays - into the form a
# learner actually needs: a short abstract, revision bullet points, exam takeaways
# or a plain-language explanation.
#
# Two model backends are supported so the report can compare an LLM against an SLM:
#   * "openai" -> gpt-4o-mini via the OpenAI API (default, best quality)
#   * "slm"    -> sshleifer/distilbart-cnn-12-6 run locally via Hugging Face
#                 transformers (no API key, no per-token cost)
#
# Documents longer than one context-friendly chunk are handled with a map-reduce
# strategy: summarise each chunk, then summarise the chunk summaries.

import re
from difflib import SequenceMatcher

from src.metrics import track

OPENAI_MODEL = "gpt-4o-mini"
SLM_MODEL = "sshleifer/distilbart-cnn-12-6"

# Words per chunk when a document has to be split. Comfortably inside the context
# window of both backends while keeping enough context for a coherent summary.
CHUNK_WORDS = 900

# The summary "styles" exposed in the UI. Each maps to an instruction appended to
# the system prompt, which is how we steer one model across several output formats.
STYLES = {
    "Concise abstract": (
        "Write a single flowing paragraph that captures the central argument, the "
        "supporting evidence and the conclusion."
    ),
    "Revision bullet points": (
        "Write between 5 and 8 bullet points. Each bullet must be a complete, "
        "self-contained fact a student could revise from."
    ),
    "Study notes": (
        "Write structured study notes using short markdown headings for each theme, "
        "with two to four bullet points under every heading. Bold the key terms."
    ),
    "Exam key takeaways": (
        "List the points most likely to be examined, numbered, each one sentence "
        "long, ordered from most to least important."
    ),
    "Explain simply": (
        "Explain the material in plain language a first-year student with no "
        "background in the subject could follow. Avoid jargon; when a technical "
        "term is unavoidable, define it in the same sentence."
    ),
}

DEFAULT_STYLE = "Concise abstract"

SYSTEM_PROMPT = (
    "You are an academic summarisation assistant for an educational platform. "
    "You condense study material faithfully: never invent facts, never add opinions, "
    "and never introduce information that is not present in the source text. "
    "If the source is too short or empty to summarise, say so plainly."
)

_slm_cache = None  # lazily created, so importing this module stays cheap


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

def word_count(text: str) -> int:
    return len(text.split())


def chunk_text(text: str, chunk_words: int = None) -> list:
    """Split text into chunks of roughly `chunk_words` words, respecting paragraphs.

    Paragraphs are kept whole where possible; a paragraph longer than the budget on
    its own is split on sentence boundaries rather than mid-sentence. The budget is
    resolved at call time so CHUNK_WORDS can be tuned when benchmarking.
    """
    chunk_words = chunk_words or CHUNK_WORDS
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    units = []
    for paragraph in paragraphs:
        if word_count(paragraph) <= chunk_words:
            units.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        buffer = []
        for sentence in sentences:
            buffer.append(sentence)
            if word_count(" ".join(buffer)) >= chunk_words:
                units.append(" ".join(buffer))
                buffer = []
        if buffer:
            units.append(" ".join(buffer))

    chunks = []
    current = []
    for unit in units:
        candidate = current + [unit]
        if current and word_count("\n\n".join(candidate)) > chunk_words:
            chunks.append("\n\n".join(current))
            current = [unit]
        else:
            current = candidate
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _tokens(text: str) -> list:
    return re.findall(r"[a-z0-9']+", text.lower())


def rouge_scores(summary: str, source: str) -> dict:
    """ROUGE-1 and ROUGE-L F1 of the summary against its source text.

    Implemented directly rather than pulled from a package so the project has no
    extra dependency. Used as the automatic quality metric for the summariser: a
    summary made only of words and orderings found in the source scores highly,
    which flags hallucination and off-topic drift.
    """
    summary_tokens = _tokens(summary)
    source_tokens = _tokens(source)
    if not summary_tokens or not source_tokens:
        return {"rouge1": 0.0, "rougeL": 0.0}

    # ROUGE-1: unigram overlap, counting repeats only as often as they appear.
    source_counts = {}
    for token in source_tokens:
        source_counts[token] = source_counts.get(token, 0) + 1
    overlap = 0
    for token in summary_tokens:
        if source_counts.get(token, 0) > 0:
            source_counts[token] -= 1
            overlap += 1
    precision = overlap / len(summary_tokens)
    recall = overlap / len(source_tokens)
    rouge1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    # ROUGE-L: longest common subsequence, order-sensitive.
    matcher = SequenceMatcher(None, summary_tokens, source_tokens, autojunk=False)
    lcs = sum(block.size for block in matcher.get_matching_blocks())
    precision_l = lcs / len(summary_tokens)
    recall_l = lcs / len(source_tokens)
    rougeL = 0.0 if precision_l + recall_l == 0 else 2 * precision_l * recall_l / (precision_l + recall_l)

    return {"rouge1": round(rouge1, 4), "rougeL": round(rougeL, 4)}


def compression_ratio(source: str, summary: str) -> float:
    """How many source words each summary word replaces. 10.0 means a 10x squeeze."""
    summary_words = word_count(summary)
    if summary_words == 0:
        return 0.0
    return round(word_count(source) / summary_words, 2)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

def _build_prompt(text: str, style: str, target_words: int, focus: str = "") -> str:
    instruction = STYLES.get(style, STYLES[DEFAULT_STYLE])
    parts = [
        instruction,
        f"Keep the summary to roughly {target_words} words.",
    ]
    if focus.strip():
        parts.append(
            f"Give particular attention to this aspect of the material: {focus.strip()}."
        )
    parts.append("Source material follows.\n\n---\n" + text + "\n---")
    return "\n".join(parts)


def _summarize_openai(text: str, style: str, target_words: int, focus: str, run) -> str:
    """One OpenAI call. `run` is the metrics record for the enclosing invocation."""
    # Imported here rather than at module scope: config.py raises if OPENAI_API_KEY is
    # missing, and the local SLM backend must stay usable without an API key.
    from config import client

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=SYSTEM_PROMPT,
        input=_build_prompt(text, style, target_words, focus),
    )
    usage = getattr(response, "usage", None)
    if usage is not None:
        run.set_usage(
            getattr(usage, "input_tokens", 0),
            getattr(usage, "output_tokens", 0),
        )
    return response.output_text.strip()


def _get_slm():
    """Load the DistilBART tokenizer and model once, on first use.

    transformers 5.x removed the "summarization" pipeline task, so the seq2seq model
    is driven directly - which is also what lets us control beam search explicitly.
    """
    global _slm_cache
    if _slm_cache is None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(SLM_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(SLM_MODEL)
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        _slm_cache = (tokenizer, model)
    return _slm_cache


def _summarize_slm(text: str, style: str, target_words: int, focus: str, run) -> str:
    """Local SLM path. DistilBART is abstractive but does not follow style
    instructions, so we only steer it with length and note the style in the log."""
    import torch

    tokenizer, model = _get_slm()
    # DistilBART thinks in tokens; ~1.4 tokens per English word is a safe conversion.
    max_new = max(56, int(target_words * 1.4))
    min_new = max(24, int(max_new * 0.4))

    # BART's encoder is capped at 1024 positions, hence the truncation.
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            # 2 beams rather than 4: on CPU the quality gain does not justify
            # doubling an already slow generation step.
            num_beams=2,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            do_sample=False,
        )
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    # No billed usage for a local model, but token counts still drive the
    # throughput metrics, so approximate them from the word counts.
    run.set_usage(int(word_count(text) * 1.4), int(word_count(summary) * 1.4))
    return summary


# `style_aware` records whether the backend can follow a style instruction. An
# instruction-tuned LLM can; DistilBART cannot - it only ever produces a plain
# abstract - which changes how the map-reduce step below is finished off.
BACKENDS = {
    "openai": {"fn": _summarize_openai, "model": OPENAI_MODEL, "style_aware": True},
    "slm": {"fn": _summarize_slm, "model": SLM_MODEL, "style_aware": False},
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def summarize(
    text: str,
    style: str = DEFAULT_STYLE,
    target_words: int = 150,
    backend: str = "openai",
    focus: str = "",
) -> dict:
    """Summarise `text` and return the summary together with its LLMOps metrics.

    Returns a dict with keys:
        summary            - the summary text
        model              - which model produced it
        chunks             - how many chunks the source was split into
        source_words       - word count of the input
        summary_words      - word count of the output
        compression_ratio  - source words per summary word
        rouge1 / rougeL    - faithfulness of the summary to the source
        latency_sec        - wall-clock time of the whole invocation
        prompt_tokens / completion_tokens / total_tokens
        cost_usd           - estimated cost of the invocation

    Raises ValueError when the input is empty or too short to be worth summarising.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Nothing to summarise - please provide some text.")
    if word_count(text) < 40:
        raise ValueError(
            "The text is too short to summarise meaningfully. "
            "Please provide at least 40 words."
        )
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from {list(BACKENDS)}.")

    spec = BACKENDS[backend]
    summarize_fn, model_name = spec["fn"], spec["model"]
    chunks = chunk_text(text)

    with track("summarization", model_name) as run:
        if len(chunks) == 1:
            summary = summarize_fn(chunks[0], style, target_words, focus, run)
        else:
            # Map: summarise every chunk to a share of the target length.
            per_chunk_words = max(60, target_words // len(chunks) + 40)
            partials = [
                summarize_fn(chunk, "Concise abstract", per_chunk_words, focus, run)
                for chunk in chunks
            ]
            plain = "\n\n".join(partials)
            if not spec["style_aware"] and word_count(plain) <= target_words * 1.3:
                # A backend that cannot follow a style instruction gains nothing from
                # a second pass once the partials already fit the target length - and
                # re-summarising its own output only degrades it further.
                summary = plain
            else:
                # Reduce: summarise the partial summaries into the requested style.
                joined = "\n\n".join(
                    f"Section {i}: {p}" for i, p in enumerate(partials, start=1)
                )
                summary = summarize_fn(joined, style, target_words, focus, run)

        scores = rouge_scores(summary, text)
        run.quality = scores["rougeL"]
        run.extra = {
            "style": style,
            "backend": backend,
            "chunks": len(chunks),
            "source_words": word_count(text),
            "summary_words": word_count(summary),
            "compression_ratio": compression_ratio(text, summary),
            "rouge1": scores["rouge1"],
            "focus": focus.strip(),
        }
        result = {
            "summary": summary,
            "model": model_name,
            "chunks": len(chunks),
            "source_words": word_count(text),
            "summary_words": word_count(summary),
            "compression_ratio": compression_ratio(text, summary),
            "rouge1": scores["rouge1"],
            "rougeL": scores["rougeL"],
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
            "total_tokens": run.total_tokens,
            "cost_usd": run.cost_usd,
        }

    # `latency_sec` is only final once the tracking context has closed.
    result["latency_sec"] = round(run.latency_sec, 3)
    return result


def summarize_text(text: str) -> str:
    """Thin wrapper matching the signature style of the other sub-task modules."""
    return summarize(text)["summary"]
