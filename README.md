# API-drivenCloudNative_Assignment2-

Designed and developed an interactive API-driven AI application combining NLP and Computer Vision with 5+ AI sub-tasks. Integrated LLM/SLM models via Hugging Face/OpenAI APIs, applied LLMOps with performance metrics, and fine-tuned a domain-specific model using a relevant dataset.

**Domain:** Education
**Categories:** Natural Language Processing + Computer Vision

## Sub-tasks

| # | Sub-task | Category | Module | Model | Status |
|---|----------|----------|--------|-------|--------|
| 1 | Text generation | NLP | `src/text_generation.py` | gpt-4o-mini | Done |
| 2 | Text summarisation | NLP | `src/summarization.py` | DistilBART local (SLM) / Qwen2.5-7B via HF API (LLM) | Done |
| 3 | Question answering | NLP | `src/question_answering.py` | — | Pending |
| 4 | Image generation | CV | `src/image_generation.py` | gpt-image-1 | Done |
| 5 | Image classification | CV | `src/image_classification.py` | — | Pending |

All five feed one objective: an AI study assistant that generates study material,
condenses it for revision, answers questions on it, and works with diagrams.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Summarisation works out of the box with **no API key and no credits** — its default
backend is a Hugging Face model that downloads once (~1.2 GB) and then runs locally.

For the other backends, copy `.env.example` to `.env` and add the keys you need:
`HF_TOKEN` (free, from huggingface.co/settings/tokens) for the Hugging Face
Inference API, and `OPENAI_API_KEY` for the OpenAI sub-tasks. `.env` is git-ignored
— never commit a key.

## Text summarisation

Condenses lecture transcripts, textbook chapters and papers into revision material.

- **Five output styles** — concise abstract, revision bullet points, study notes,
  exam key takeaways, and a plain-language explanation.
- **Map-reduce for long documents** — input over `CHUNK_WORDS` (900) is split on
  paragraph/sentence boundaries, each chunk is summarised, then the partial summaries
  are summarised into the requested style.
- **Optional focus prompt** to steer the summary at one aspect of the material.

### Backends

Hugging Face is the primary provider, so the sub-task is not gated on OpenAI credits.

| Backend | Model | Key needed | Cost | Styles |
|---|---|---|---|---|
| `hf_local` *(default)* | `sshleifer/distilbart-cnn-12-6` | none | free, runs offline | formatter |
| `hf_api` | `Qwen/Qwen2.5-7B-Instruct` | free `HF_TOKEN` | free monthly credits | model |
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` | ~$0.0003 / run | model |

The **styles** column is the honest caveat: DistilBART is a purpose-built abstractive
summariser and cannot follow an instruction like "give me bullet points". On
`hf_local`, non-default styles are produced by `apply_style()`, which splits the
model's own summary on sentence boundaries and re-lays it out — it never adds or
rewrites content. Only the instruction-tuned backends generate genuinely styled
output. `result["style_applied_by"]` records which happened, and the app says so
under the summary.

```python
from src.summarization import summarize

result = summarize(open("Data/sample_lecture_notes.txt").read(),
                   style="Revision bullet points", target_words=150)
print(result["summary"], result["latency_sec"], result["rougeL"])
```

### Measured comparison

Reproduce with `python scripts/benchmark_summarization.py`, which runs every backend
over the same document with identical settings. Sample notes (831 words), revision
bullet points, target 150 words, CPU:

| Backend | Latency | Tokens | Cost | Words out | Compression | ROUGE-1 | ROUGE-L |
|---|---|---|---|---|---|---|---|
| `hf_local` | 39.7 s | 1315 | $0.00 | 116 | 7.2x | 0.225 | 0.225 |
| `hf_api` | 2.2 s | 1299 | $0.00 | 118 | 7.0x | 0.189 | 0.128 |

**The local SLM scores higher on ROUGE but writes worse summaries** — and that is a
property of the metric, not a ranking. ROUGE here is measured against the *source*,
because the project has no human reference summaries. It therefore rewards reusing
the source's own wording: DistilBART largely stitches together sentences it copied,
while Qwen paraphrases, which costs it ROUGE points despite the output being more
readable and better organised. Read ROUGE as a faithfulness / anti-hallucination
signal, not as a quality score. Judge quality by reading the two outputs.

The 18x latency gap is the other half of the trade: the local model is free and
private but CPU-bound, the API model is fast but spends credits and leaves the
machine.

`Data/sample_lecture_notes.txt` is a ready-made input for the demo.

## LLMOps metrics

Every model call in the application is wrapped by `src/metrics.py`, which appends one
row per invocation to `Data/metrics_log.csv` and surfaces the aggregates in the
"LLMOps metrics dashboard" expander in the app.

| Metric | Meaning |
|---|---|
| Latency (avg, p95) | Wall-clock time per model call |
| Token usage | Prompt + completion tokens, per call and cumulative |
| Cost (USD) | Estimated from published per-token pricing |
| Success rate | Fraction of calls that completed without an exception |
| Quality (ROUGE-1 / ROUGE-L) | Faithfulness of a summary to its source text |
| Compression ratio | Source words per summary word |

Adding metrics to another sub-task is three lines:

```python
from src.metrics import track

with track("question_answering", "gpt-4o-mini") as run:
    response = client.responses.create(...)
    run.set_usage(response.usage.input_tokens, response.usage.output_tokens)
```
