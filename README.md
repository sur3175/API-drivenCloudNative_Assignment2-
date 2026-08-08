# API-drivenCloudNative_Assignment2-

Designed and developed an interactive API-driven AI application combining NLP and Computer Vision with 5+ AI sub-tasks. Integrated LLM/SLM models via Hugging Face/OpenAI APIs, applied LLMOps with performance metrics, and fine-tuned a domain-specific model using a relevant dataset.

**Domain:** Education
**Categories:** Natural Language Processing + Computer Vision

## Sub-tasks

| # | Sub-task | Category | Module | Model | Status |
|---|----------|----------|--------|-------|--------|
| 1 | Text generation | NLP | `src/text_generation.py` | gpt-4o-mini | Done |
| 2 | Text summarisation | NLP | `src/summarization.py` | gpt-4o-mini (LLM) / DistilBART (SLM) | Done |
| 3 | Question answering | NLP | `src/question_answering.py` | — | Pending |
| 4 | Image generation | CV | `src/image_generation.py` | gpt-image-1 | Done |
| 5 | Image classification | CV | `src/image_classification.py` | — | Pending |

All five feed one objective: an AI study assistant that generates study material,
condenses it for revision, answers questions on it, and works with diagrams.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then put your real OPENAI_API_KEY in .env
streamlit run app.py
```

`.env` is git-ignored — never commit the key.

## Text summarisation

Condenses lecture transcripts, textbook chapters and papers into revision material.

- **Five output styles** — concise abstract, revision bullet points, study notes,
  exam key takeaways, and a plain-language explanation.
- **Map-reduce for long documents** — input over `CHUNK_WORDS` (900) is split on
  paragraph/sentence boundaries, each chunk is summarised, then the partial summaries
  are summarised into the requested style.
- **Two backends for the LLM-vs-SLM comparison** — `backend="openai"` uses
  gpt-4o-mini via the OpenAI API; `backend="slm"` runs
  `sshleifer/distilbart-cnn-12-6` locally through Hugging Face transformers, with no
  API key and no per-token cost.
- **Optional focus prompt** to steer the summary at one aspect of the material.

```python
from src.summarization import summarize

result = summarize(open("Data/sample_lecture_notes.txt").read(),
                   style="Revision bullet points", target_words=150)
print(result["summary"], result["latency_sec"], result["rougeL"])
```

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
