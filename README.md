# API-drivenCloudNative_Assignment2-

Designed and developed an interactive API-driven AI application combining NLP and Computer Vision with 5+ AI sub-tasks. Integrated LLM/SLM models via Hugging Face/OpenAI APIs, applied LLMOps with performance metrics, and fine-tuned a domain-specific model using a relevant dataset.

**Domain:** Education
**Categories:** Natural Language Processing + Computer Vision

## Sub-tasks

| # | Sub-task | Category | Module | Model | Status |
|---|----------|----------|--------|-------|--------|
| 1 | Text generation | NLP | `src/text_generation.py` | openai/gpt-oss-120b:cheapest | Done |
| 2 | Text summarisation | NLP | `src/summarization.py` | DistilBART local (SLM) / Qwen2.5-7B via HF API (LLM) | Done |
| 3 | Question answering | NLP | `src/question_answering.py` | DistilBERT SQuAD (SLM) / Qwen2.5-7B via HF API (LLM) | Done |
| 4 | Image generation | CV | `src/image_generation.py` | gpt-image-1 | Done |
| 5 | Image classification | CV | `src/image_classification.py` | ViT-base ImageNet-1k, local + Inference API (SLM) / gpt-4o-mini vision (LLM) | Done |
| 6 | Practice question generation | NLP | `src/practice_questions.py` | t5-small fine-tuned on SciQ | Done |

All five feed one objective: an AI study assistant that generates study material,
condenses it for revision, answers questions on it, and works with diagrams.

## Setup

```bash
pip install -r requirements.txt
```

The app takes a provider argument that selects the client used by text generation
and image generation:

```bash
python -m streamlit run app.py -- hf
```

```bash
python -m streamlit run app.py -- openai
```

Copy `.env.example` to `.env` and add the keys you need: `HF_API_KEY` (free, from
huggingface.co/settings/tokens) for the Hugging Face provider, `OPENAI_API_KEY` for
the OpenAI provider. `.env` is git-ignored — never commit a key.

Summarisation is the exception: its default backend needs **no key and no credits**,
because the model downloads once (~1.2 GB) and then runs locally. It picks its own
backend from the dropdown in its section rather than from the provider argument, so
the LLM-vs-SLM comparison can be run without restarting the app.

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

result = summarize(open("data/sample_lecture_notes.txt").read(),
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

`data/sample_lecture_notes.txt` is a ready-made input for the demo.

## Question answering

Answers student questions **from the study material they provide**, not from the
model's own knowledge. When the material doesn't cover the question, the answer is
`The study material does not cover this.` rather than a confident guess — which is
the behaviour a study assistant needs.

Two backends, deliberately different in kind:

| Backend | Model | How it answers |
|---|---|---|
| `hf_local` (default) | `distilbert-base-cased-distilled-squad` | **Extractive** — selects a span out of your material, so it cannot hallucinate. Cites the sentence it came from. Free, offline, ~0.8 s. |
| `hf_api` | `Qwen/Qwen2.5-7B-Instruct` | **Generative** — reads far better and handles answers spread across several sentences, but is only held to the material by the prompt. ~1 s. |

```python
from src.question_answering import answer_question

r = answer_question("Why report latency as percentiles?", notes, backend="hf_api")
print(r["answer"], r["groundedness"], r["latency_sec"])
```

**Two quality metrics, one per backend kind.** Extractive answers carry the model's
span `confidence`; generative answers carry `groundedness` — the fraction of the
answer's content words that appear in the source. A generative answer that drifts
into the model's own knowledge scores lower, which makes the drift visible in the
metrics rather than invisible in the prose. In testing, "What is an error budget?"
scored 0.59 on the API backend because the model added correct but unsourced detail.

**Known limitation of the extractive backend.** DistilBERT picks one span per
question, and on a long document it sometimes picks from the wrong section — asked
for the CNCF's four practices it returned "unreliable, elastic infrastructure" with
0.89 confidence, an answer from the summary paragraph. High confidence is not
correctness. The generative backend answered the same question correctly. This is a
real SLM-vs-LLM trade-off and is worth demonstrating rather than hiding.

## Fine-tuning (assignment requirement 8)

`scripts/finetune_qa.py` fine-tunes **t5-small** on **SciQ** — 11,679 crowdsourced
science exam questions, each with a support paragraph. School science material, so
the same Education domain as the rest of the application.

```bash
python scripts/finetune_qa.py                 # train + evaluate
python scripts/finetune_qa.py --evaluate-only # re-run the evaluation
```

**Task: practice question generation.** `answer + passage → exam question`. The
fine-tuned model is served by `src/practice_questions.py` and demoed in the app's
Practice Question Generator section.

### Measured before/after

1,600 training examples, 1 epoch, batch 8, CPU, 56 min. Evaluated on 150 held-out
SciQ test items:

| | Exact match | Token F1 | ROUGE-L | Copy rate |
|---|---|---|---|---|
| base t5-small | 0.0% | 41.78% | 38.47% | 99.31% |
| fine-tuned | **13.33%** | **62.20%** | **56.91%** | **90.75%** |
| change | +13.33 | +20.42 | +18.44 | −8.56 |

**Copy rate is the metric that shows what actually changed.** Base t5-small has never
seen the `generate question:` prefix, so it falls back to echoing the passage —
99.3% of its output words come straight from the input, and it scores 0% exact match
because it never produces a question at all. Fine-tuning teaches it the task:

| | Output |
|---|---|
| reference | *What term in biotechnology means a genetically exact copy of an organism?* |
| base | *human cloning is one of the inevitable outcomes of modern biotechnology* |
| fine-tuned | *What is a genetically exact copy of an organism developed using techniques associated with biotechnology?* |

### Why this task, and not question answering

The obvious choice was fine-tuning for grounded QA. **Measuring the baseline first
killed that idea**: base t5-small already scores 70.0% EM / 81.2% F1 on SciQ QA,
because T5's original pre-training mixture includes SQuAD in exactly the
`question: … context: …` format. A pilot fine-tune came out *worse* than base. There
was nothing to demonstrate.

That run is kept as `--task qa` and its numbers in `data/finetuning_results_qa.json`,
because "the base model was already good enough" is a legitimate finding.

### Honest limitations

- **13% exact match is low in absolute terms.** It is a 60M-parameter model, 1,600
  examples, one epoch, on CPU. The direction and size of the change is the result,
  not the absolute score.
- **It is fine-tuned on science.** On `data/sample_biology_notes.txt` it produces
  usable questions; on the cloud-architecture notes it is visibly worse, because that
  is out of its training domain. Demo it on the biology notes.
- **It still degrades on generic terms.** Given a vague answer term it reverts to
  copying a sentence. `looks_like_question()` filters those out and tries the next
  candidate, so roughly 4 in 5 requested questions come back usable.

## LLMOps metrics

Every model call in the application is wrapped by `src/metrics.py`, which appends one
row per invocation to `data/metrics_log.csv` and surfaces the aggregates in the
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
