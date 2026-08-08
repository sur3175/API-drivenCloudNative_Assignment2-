# Demo video — speaker notes

Target length **8–10 minutes**. Timings below are cumulative.

Everything in the *Say* lines is factual and checkable against the repo. Nothing here
overstates what the project does — the parts that don't work are called out
deliberately, because a viva examiner will find them anyway and it reads far better
coming from you first.

---

## Before you hit record

```bash
python -m streamlit run app.py -- hf
```

- [ ] `.env` has `HF_API_KEY` set
- [ ] **Delete `data/metrics_log.csv`** so the LLMOps dashboard starts empty and fills
      up live during the demo — this is the single most convincing moment in the video
- [ ] Warm the local models first: run one summarisation on DistilBART and one
      practice-question generation **before recording**. First load downloads ~1.2 GB
      and takes ~40 s; warm it's ~7 s. Otherwise you have a minute of dead air.
- [ ] Open `data/sample_lecture_notes.txt` and `data/sample_biology_notes.txt` in
      tabs ready to copy
- [ ] Browser zoom ~110%, close other tabs (the machine pages badly under memory
      pressure and it will show as lag)

---

## 0:00 — Opening (30 s)

**Say:** "This is our API-driven AI application for the Education domain — an AI study
assistant. A student gives it their own study material, and it generates content from
it, condenses it for revision, answers questions about it, and produces practice
questions to test recall. We chose Natural Language Processing and Computer Vision as
our two categories."

**Say:** "The important design point is that the sub-tasks aren't five unrelated demos
— they all operate on the same student-supplied material and feed one objective."

**Show:** the app's landing view, scroll slowly top to bottom so all sections are seen.

---

## 0:30 — Architecture and providers (1 min)

**Show:** the sidebar reading `Provider: HF`, then briefly `config.py`.

**Say:** "The app takes a provider argument at launch. `config.py` centralises the
credentials and exposes a single `create_client(provider)` factory, so no sub-task
reads `.env` itself. We made Hugging Face the primary provider so the project isn't
gated on OpenAI credits."

**Say:** "Summarisation and question answering also let you pick the model per run,
which is what lets us compare a small local model against a hosted large one without
restarting anything."

---

## 1:30 — Text summarisation (2 min)

**Do:** paste `sample_lecture_notes.txt`, style **Revision bullet points**, model
**HF Qwen2.5-7B**, click Summarise.

**Say while it runs:** "Long documents are handled map-reduce style — split on
paragraph boundaries, summarise each chunk, then summarise the summaries into the
requested style."

**Show:** the metric tiles.

**Say:** "Roughly two seconds, and every run reports its own latency, tokens, cost and
compression ratio."

**Do:** switch model to **HF DistilBART local**, click Summarise again on the same text.

**Say:** "Same document, local small model. Slower — this is CPU — and free. And notice
the caption changed from *style applied by model* to *style applied by formatter*.
DistilBART is a purpose-built summariser and physically cannot follow a style
instruction, so on that backend we lay its own sentences out ourselves. We label which
one happened rather than pretending both are the same thing."

> **If asked about ROUGE:** the local model scores *higher* on ROUGE — 0.225 against
> 0.189 — while writing worse summaries. We have no human reference summaries, so ROUGE
> is measured against the source, which rewards copying the source's wording.
> Extractive output wins on it by construction. We report it as a faithfulness signal,
> not a quality score. **Do not claim the SLM is better because of this number.**

---

## 3:30 — Question answering (1.5 min)

**Do:** same notes in the QA box. Ask *"Why should latency be reported as percentiles
rather than an average?"* with **DistilBERT local** selected.

**Say:** "This is extractive — it selects a span out of your material and cites the
sentence it came from, so it physically cannot hallucinate."

**Do:** switch to **Qwen2.5-7B**, same question.

**Say:** "Generative — reads better, handles answers spread across sentences, but it's
only held to the material by the prompt. So we measure groundedness: the share of the
answer's content words that appear in the source. When we asked it *what is an error
budget*, it scored 0.59 because it added correct but unsourced detail. That's the
metric doing its job."

**Do:** ask *"What is the capital of Portugal?"*

**Say:** "And when the material doesn't cover it, it declines instead of guessing.
For a study assistant that's the behaviour you want."

---

## 5:00 — Fine-tuning (2.5 min) — *the highest-value section*

**Say:** "Requirement 8. We fine-tuned t5-small on SciQ — around 11,700 crowdsourced
science exam questions, each with a support paragraph. School science material, so the
same Education domain."

**Say:** "The task is practice question generation: given an answer and a passage,
write the exam question. That's a genuine study-assistant feature."

**Show:** the results table in the doc or README.

**Say:** "Held out 150 test items. Exact match went from 0 to 13.3%, token F1 from 41.8
to 62.2, ROUGE-L from 38.5 to 56.9."

**Say — this is the line that lands:** "The metric that actually shows what changed is
copy rate. Base t5-small has never seen the `generate question:` prefix, so it just
echoes the passage back — 99.3% of its output words come straight from the input, and
its exact match is zero because it never produces a question at all. After fine-tuning
it produces actual questions."

**Show:** the base-vs-fine-tuned example row:
- reference: *What term in biotechnology means a genetically exact copy of an organism?*
- base: *human cloning is one of the inevitable outcomes of modern biotechnology*
- fine-tuned: *What is a genetically exact copy of an organism developed using techniques associated with biotechnology?*

**Do:** paste `sample_biology_notes.txt` into the Practice Question Generator, generate.

**Say:** "And here it is running live in the app on biology revision notes."

> **Say this yourself, don't wait to be asked:** "We tried fine-tuning for question
> answering first and abandoned it on evidence. Base t5-small already scores 70% exact
> match on SciQ QA, because T5's original pre-training mixture includes SQuAD in exactly
> that prompt format. There was no headroom — our pilot fine-tune actually scored
> slightly worse than base. So we changed to a task the base model genuinely cannot do.
> We kept the QA run in the code as `--task qa` because the negative result is still a
> result."

---

## 7:30 — LLMOps (1 min)

**Do:** open the **LLMOps metrics dashboard** expander.

**Say:** "Every model call in the app goes through one `track()` context manager. It
writes a row per invocation — latency, prompt and completion tokens, estimated cost,
success flag, and a task-specific quality score — and the dashboard aggregates them.
That's seven metrics against the five required."

**Say:** "Note this filled up from the calls we just made in this demo — it's live
instrumentation, not a static table. Adding a new sub-task to it is three lines."

**Point out:** avg *and* p95 latency. "We report p95 because an average hides exactly
the tail a user experiences as the app being broken."

---

## 8:30 — Honesty slide and close (45 s)

**Say:** "What isn't done: image classification is designed and stubbed but not
implemented — the module raises `NotImplementedError` and the app says so rather than
faking it. And text and image generation aren't wired into the metrics layer yet."

**Say:** "On the fine-tuned model, 13% exact match is low in absolute terms. It's a
60-million-parameter model, 1,600 examples, one epoch, on CPU. The size and direction
of the change is the result, not the absolute score. It's also trained on science, so
it degrades on out-of-domain notes."

**Close:** "Everything shown is reproducible from the repo — `scripts/finetune_qa.py`
retrains the model and `scripts/benchmark_summarization.py` regenerates the
summarisation comparison."

---

## Likely viva questions

**"Why two models per sub-task?"**
To make the SLM-vs-LLM trade-off measurable rather than asserted. Local models are free,
private and offline but slower and weaker; hosted models are fast and better but spend
credits and send data off the machine. The metrics layer prices both.

**"Your ROUGE numbers favour the weaker model. Explain."**
Covered above — measured against source, not a human reference, so it rewards
extractive copying. Faithfulness signal, not a quality ranking. *Being the one to raise
this is worth more than being caught by it.*

**"How do you stop the QA sub-task hallucinating?"**
Two ways. The extractive backend can only copy spans, so it's structurally incapable of
it. The generative backend is prompted to answer only from the material and to decline
otherwise, and we measure groundedness to detect when it drifts anyway.

**"Why is exact match only 13%?"**
Small model, small data, one epoch, CPU-only. Question generation also has many valid
phrasings for one passage, so exact match is a harsh metric — token F1 at 62% and
ROUGE-L at 57% are the fairer reads.

**"Why did you write your own ROUGE instead of using a library?"**
To avoid an extra dependency for about thirty lines of code. It's standard ROUGE-1 F1
on unigram overlap and ROUGE-L on longest common subsequence.

**"What would you do with more time?"**
Implement image classification, instrument the remaining two sub-tasks, train longer on
more data with a larger base model, and collect human reference summaries so ROUGE can
measure quality rather than faithfulness.

---

## Recording tips

- Do the local-model runs **warm**; never let the video sit on a spinner for 40 seconds
- If a live call fails on the day, keep going and narrate it — the metrics layer logs
  failures with `success=0`, so a failure is a legitimate thing to show
- Say numbers out loud as they appear on screen; assessors watching at speed will miss
  small text
- Cover screen sharing of `.env` — never show the token

## Division of speaking

Adjust to your group; roughly:

| Section | Minutes | Speaker |
|---|---|---|
| Opening + architecture | 1.5 | «name» |
| Summarisation | 2.0 | «name» |
| Question answering | 1.5 | «name» |
| Fine-tuning | 2.5 | «name» |
| LLMOps + close | 1.75 | «name» |

The brief marks individual contribution, so **every member should speak to the part
they built**.
