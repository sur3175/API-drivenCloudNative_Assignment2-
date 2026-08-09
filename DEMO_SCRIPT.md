# Demo video — speaker notes

Target length **11–12 minutes**. Timings below are cumulative.

**Running order** — the six sub-tasks in sequence, matching the order they appear in
the app, so you scroll straight down the page and never jump around:

1. Text generation
2. Image generation
3. Question answering
4. Text summarisation
5. Image classification
6. Practice question generation (the fine-tuned model)

then the LLMOps dashboard and a short limitations close.

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
- [ ] **Warm every local model before recording.** In the running app, do one
      summarisation on DistilBART, one question answer on DistilBERT, one image
      classification on ViT, and one practice-question generation. Measured cold-load
      times on this machine: DistilBERT **106 s**, DistilBART ~40 s. Warm they are
      1–10 s. Skip this and you will have minutes of dead air.
- [ ] Have a diagram or a photo of handwritten notes ready for image classification
- [ ] Browser zoom ~110%, close other tabs (the machine pages badly under memory
      pressure and it will show as lag)

**No copy-pasting needed:** the Study Material section at the top has one-click
buttons for both sample documents, and everything downstream reads from it.

**Backend choice on the day:** for the smoothest video use the **HF Inference API**
backends (~1–3 s). Switch to the local models only for the deliberate SLM-vs-LLM
comparison, and only after warming them.

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

## 0:30 — Architecture and providers (45 s)

**Show:** the sidebar reading `Provider: HF`, then briefly `config.py`.

**Say:** "The app takes a provider argument at launch. `config.py` centralises the
credentials and exposes a single `create_client(provider)` factory, so no sub-task
reads `.env` itself. We made Hugging Face the primary provider so the project isn't
gated on OpenAI credits."

**Say:** "Most sub-tasks also let you pick the model per run, which is what lets us
compare a small local model against a hosted large one without restarting anything."

---

## 1:15 — Study Material: load once, use everywhere (30 s)

**Do:** click **Cloud-native lecture notes**.

**Say:** "The document is loaded once here. Question answering, summarisation and the
practice question generator all read from it — you don't paste the same notes into
three boxes. You can upload a `.txt`/`.md` file, paste directly, or load one of our
two samples."

**Point out:** the word count under the box, and the same count echoed in each section
below. That's the shared state working.

---

# The six sub-tasks, in order

## 1:45 — Sub-task 1: Text generation (45 s)

**Do:** prompt it with something a student would actually ask, e.g.
*"Explain the difference between aerobic and anaerobic respiration for a GCSE student."*

**Say:** "This is the entry point of the workflow. A student generates study material
on a topic, and everything downstream operates on material like this. It runs through
whichever provider the app was launched with."

---

## 2:30 — Sub-task 2: Image generation (45 s)

**Do:** prompt e.g. *"A simple labelled diagram of a plant cell."* Generate, then show
the **Save as PNG** button.

**Say:** "The same study assistant produces the visual side — a supporting diagram for
the topic being revised. Generated images render in the app and save to `data/` as
PNG, JPG or JPEG."

> If image generation is unavailable on the day (gpt-image-1 needs a verified OpenAI
> org and paid credits), say so plainly and show a previously generated image. Do not
> spend demo time debugging it.

---

## 3:15 — Sub-task 3: Question answering (1.5 min)

No pasting — the section already reads the shared material. Point that out.

**Do:** ask *"Why should latency be reported as percentiles rather than an average?"*
with **DistilBERT local** selected (warmed beforehand).

**Say:** "This is extractive — it selects a span out of your material and cites the
sentence it came from, so it physically cannot hallucinate."

**Do:** switch to **Qwen2.5-7B**, same question.

**Say:** "Generative — reads better, handles answers spread across sentences, but it's
only held to the material by the prompt. So we measure groundedness: the share of the
answer's content words that appear in the source. When we asked it *what is an error
budget*, it scored 0.59 because it added correct but unsourced detail. That's the
metric doing its job."

**Point out:** the metric tile label changes between backends — `Confidence` for the
extractive model, `Groundedness` for the generative one. Different measurements, not
one number.

**Do:** ask *"What is the capital of Portugal?"*

**Say:** "And when the material doesn't cover it, it declines instead of guessing.
For a study assistant that's the behaviour you want."

---

## 4:45 — Sub-task 4: Text summarisation (2 min)

**Do:** style **Revision bullet points**, model **HF Qwen2.5-7B**, click Summarise.
Same shared document — nothing to re-paste.

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

## 6:45 — Sub-task 5: Image classification (1 min)

**Do:** upload a diagram or a photo of handwritten notes. Run **ViT local** first.

**Say:** "Vision Transformer, ImageNet-1k — a thousand object classes, running locally
with no key and no cost. It gives us the top labels with their confidence scores."

**Do:** switch to **OpenAI gpt-4o-mini vision**, same image.

**Say — this is the interesting bit:** "The ImageNet model can only answer with one of
its thousand object classes, so on a page of handwritten notes it reaches for the
nearest object — something like *envelope* or *menu*. The generative model isn't
restricted to a fixed label set, so it can actually say *handwritten notes* or
*circuit diagram*. It's the same small-model-versus-large-model trade-off we showed in
question answering, in a different modality."

**Say:** "In the study-assistant workflow this is the router: it tells us whether the
student photographed a diagram or a page of notes, so we know which sub-task to send
it to."

---

## 7:45 — Sub-task 6: Fine-tuning (2.5 min) — *the highest-value section*

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

**Do:** scroll up to **Study Material**, click **Biology revision notes**, then scroll
to the Practice Question Generator and generate.

**Say:** "And here it is running live in the app. One click swaps the shared document,
and every section picks it up. The model is fine-tuned on science, so we're giving it
science notes."

> **Say this yourself, don't wait to be asked:** "We tried fine-tuning for question
> answering first and abandoned it on evidence. Base t5-small already scores 70% exact
> match on SciQ QA, because T5's original pre-training mixture includes SQuAD in exactly
> that prompt format. There was no headroom — our pilot fine-tune actually scored
> slightly worse than base. So we changed to a task the base model genuinely cannot do.
> We kept the QA run in the code as `--task qa` because the negative result is still a
> result."

---

## 10:15 — LLMOps (1 min)

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

## 11:15 — Honesty slide and close (45 s)

**Say:** "What isn't done: text generation and image generation aren't wired into the
metrics layer yet, so the dashboard covers four of the six sub-tasks rather than all
six."

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
| Opening, architecture, shared material | 1.75 | «name» |
| 1. Text generation + 2. Image generation | 1.5 | «name» |
| 3. Question answering | 1.5 | «name» |
| 4. Text summarisation | 2.0 | «name» |
| 5. Image classification | 1.0 | «name» |
| 6. Fine-tuning | 2.5 | «name» |
| LLMOps + close | 1.75 | «name» |

The brief marks individual contribution, so **every member should speak to the part
they built**.
