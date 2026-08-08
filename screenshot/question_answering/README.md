# Question Answering — screenshots

Captured from the running app (`python -m streamlit run app.py -- hf`) at 2x scale
against `data/sample_lecture_notes.txt`.

| File | Shows |
|---|---|
| `01_qa_input_form.png` | The controls: material upload, context box, question, model selector |
| `02_qa_extractive_local_slm.png` | DistilBERT SQuAD, local and extractive — answer span plus the cited source sentence |
| `03_qa_generative_hf_api.png` | Qwen2.5-7B via the Inference API answering the same question generatively |
| `04_qa_declines_when_not_covered.png` | Asked "What is the capital of Portugal?", the assistant declines rather than guessing |

## Notes for whoever writes them up

**2 and 3 are the comparison pair** — same question, same material, extractive SLM vs
generative LLM, metrics visible in both frames.

**Shot 4 is the one worth dwelling on.** Declining to answer is the designed behaviour
for a study assistant, not a failure. Groundedness reads 1.0 on a refusal because there
is nothing unsourced in it.

**The metric label changes between backends.** Extractive answers show `Confidence`
(the model's span probability); generative answers show `Groundedness` (share of the
answer's content words found in the source). They are not the same measurement — don't
present them in one column as if they were.
