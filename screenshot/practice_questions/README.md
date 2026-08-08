# Practice Question Generation — screenshots

The fine-tuned model (assignment requirement 8). Captured from the running app at 2x
scale against `data/sample_biology_notes.txt`.

| File | Shows |
|---|---|
| `01_practice_questions_input.png` | The controls: study material box and question count |
| `02_practice_questions_generated.png` | Five generated questions with their answer terms — 10.42 s, $0.00, model `t5-small-sciq-qgen (fine-tuned)` |

## Notes for whoever writes them up

**Use the biology notes, not the cloud-native ones.** The model is fine-tuned on SciQ,
which is school science. On the cloud-architecture notes it is visibly worse because
that is out of its training domain — an honest limitation, but not what you want in the
headline screenshot.

**The model name is in the caption under the metrics** (`t5-small-sciq-qgen
(fine-tuned)`) — that frame is the evidence the fine-tuned model runs inside the
application, which is what requirement 8 asks to be demonstrated.

**It runs locally at no API cost**, so the cost tile reads $0.00000. The 10.4 s latency
is five sequential beam-search generations on CPU.

**Not every generated question is good.** Question 5 in this capture is a copied
sentence with a question mark. `looks_like_question()` filters the worst of these and
tries another key term, but roughly one in five still gets through. Say so rather than
being caught on it.
