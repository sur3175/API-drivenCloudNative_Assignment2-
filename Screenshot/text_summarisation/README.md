# Text Summarisation — screenshots

Captured from the running app (`python -m streamlit run app.py -- hf`) at 2x scale
against `Data/sample_lecture_notes.txt` (831 words), target length 150 words.

| File | Shows |
|---|---|
| `01_summarisation_input_form.png` | The controls: document upload, text area, style, model, length, focus |
| `02_summary_hf_api_bullets.png` | Qwen2.5-7B (Inference API), revision bullet points — 1.88 s, 1314 tokens, $0.00, 6.25x compression, ROUGE-1 0.203 |
| `03_summary_hf_api_study_notes.png` | Same document and model, Study notes style — headed sections with bolded key terms |
| `04_summary_hf_local_slm.png` | DistilBART running locally — 7.58 s, 2 chunks, 7.55x compression, ROUGE-1 0.225 |
| `05_llmops_metrics_dashboard.png` | Aggregates across the three runs: avg 4.06 s, p95 7.58 s, 4045 tokens, $0.0000, 100% success |

## Notes for whoever writes them up

**2 and 4 are the comparison pair** — same document, same target length, LLM vs SLM,
with the metrics visible in each frame.

**The local model was warm.** It reads 7.6 s here; a cold start with the model still
downloading is closer to 40 s. Worth stating which one a quoted figure refers to.

**Watch the ROUGE claim.** The local SLM scores *higher* on ROUGE (0.225 vs 0.203)
while writing visibly worse summaries. ROUGE is measured against the source, not a
human reference, so it rewards copying the source's wording — extractive output wins
on it by construction. Present it as a faithfulness signal, not a quality ranking.

**Note the caption difference.** Shot 2 reads "style applied by model"; shot 4 reads
"style applied by formatter". DistilBART cannot follow style instructions, so on that
backend the non-abstract styles are laid out by `apply_style()` rather than generated.

**Shot 3's metrics are not in frame** — the study-notes output was long enough to push
the caption below the viewport. Use 2 or 4 when you need the numbers.
