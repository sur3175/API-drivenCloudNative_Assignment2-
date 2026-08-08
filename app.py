# We are using Streamlit which is python framework that lets us quickly turn 
# python code—especially ML/AI models—into an interactive web application.

import streamlit as sl
from unittest import result
from src.text_generation import generate_text
from src.image_generation import generate_image
from src.summarization import summarize, STYLES, DEFAULT_STYLE
from src.metrics import summarise_metrics

sl.title("AI Application - Education")

#Text Generator
sl.header("Text Generator")

prompt = sl.text_area(
    "Enter your prompt for text generation:",placeholder="" 
)

if sl.button("Generate Text"):
    if not prompt.strip():
        sl.warning("Please enter a prompt")
    else:
        with sl.spinner("Generating text..."):
        
            result = generate_text(prompt)
        sl.subheader("Generated Text")
        sl.write(result)


#Image Generator
sl.header("Image Generator")
prompt = sl.text_area(
    "Enter your prompt for image generation:",placeholder="" 
)
if sl.button("Generate Image") :
    if not prompt.strip():
        sl.warning("Please enter a prompt")
    else:
        with sl.spinner("Generating image..."):
        
            result = generate_image(prompt)
        sl.subheader("Generated Image")
        sl.write(result)

#Question Answering
sl.header("Question Answering")
question = sl.text_input("Enter your question")

if sl.button("Generate Answer") :
    answer = "This is my AI-generated answer"
    sl.write("### Answer")
    sl.write(answer)


#Text Summarisation
sl.header("Text Summarisation")
sl.caption(
    "Condense lecture notes, textbook chapters or research papers into revision-ready "
    "material. Long documents are split and summarised map-reduce style."
)

uploaded = sl.file_uploader(
    "Upload a study document (optional)", type=["txt", "md"], key="summary_upload"
)

default_source = ""
if uploaded is not None:
    default_source = uploaded.read().decode("utf-8", errors="ignore")

source_text = sl.text_area(
    "Text to summarise:",
    value=default_source,
    height=220,
    placeholder="Paste lecture notes, a textbook section or an article here...",
    key="summary_source",
)

col_left, col_right = sl.columns(2)
with col_left:
    style = sl.selectbox("Summary style", list(STYLES), index=list(STYLES).index(DEFAULT_STYLE))
    backend_label = sl.radio(
        "Model",
        [
            "HF DistilBART, local (SLM)",
            "HF Qwen2.5-7B, Inference API (LLM)",
            "OpenAI gpt-4o-mini (LLM)",
        ],
        help="The local Hugging Face model needs no key and consumes no credits. "
             "The Inference API option needs a free HF_TOKEN and is the only "
             "backend that can genuinely follow the output styles.",
    )
with col_right:
    target_words = sl.slider("Approximate summary length (words)", 50, 400, 150, step=25)
    focus = sl.text_input(
        "Focus on a particular aspect (optional)",
        placeholder="e.g. only the evaluation methodology",
    )

BACKEND_BY_LABEL = {
    "HF DistilBART, local (SLM)": "hf_local",
    "HF Qwen2.5-7B, Inference API (LLM)": "hf_api",
    "OpenAI gpt-4o-mini (LLM)": "openai",
}
backend = BACKEND_BY_LABEL[backend_label]

if sl.button("Summarise"):
    if not source_text.strip():
        sl.warning("Please paste some text or upload a document")
    else:
        try:
            with sl.spinner("Summarising..."):
                result = summarize(
                    source_text,
                    style=style,
                    target_words=target_words,
                    backend=backend,
                    focus=focus,
                )
        except ValueError as exc:
            sl.warning(str(exc))
        except Exception as exc:
            sl.error(f"Summarisation failed: {exc}")
        else:
            sl.subheader("Summary")
            sl.write(result["summary"])
            sl.download_button(
                "Download summary", result["summary"], file_name="summary.txt"
            )

            sl.markdown("**Metrics for this run**")
            m1, m2, m3, m4 = sl.columns(4)
            m1.metric("Latency", f"{result['latency_sec']} s")
            m2.metric("Tokens", result["total_tokens"])
            m3.metric("Cost", f"${result['cost_usd']:.5f}")
            m4.metric("Compression", f"{result['compression_ratio']}x")
            sl.caption(
                f"Model: {result['model']} · chunks: {result['chunks']} · "
                f"{result['source_words']} words in, {result['summary_words']} words out · "
                f"ROUGE-1 {result['rouge1']} · ROUGE-L {result['rougeL']} · "
                f"style applied by {result['style_applied_by']}"
            )
            if result["style_applied_by"] == "formatter" and style != DEFAULT_STYLE:
                sl.caption(
                    ":grey[DistilBART cannot follow style instructions, so its own "
                    "sentences were re-laid-out into this shape. Use the Inference "
                    "API backend for model-generated styling.]"
                )


#LLMOps metrics dashboard - aggregates every model call logged to Data/metrics_log.csv
with sl.expander("LLMOps metrics dashboard"):
    overall = summarise_metrics()
    if not overall:
        sl.info("No model calls logged yet. Run a sub-task above to populate the metrics.")
    else:
        d1, d2, d3, d4 = sl.columns(4)
        d1.metric("Total calls", overall["calls"])
        d2.metric("Avg latency", f"{overall['avg_latency_sec']} s")
        d3.metric("Total cost", f"${overall['total_cost_usd']:.4f}")
        d4.metric("Success rate", f"{overall['success_rate'] * 100:.1f}%")
        sl.caption(
            f"p95 latency {overall['p95_latency_sec']} s · "
            f"{overall['total_tokens']} tokens consumed"
        )

        summary_stats = summarise_metrics("summarization")
        if summary_stats:
            sl.markdown("**Summarisation sub-task**")
            sl.json(summary_stats)
