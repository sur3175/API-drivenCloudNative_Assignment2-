# We are using Streamlit which is python framework that lets us quickly turn 
# python code—especially ML/AI models—into an interactive web application.
import sys
import streamlit as sl
from src.text_generation import generate_text

from src.image_generation import generate_image
from src.image_generation import save_image

from config import create_client
from src.summarization import summarize, STYLES, DEFAULT_STYLE
from src.question_answering import answer_question
from src.practice_questions import generate_questions, is_available as pq_available
from src.practice_questions import candidate_answers
from src.image_classification import classify_image, is_implemented as ic_implemented
from src.metrics import summarise_metrics
from config import DATA_DIR

# Validate and process the command-line provider argument.
# The provider must be either "openai" or "hf"; based on the selected
# provider, create_client() initializes the corresponding API client.
if len(sys.argv) < 2:
    raise RuntimeError("Usage: streamlit run app.py -- [openai|hf]")

provider = sys.argv[1].lower()

if provider not in ("openai", "hf"):
    raise RuntimeError("Provider must be 'openai' or 'hf'")

client = create_client(provider)
sl.sidebar.write(f"Provider: {provider.upper()}")

# Application title
sl.title("AI Application - Education")

# ---------------------------------------------------------------------------
# Shared study material
#
# One document, uploaded or pasted once, used by summarisation, question
# answering and practice question generation. Previously each section carried its
# own uploader and its own text box, so the same notes had to be pasted three
# times - and the uploads did not stick, because passing both `value=` and `key=`
# to a Streamlit widget means session state wins on every rerun and overwrites
# what was uploaded. Writing to session state *before* the widget is created is
# the pattern that actually works.
# ---------------------------------------------------------------------------
sl.header("Study Material")
sl.caption(
    "Upload or paste your notes once here. Summarisation, question answering and "
    "the practice question generator all read from this."
)

if "study_material" not in sl.session_state:
    sl.session_state["study_material"] = ""

SAMPLES = {
    "Cloud-native lecture notes": DATA_DIR / "sample_lecture_notes.txt",
    "Biology revision notes": DATA_DIR / "sample_biology_notes.txt",
}

mat_col1, mat_col2 = sl.columns([2, 1])

with mat_col1:
    shared_upload = sl.file_uploader(
        "Upload a document", type=["txt", "md"], key="shared_upload"
    )
    if shared_upload is not None:
        uploaded_text = shared_upload.read().decode("utf-8", errors="ignore")
        # Only overwrite when a genuinely new file arrives, so edits made in the
        # text box below are not wiped out on every rerun.
        if sl.session_state.get("_loaded_file") != shared_upload.name:
            sl.session_state["study_material"] = uploaded_text
            sl.session_state["_loaded_file"] = shared_upload.name
            sl.rerun()

with mat_col2:
    sl.write("Or load a sample:")
    for sample_name, sample_path in SAMPLES.items():
        if sl.button(sample_name, key=f"sample_{sample_name}"):
            sl.session_state["study_material"] = sample_path.read_text(encoding="utf-8")
            sl.session_state["_loaded_file"] = sample_name
            sl.rerun()

study_material = sl.text_area(
    "Study material:",
    height=200,
    placeholder="Paste lecture notes, a textbook section or an article here...",
    key="study_material",
)

if study_material.strip():
    sl.caption(f"{len(study_material.split())} words loaded · used by the sections below")
else:
    sl.caption("No material loaded yet.")

#Text Generator
sl.header("Text Generator")
sl.caption(
    "Generate study material on a topic, or expand on the notes you loaded above."
)

prompt = sl.text_area(
    "Enter your prompt for text generation:", placeholder=""
)

# Ties the sub-task to the shared document rather than leaving it a standalone
# prompt box: with this ticked the model is answering about the student's own
# material, which is what makes it part of one workflow.
tg_use_material = sl.checkbox(
    "Use my study material as context",
    value=bool(study_material.strip()),
    key="tg_use_material",
    disabled=not study_material.strip(),
    help="Sends the document from the Study Material section along with your prompt.",
)

if sl.button("Generate Text"):
    if not prompt.strip():
        sl.warning("Please enter a prompt")
    else:
        final_prompt = prompt
        if tg_use_material and study_material.strip():
            # Truncated so a long document cannot blow the context window.
            final_prompt = (
                "Using the student's study material below, answer this request.\n\n"
                f"Request: {prompt}\n\n"
                "Study material:\n---\n"
                f"{' '.join(study_material.split()[:1200])}\n---"
            )
        with sl.spinner("Generating text..."):
            generated_text = generate_text(client, final_prompt, provider)
        sl.subheader("Generated Text")
        sl.write(generated_text)
        if tg_use_material and study_material.strip():
            sl.caption("Generated from your uploaded study material.")

# Image Generator
sl.header("Image Generator")
sl.caption(
    "Generate a diagram to revise from - either from your own prompt, or as a mind "
    "map built automatically from the study material above."
)

# Build a mind-map prompt out of the loaded document. The key terms come from the
# same extractor the practice question generator uses, so the diagram is drawn from
# what the notes actually emphasise rather than from a generic prompt. This is what
# connects image generation to the rest of the workflow.
mindmap_terms = []
if study_material.strip():
    try:
        mindmap_terms = candidate_answers(study_material, 8)
    except Exception:
        mindmap_terms = []

ig_from_material = sl.checkbox(
    "Build a mind map from my study material",
    value=False,
    key="ig_from_material",
    disabled=not mindmap_terms,
    help="Extracts the key terms from your document and writes the diagram prompt "
         "for you.",
)

suggested_prompt = ""
if ig_from_material and mindmap_terms:
    central, *branches = mindmap_terms
    suggested_prompt = (
        f"A clean, labelled educational mind map diagram about '{central}'. "
        f"Central node '{central}' with clearly labelled branches for: "
        f"{', '.join(branches)}. Flat vector illustration, white background, "
        "high-contrast legible text, no decorative clutter, suitable for revision."
    )
    sl.caption(f"Key terms found: {', '.join(mindmap_terms)}")

image_prompt = sl.text_area(
    "Enter your prompt for image generation:",
    value=suggested_prompt,
    placeholder="",
    key=f"image_prompt_{ig_from_material}",
)

if sl.button("Generate Image"):
    if not image_prompt.strip():
        sl.warning("Please enter a prompt")
    else:
        with sl.spinner("Generating image..."):
            generated_image = generate_image(
                client,
                image_prompt,
                provider
            )
        sl.session_state["generated_image"] = generated_image
        sl.session_state["image_provider"] = provider


# Display generated image
if "generated_image" in sl.session_state:
    sl.subheader("Generated Image")
    sl.image(
        sl.session_state["generated_image"]
    )

    # Save generated image
    sl.subheader("Save Generated Image")
    col1, col2, col3 = sl.columns(3)
    with col1:
        if sl.button("Save as PNG"):
            try:
                filepath = save_image(
                    sl.session_state["generated_image"],
                    sl.session_state["image_provider"],
                    "PNG"
                )
                sl.success(
                    f"Image saved as PNG: {filepath.name}"
                )
            except Exception as e:
                sl.error(
                    f"Failed to save PNG: {e}"
                )

    with col2:
        if sl.button("Save as JPG"):
            try:
                filepath = save_image(
                    sl.session_state["generated_image"],
                    sl.session_state["image_provider"],
                    "JPG"
                )
                sl.success(
                    f"Image saved as JPG: {filepath.name}"
                )

            except Exception as e:
                sl.error(
                    f"Failed to save JPG: {e}"
                )

    with col3:
        if sl.button("Save as JPEG"):
            try:
                filepath = save_image(
                    sl.session_state["generated_image"],
                    sl.session_state["image_provider"],
                    "JPEG"
                )
                sl.success(
                    f"Image saved as JPEG: {filepath.name}"
                )

            except Exception as e:
                sl.error(
                    f"Failed to save JPEG: {e}"
                )


#Question Answering
sl.header("Question Answering")
sl.caption(
    "Ask questions about your study material. Answers are grounded in the material "
    "you provide - if it does not cover the question, the assistant says so rather "
    "than guessing."
)

qa_context = sl.session_state.get("study_material", "")
if qa_context.strip():
    sl.caption(f"Answering from the shared study material ({len(qa_context.split())} words).")
else:
    sl.warning("Load your study material in the **Study Material** section above first.")

question = sl.text_input("Enter your question", key="qa_question")

qa_backend_label = sl.radio(
    "Model",
    [
        "HF DistilBERT SQuAD, local (SLM, extractive)",
        "HF Qwen2.5-7B, Inference API (LLM, generative)",
        "OpenAI gpt-4o-mini (LLM, generative)",
    ],
    key="qa_backend",
    help="The extractive model can only copy spans out of your material, so it "
         "cannot hallucinate. The generative models read better but must be held "
         "to the material by the prompt.",
)
qa_backend = (
    "hf_local" if qa_backend_label.startswith("HF DistilBERT")
    else "hf_api" if qa_backend_label.startswith("HF Qwen")
    else "openai"
)

if sl.button("Generate Answer"):
    try:
        with sl.spinner("Finding the answer..."):
            qa_result = answer_question(question, qa_context, backend=qa_backend)
    except ValueError as exc:
        sl.warning(str(exc))
    except Exception as exc:
        sl.error(f"Question answering failed: {exc}")
    else:
        sl.subheader("Answer")
        if qa_result["answered"]:
            sl.write(qa_result["answer"])
            if qa_result["evidence"]:
                sl.caption(f"From the material: “{qa_result['evidence']}”")
        else:
            sl.info(qa_result["answer"])

        sl.markdown("**Metrics for this run**")
        q1, q2, q3, q4 = sl.columns(4)
        q1.metric("Latency", f"{qa_result['latency_sec']} s")
        q2.metric("Tokens", qa_result["total_tokens"])
        q3.metric("Cost", f"${qa_result['cost_usd']:.5f}")
        q4.metric(
            "Confidence" if qa_result["confidence"] is not None else "Groundedness",
            qa_result["confidence"] if qa_result["confidence"] is not None
            else qa_result["groundedness"],
        )
        sl.caption(
            f"Model: {qa_result['model']} · "
            f"{'extractive' if qa_result['extractive'] else 'generative'} · "
            f"groundedness {qa_result['groundedness']}"
        )


#Text Summarisation
sl.header("Text Summarisation")
sl.caption(
    "Condense lecture notes, textbook chapters or research papers into revision-ready "
    "material. Long documents are split and summarised map-reduce style."
)

source_text = sl.session_state.get("study_material", "")
if source_text.strip():
    sl.caption(f"Summarising the shared study material ({len(source_text.split())} words).")
else:
    sl.warning("Load your study material in the **Study Material** section above first.")

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


#Image Classification
sl.header("Image Classification")
sl.caption(
    "Upload a photo of a diagram, a page of handwritten notes, or a textbook page "
    "and the model labels what it sees, so the material could be routed to the "
    "right sub-task (e.g. notes to the summariser)."
)

if not ic_implemented():
    sl.info(
        "Not built yet. The module skeleton and intended design are in "
        "`src/image_classification.py`; the planned model is "
        "`google/vit-base-patch16-224`, local and API backends, wired to the same "
        "metrics layer as the other sub-tasks."
    )
else:
    ic_uploaded = sl.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], key="ic_upload"
    )

    ic_backend_label = sl.radio(
        "Model",
        [
            "HF ViT-base, local (SLM)",
            "HF ViT-base, Inference API (SLM via API)",
            "OpenAI gpt-4o-mini (LLM, vision)",
        ],
        key="ic_backend",
        help="The Hugging Face ViT model is trained on the 1000 ImageNet object "
             "categories, so it labels things like 'notebook' or 'envelope' - "
             "accurate, but not study-material-aware. The OpenAI backend reads the "
             "image freely and can say 'handwritten notes' or 'circuit diagram' "
             "directly, the same generative-vs-fixed-label trade-off as the QA "
             "extractive/generative backends above.",
    )
    ic_backend = (
        "hf_local" if ic_backend_label.startswith("HF ViT-base, local")
        else "hf_api" if ic_backend_label.startswith("HF ViT-base, Inference")
        else "openai"
    )
    ic_top_k = sl.slider("How many labels", 1, 10, 5, key="ic_top_k")

    if ic_uploaded is not None:
        sl.image(ic_uploaded, width=300)

    if sl.button("Classify image"):
        if ic_uploaded is None:
            sl.warning("Please upload an image first.")
        else:
            try:
                with sl.spinner("Classifying..."):
                    ic_result = classify_image(
                        ic_uploaded.getvalue(), backend=ic_backend, top_k=ic_top_k
                    )
            except ValueError as exc:
                sl.warning(str(exc))
            except Exception as exc:
                sl.error(f"Image classification failed: {exc}")
            else:
                sl.subheader("Predicted labels")
                for item in ic_result["labels"]:
                    sl.write(f"**{item['label']}** — {item['score']:.2f}")

                sl.markdown("**Metrics for this run**")
                i1, i2, i3, i4 = sl.columns(4)
                i1.metric("Latency", f"{ic_result['latency_sec']} s")
                i2.metric("Tokens", ic_result["total_tokens"])
                i3.metric("Cost", f"${ic_result['cost_usd']:.5f}")
                i4.metric("Top-1 confidence", ic_result["confidence"])
                sl.caption(f"Model: {ic_result['model']} · backend: {ic_backend}")


#Practice Question Generator - the fine-tuned model (assignment requirement 8)
sl.header("Practice Question Generator")
sl.caption(
    "Turns study notes into exam-style practice questions, using a t5-small "
    "fine-tuned on the SciQ science-exam dataset. Key terms are picked out of your "
    "notes automatically and the model writes a question for each one."
)

if not pq_available():
    sl.info(
        "The fine-tuned model has not been trained on this machine yet. Run:\n\n"
        "`python scripts/finetune_qa.py --task qgen`"
    )
else:
    pq_context = sl.session_state.get("study_material", "")
    if pq_context.strip():
        sl.caption(
            f"Using the shared study material ({len(pq_context.split())} words). "
            "The model is fine-tuned on science, so the biology sample gives the "
            "best results."
        )
    else:
        sl.warning("Load your study material in the **Study Material** section above first.")
    pq_count = sl.slider("How many questions", 3, 10, 5, key="pq_count")

    if sl.button("Generate practice questions"):
        try:
            with sl.spinner("Writing questions..."):
                pq_result = generate_questions(pq_context, num_questions=pq_count)
        except ValueError as exc:
            sl.warning(str(exc))
        except Exception as exc:
            sl.error(f"Question generation failed: {exc}")
        else:
            sl.subheader("Practice questions")
            for i, item in enumerate(pq_result["questions"], start=1):
                sl.markdown(f"**{i}. {item['question']}**")
                sl.caption(f"Answer: {item['answer']}")

            sl.markdown("**Metrics for this run**")
            p1, p2, p3 = sl.columns(3)
            p1.metric("Latency", f"{pq_result['latency_sec']} s")
            p2.metric("Questions", len(pq_result["questions"]))
            p3.metric("Cost", f"${pq_result['cost_usd']:.5f}")
            sl.caption(f"Model: {pq_result['model']} · runs locally, no API cost")


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
